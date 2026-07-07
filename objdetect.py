"""
Object detection pre-filter using the Hailo-8L NPU.

Prefers a MegaDetector (YOLOv9-c, class-agnostic animal/person/vehicle) HEF when
present (models/ dir first, then /usr/share/hailo-models/), so daytime wildlife
beyond COCO's handful of classes (squirrels, deer, foxes, raccoons, ...) is
detected. Falls back to the COCO YOLOv8s HEF, then to TFLite on CPU.

Runs on the Hailo NPU for near-zero CPU overhead. Falls back to TFLite on CPU
if no HEF model is found. Skips classification entirely if no animal detected.

analyze_frame() additionally returns the best animal bounding box (normalized
x1,y1,x2,y2) so callers can classify a bird-centered crop instead of the full
frame — the bird fills the classifier input instead of ~5% of it, which
dramatically improves species confidence.

COCO fallback HEF sourced from hailo-tappas-core: /usr/share/hailo-models/yolov8s_h8l.hef
"""

import logging
import threading
import numpy as np
from pathlib import Path
from PIL import Image

log = logging.getLogger("perchsentry")

# A MegaDetector HEF takes priority over the COCO fallbacks below. It is
# class-agnostic (animal/person/vehicle) so it catches wildlife COCO never
# learned (squirrels, deer, foxes, ...). Checked in the app's models/ dir first
# (drop-in override), then the shared system model dir alongside the COCO HEF.
MEGADETECTOR_HEFS = [
    Path(__file__).parent / "models" / "megadetector_h8l.hef",
    Path("/usr/share/hailo-models/megadetector_h8l.hef"),
]

HEF_SEARCH_PATHS = [
    "/usr/share/hailo-models/yolov8s_h8l.hef",
    "/usr/share/hailo-models/yolov8m_h8l.hef",
    "/usr/share/hailo-models/yolov8n.hef",
    "/usr/share/hailo-models/yolov8s.hef",
]

# MegaDetector class index -> label. Only class 0 (animal) is of interest; person
# and vehicle are intentionally ignored so passers-by never trigger a capture.
MEGADETECTOR_LABELS = {0: "animal"}

TFLITE_MODEL = Path(__file__).parent / "models" / "detect.tflite"
TFLITE_LABELS = Path(__file__).parent / "models" / "labelmap.txt"

ANIMAL_CLASSES = {"bird", "cat", "dog", "horse", "sheep", "cow", "bear", "zebra", "giraffe"}

_COCO_LABELS = {
    0: "person", 14: "bird", 15: "cat", 16: "dog", 17: "horse",
    18: "sheep", 19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
}

_backend = None
# "megadetector" or "coco" -- selects output class map + preprocessing.
_model_kind = None
_hailo_infer = None
# Keeps the VDevice (and HEF) alive for the process lifetime. Without this,
# `target` in _init_hailo is garbage-collected when the function returns —
# the closure only captures network_group — which releases the device and
# makes every subsequent inference fail with HailoRTStatusException: 8.
_hailo_keepalive = None
_tflite_interp = None
_tflite_labels = None
# The detector thread and the nightly slow-mo verifier share this pipeline;
# InferVStreams on a single network group is not safe to enter concurrently.
_infer_lock = threading.Lock()


def _find_hef():
    for p in MEGADETECTOR_HEFS:
        if p.exists():
            return str(p)
    for p in HEF_SEARCH_PATHS:
        if Path(p).exists():
            return p
    if Path("/usr/share/hailo-models").exists():
        found = list(Path("/usr/share/hailo-models").glob("yolo*.hef"))
        if found:
            return str(found[0])
    return None


def _letterbox(img, size, pad=114):
    """Resize `img` (PIL RGB) into a `size`x`size` square, preserving aspect
    ratio and padding the remainder with `pad`. Returns (uint8 HWC array, scale,
    pad_x, pad_y) so detections can be mapped back to the original image."""
    orig_w, orig_h = img.size
    scale = min(size / orig_w, size / orig_h)
    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
    canvas = Image.new("RGB", (size, size), (pad, pad, pad))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(img.resize((new_w, new_h)), (pad_x, pad_y))
    # np.array (not asarray) forces a writeable copy — HailoRT's infer() rejects
    # the read-only buffer that PIL images expose.
    return np.array(canvas, dtype=np.uint8), scale, pad_x, pad_y


def _init_hailo(hef_path):
    global _hailo_infer, _hailo_keepalive, _model_kind
    try:
        from hailo_platform import (HEF, VDevice, HailoStreamInterface,
            InferVStreams, ConfigureParams, InputVStreamParams,
            OutputVStreamParams, FormatType)
        hef = HEF(hef_path)
        params = VDevice.create_params()
        target = VDevice(params)
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_groups = target.configure(hef, configure_params)
        network_group = network_groups[0]
        network_group_params = network_group.create_params()
        input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        input_info = hef.get_input_vstream_infos()[0]
        h, w = input_info.shape[0], input_info.shape[1]
        kind = "megadetector" if "megadetector" in Path(hef_path).name.lower() else "coco"

        def infer(image_path):
            img = Image.open(image_path).convert("RGB")
            orig_w, orig_h = img.size
            if kind == "megadetector":
                # MegaDetector expects letterbox+pad-114; keep the transform so
                # output boxes can be un-letterboxed back to the original frame.
                arr, scale, pad_x, pad_y = _letterbox(img, w)
                meta = {"scale": scale, "pad_x": pad_x, "pad_y": pad_y,
                        "orig_w": orig_w, "orig_h": orig_h, "size": w}
            else:
                # COCO YOLOv8 path uses a plain (stretch) resize, as before.
                arr = np.array(img.resize((w, h)), dtype=np.uint8)
                meta = None
            data = {input_info.name: np.expand_dims(arr, 0)}
            with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as pipeline:
                with network_group.activate(network_group_params):
                    return pipeline.infer(data), meta

        _model_kind = kind
        _hailo_keepalive = (target, hef)
        _hailo_infer = (infer, w, h)
        log.info(f"Hailo-8L object detection ready: {Path(hef_path).name} ({w}x{h}) [{kind}]")
        return True
    except Exception as e:
        log.warning(f"Hailo init failed: {e}")
        return False


def _init_tflite():
    global _tflite_interp, _tflite_labels
    if not TFLITE_MODEL.exists():
        return False
    try:
        from ai_edge_litert.interpreter import Interpreter
        _tflite_interp = Interpreter(model_path=str(TFLITE_MODEL))
        _tflite_interp.allocate_tensors()
        if TFLITE_LABELS.exists():
            _tflite_labels = [l.strip() for l in TFLITE_LABELS.read_text().splitlines()]
        log.info("TFLite object detection fallback loaded")
        return True
    except Exception as e:
        log.warning(f"TFLite object detection load failed: {e}")
        return False


def _init():
    global _backend
    hef = _find_hef()
    if hef and _init_hailo(hef):
        _backend = "hailo"
    elif _init_tflite():
        _backend = "tflite"
    else:
        log.info("No object detection model available — pre-filter disabled")
        _backend = "passthrough"


def _label_for(cls_id):
    """Map a class index to an animal label, or None to ignore it. Depends on
    which model is loaded (MegaDetector's 3 classes vs COCO's 80)."""
    if _model_kind == "megadetector":
        return MEGADETECTOR_LABELS.get(cls_id)  # None for person(1)/vehicle(2)
    label = _COCO_LABELS.get(cls_id, "")
    return label if label in ANIMAL_CLASSES else None


def _map_box(x1, y1, x2, y2, meta):
    """Convert an NMS box (normalized 0-1 in the network's square input) to a
    normalized (x1,y1,x2,y2) box in the ORIGINAL image. With `meta` (letterbox)
    the padding/scale is undone; without it (plain resize) coords map directly."""
    def clamp(v):
        return max(0.0, min(1.0, v))
    if meta is None:
        return (clamp(x1), clamp(y1), clamp(x2), clamp(y2))
    size, scale = meta["size"], meta["scale"]
    ow, oh = meta["orig_w"], meta["orig_h"]
    px, py = meta["pad_x"], meta["pad_y"]
    return (clamp((x1 * size - px) / scale / ow),
            clamp((y1 * size - py) / scale / oh),
            clamp((x2 * size - px) / scale / ow),
            clamp((y2 * size - py) / scale / oh))


def _hailo_boxes(image_path, min_confidence):
    """All animal detections as [{'label', 'score', 'box': (x1,y1,x2,y2) 0-1}]."""
    infer_fn, w, h = _hailo_infer
    with _infer_lock:
        outputs, meta = infer_fn(image_path)
    dets = []
    for key, tensor in outputs.items():
        arr = tensor[0]
        # NMS-postprocessed HEF (yolov8s_h8l / megadetector): a list with one
        # entry per class, each an (N, 5) array of [y1, x1, y2, x2, score],
        # coordinates normalized 0-1 in the (letterboxed) model input.
        if isinstance(arr, list):
            for cls_id, cls_dets in enumerate(arr):
                label = _label_for(cls_id)
                if label is None:
                    continue
                for det in cls_dets:
                    score = float(det[4])
                    if score >= min_confidence:
                        box = _map_box(float(det[1]), float(det[0]),
                                       float(det[3]), float(det[2]), meta)
                        dets.append({"label": label, "score": score, "box": box})
        # Raw (no-NMS) HEF: flat (N, 6) array of [x1, y1, x2, y2, conf, cls]
        # in model-input pixels (COCO fallback only).
        elif hasattr(arr, "ndim") and arr.ndim == 2:
            for det in arr:
                if len(det) >= 6:
                    score = float(det[4])
                    cls_id = int(det[5])
                    label = _label_for(cls_id)
                    if score >= min_confidence and label is not None:
                        dets.append({
                            "label": label, "score": score,
                            "box": (max(0.0, float(det[0]) / w), max(0.0, float(det[1]) / h),
                                    min(1.0, float(det[2]) / w), min(1.0, float(det[3]) / h)),
                        })
    return dets


def analyze_frame(image_path, min_confidence=0.3):
    """Detect animals in a frame.

    Returns {"supported": bool, "has_animal": bool,
             "box": (x1,y1,x2,y2) normalized or None,
             "label": str or None, "score": float}.
    "supported" is True only on the Hailo backend (boxes available).
    Fail-open on errors: has_animal True, box None.
    """
    global _backend
    if _backend is None:
        _init()

    if _backend == "hailo":
        try:
            dets = _hailo_boxes(image_path, min_confidence)
        except Exception as e:
            log.debug(f"Hailo detection error (fail-open): {type(e).__name__}: {e}")
            return {"supported": True, "has_animal": True, "box": None,
                    "label": None, "score": 0.0}
        if not dets:
            return {"supported": True, "has_animal": False, "box": None,
                    "label": None, "score": 0.0}
        # Prefer bird boxes, then highest score
        dets.sort(key=lambda d: (d["label"] == "bird", d["score"]), reverse=True)
        best = dets[0]
        return {"supported": True, "has_animal": True, "box": best["box"],
                "label": best["label"], "score": best["score"]}

    if _backend == "tflite":
        return {"supported": False,
                "has_animal": _tflite_detect(image_path, min_confidence),
                "box": None, "label": None, "score": 0.0}

    return {"supported": False, "has_animal": True, "box": None,
            "label": None, "score": 0.0}


def contains_bird(image_path, min_confidence=0.3):
    return analyze_frame(image_path, min_confidence)["has_animal"]


def _tflite_detect(image_path, min_confidence):
    try:
        inp = _tflite_interp.get_input_details()[0]
        out = _tflite_interp.get_output_details()
        h, w = inp["shape"][1], inp["shape"][2]
        img = np.array(Image.open(image_path).resize((w, h))).astype(np.uint8)
        _tflite_interp.set_tensor(inp["index"], img[np.newaxis])
        _tflite_interp.invoke()
        classes = _tflite_interp.get_tensor(out[1]["index"])[0].astype(int)
        scores = _tflite_interp.get_tensor(out[2]["index"])[0]
        for cls, score in zip(classes, scores):
            if score < min_confidence:
                continue
            label = _tflite_labels[cls + 1] if _tflite_labels and cls + 1 < len(_tflite_labels) else ""
            if label.lower() in ANIMAL_CLASSES:
                return True
        return False
    except Exception as e:
        log.debug(f"TFLite detection error: {e}")
        return True
