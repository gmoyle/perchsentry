"""SpeciesNet wildlife classifier (Google cameratrapai), run on the Pi CPU via
ONNX Runtime. Parallel to classify.py (the TFLite bird classifier), but this
one names the *species* of a non-bird animal crop that MegaDetector found.

The exported graph does its own Efficient-Net rescaling, so preprocessing is
just: RGB, EXIF-transpose, resize to 480x480, /255 -> float32 NHWC. See
speciesnet_handoff_notes.md for the full I/O spec this is written against.
"""

import logging
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

MODEL_PATH = Path(__file__).parent / "models" / "speciesnet_classifier.onnx"
LABELS_PATH = Path(__file__).parent / "models" / "speciesnet_labels.txt"

INPUT_SIZE = 480

# Taxonomy strings for the non-species outputs — treat these as "not a named
# animal" so callers fall back to the generic MegaDetector label instead of
# reporting a nonsense sighting.
NON_SPECIES = {"blank", "no cv result", "animal", "unknown", "vehicle"}

log = logging.getLogger("perchsentry")

_session = None
_labels = None


def common_name(label):
    """Last field of a `uuid;class;order;family;genus;species;common name`
    taxonomy string. Falls back to the whole label if it has no common name."""
    parts = label.split(";")
    name = parts[-1].strip() if parts else ""
    return name or label.strip()


def load_labels():
    global _labels
    if _labels is None:
        _labels = [l.strip() for l in LABELS_PATH.read_text().splitlines() if l.strip()]
    return _labels


def load_session():
    """Build (and cache) the ONNX Runtime session. Raises if the model file is
    missing or onnxruntime can't load it — the caller decides how to degrade."""
    global _session
    if _session is None:
        import onnxruntime as ort

        _session = ort.InferenceSession(
            str(MODEL_PATH), providers=["CPUExecutionProvider"]
        )
    return _session


def _softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _preprocess(image_path):
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]  # NHWC [1, 480, 480, 3]


def classify_image(image_path, session=None, labels=None):
    """Top-1 SpeciesNet prediction for a crop. Returns
    {"label": full taxonomy string, "species": common name,
     "confidence": softmax prob, "index": class index}."""
    if session is None:
        session = load_session()
    if labels is None:
        labels = load_labels()

    inp = _preprocess(image_path)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: inp})[0][0]
    probs = _softmax(logits)

    top = int(np.argmax(probs))
    label = labels[top] if top < len(labels) else "unknown"
    return {
        "label": label,
        "species": common_name(label),
        "confidence": float(probs[top]),
        "index": top,
    }


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        frames = sorted(Path(__file__).parent.glob("timelapse/*.jpg"))
        if not frames:
            print("No image given and no timelapse frames to test on.")
            sys.exit(1)
        path = frames[-1]

    r = classify_image(path)
    print(f"Image: {path}")
    print(f"Top-1: {r['species']} ({r['confidence']:.1%})  [{r['label']}]")
