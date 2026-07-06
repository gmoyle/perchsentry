# Wildlife detection model — build & deploy runbook (MegaDetector → Hailo-8L)

**Audience:** an autonomous agent running on a Windows PC (via WSL2) that will
compile the model. This document is fully self-contained — you do not need any
outside context. Produce the deliverables in the final section and hand them
back to the Raspberry Pi.

---

## 1. Objective

The "BirdBuddy" system is a Raspberry Pi 5 bird camera with a **Hailo-8L** AI
accelerator. Its current object detector (`yolov8s_h8l.hef`, COCO) only
recognizes ~10 animal types — notably **not** squirrels, deer, foxes, raccoons,
etc. We want to detect **any** animal in daytime footage.

**MegaDetector** (Microsoft) is a class-agnostic camera-trap detector that
finds `animal / person / vehicle` regardless of species. Your job: compile a
compact MegaDetector to a Hailo `.hef` that will run on this Pi. Naming the
species (squirrel vs deer) is a *later* project step and is out of scope here —
just get "animal" detection working.

## 2. Hard constraints (read first)

- **Target device:** Raspberry Pi 5 + **Hailo-8L**, Hailo architecture name
  **`hailo8l`** (note the trailing `l` — it is the 13-TOPS "Lite" part, not the
  26-TOPS `hailo8`). Compiling for the wrong arch yields a `.hef` that will not
  run.
- **Runtime version pin:** the Pi runs **HailoRT 4.20.0**. Install a **Hailo
  Dataflow Compiler (DFC) from the release that pairs with HailoRT 4.20** (the
  matching Hailo AI Software Suite). A `.hef` from a mismatched DFC major/minor
  can fail to load on the Pi. Confirm the pairing on the Hailo Developer Zone
  release notes.
- **No Hailo hardware is needed on the PC.** Compilation is pure CPU. The NPU
  only matters for running the model, which happens on the Pi.

## 3. PC prerequisites

- **x86-64** Windows (Intel/AMD). ARM Windows (e.g. Snapdragon) will not work.
- **16 GB RAM minimum, 32 GB recommended.** The optimizer is memory-hungry.
- **~30 GB free disk.**
- Windows 10/11 with virtualization enabled (for WSL2).

## 4. Set up WSL2 + Ubuntu

In an **Administrator PowerShell**:

```powershell
wsl --install -d Ubuntu-22.04
```

Reboot if prompted, then launch "Ubuntu 22.04" and create a user.

Give WSL enough memory — create `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=24GB
swap=16GB
processors=8
```

Then in PowerShell: `wsl --shutdown`, and reopen Ubuntu.

Inside Ubuntu, install basics:

```bash
sudo apt update && sudo apt install -y python3.10 python3.10-venv python3-pip \
    build-essential ffmpeg git wget unzip
```

## 5. Install the Hailo Dataflow Compiler (DFC)

1. Create a free account at the **Hailo Developer Zone** (hailo.ai → Developer
   Zone / Software Downloads).
2. Download the **Dataflow Compiler** wheel (or the Hailo AI Software Suite)
   for the release matching **HailoRT 4.20**. There are two ways:
   - **Docker (simplest):** pull the "Hailo AI Software Suite" Docker image and
     run it — the DFC, Model Zoo, and tutorials are preinstalled.
   - **pip:** create a venv and install the DFC wheel:
     ```bash
     python3.10 -m venv ~/hailo && source ~/hailo/bin/activate
     pip install --upgrade pip
     pip install hailo_dataflow_compiler-<version>-cp310-cp310-linux_x86_64.whl
     ```
3. Also install the **Hailo Model Zoo** (`hailo_model_zoo`) from the same
   release — it provides YOLO parsing/NMS config templates you will reuse:
   ```bash
   git clone https://github.com/hailo-ai/hailo_model_zoo.git
   pip install -e hailo_model_zoo
   ```
4. Sanity check: `hailo --version` should report the DFC version. Run
   `hailo tutorial` to open the official DFC Jupyter notebooks — **use these as
   the authoritative reference for exact command syntax on your DFC version**,
   as flags change between releases.

## 6. Get MegaDetector v6 and export to ONNX

Use **MegaDetector v6, a COMPACT variant** — it is YOLOv9/YOLOv10-based and
maps cleanly onto Hailo's supported ops (v5 is YOLOv5x6 at 1280px and is a poor
fit — do not use v5).

```bash
pip install pytorch-wildlife    # Microsoft's PytorchWildlife / MegaDetector
```

- Load MegaDetector v6 and export the detector to **ONNX at 640×640, opset 11–13**.
  Prefer the smallest v6 variant that exports cleanly, e.g. **`MDV6-yolov9-c`**
  (fallback: `MDV6-yolov10`). Follow the current PytorchWildlife export example
  (`microsoft/CameraTraps` repo → MegaDetector v6 / ONNX export docs) for the
  exact API, since the call signature changes between releases.
- **Record the class index → label map** from the model. MegaDetector uses
  three classes; confirm the exact order (commonly `0=animal, 1=person,
  2=vehicle`). The Pi needs this map.
- Verify the ONNX: input `1×3×640×640`, a single image input, detection outputs.

## 7. Calibration images (for int8 quantization)

Quantization needs **~256–1024 representative images**. Quality here directly
affects detection accuracy.

- **Primary (v1):** download a public outdoor/camera-trap sample — e.g. a few
  hundred images from **LILA.science** (Caltech Camera Traps / Snapshot
  Serengeti samples) or COCO images containing animals + outdoor scenes. Put
  them in `~/calib/`.
- **Optional real-camera boost:** BirdBuddy's own footage improves accuracy for
  this exact scene. If provided a clip (e.g. a downloaded slow-mo file such as
  `slowmo_20260630_153447.mp4`), extract frames and add them to `~/calib/`:
  ```bash
  ffmpeg -i "slowmo_20260630_153447.mp4" -vf fps=5 ~/calib/bb_%04d.jpg
  ```
  (A single short clip is low-diversity, so use it *in addition to* the public
  set, not instead of it.)

The DFC resizes calibration images to the model input; JPEGs of any size are
fine.

## 8. Compile: ONNX → HAR → quantized HAR → HEF

The DFC flow has three stages. **Bake the NMS post-process into the HEF** so the
Pi gets clean bounding boxes (matching how the existing `yolov8s_h8l.hef` works
— output type `HAILO NMS BY CLASS`, UINT8 input, FLOAT32 output). Representative
CLI (confirm exact flags against your DFC version's `hailo tutorial` notebooks):

```bash
# 8a. Parse ONNX → HAR, targeting the Lite chip
hailo parser onnx megadetector.onnx \
    --hw-arch hailo8l \
    --har-path megadetector.har

# 8b. Optimize/quantize with the calibration set, adding YOLO NMS post-process.
#     Use a model script (.alls) based on the Model Zoo's yolov8/yolov9 NMS
#     example, edited for MegaDetector's class count (3) and your output-layer
#     names. See hailo_model_zoo/cfg/networks/yolov9c.yaml + its .alls as a
#     template for the nms_postprocess() config.
hailo optimize megadetector.har \
    --hw-arch hailo8l \
    --calib-set-path ~/calib/ \
    --model-script megadetector_nms.alls \
    --output-har-path megadetector_optimized.har

# 8c. Compile → HEF
hailo compiler megadetector_optimized.har \
    --hw-arch hailo8l \
    --output-dir .
# → produces megadetector.hef ; rename it:
mv megadetector.hef megadetector_h8l.hef
```

Notes:
- The **NMS config** (step 8b) is the fiddly part. The cleanest path is to copy
  the Hailo Model Zoo's YOLOv9-c `.alls`/NMS JSON and change: number of classes
  → 3, and the output layer names → those in your ONNX. If baking NMS proves
  hard, a fallback is to compile **without** NMS (raw output) and note that
  clearly in the deliverables — the Pi can do NMS on-CPU, but in-HEF NMS is
  strongly preferred.
- If `hailomz` (Model Zoo CLI) is available, `hailomz compile` with a custom
  YAML pointing at your ONNX + calib set can do parse+optimize+compile in one
  step — either approach is fine.

## 9. Verify before handing back

```bash
hailortcli parse-hef megadetector_h8l.hef
```

Confirm:
- `Architecture HEF was compiled for: HAILO8L`
- Input vstream ≈ `640x640x3`, UINT8
- An NMS output op (if you baked NMS) with **3 classes**

(You cannot do a full inference run without a Hailo device; `parse-hef` is the
static check that matters.)

## 10. DELIVERABLES — hand these back to the Pi

Place all of these somewhere the Pi can retrieve (or attach them):

1. **`megadetector_h8l.hef`** — the compiled model.
2. **Full `hailortcli parse-hef` output** — input/output vstream names, the NMS
   op config, class count.
3. **Class index → label map** used (e.g. `0=animal, 1=person, 2=vehicle`).
4. **Preprocessing details** — input size, letterbox vs plain resize, and any
   normalization (0–255 vs 0–1). State whether NMS is baked in or raw.
5. **Notes on any deviations** — variant used, whether NMS was baked, DFC
   version, and anything that differed from this runbook.

With these, the Pi's `objdetect.py` will get a MegaDetector backend written
against the real tensor layout (reusing its existing NMS-by-class parsing), and
detections route into the already-built "Animals" track. Squirrels/deer/foxes
then start getting caught in daylight.

---

## Appendix: what happens on the Pi afterward (context, not your task)

- `objdetect.py` gains a MegaDetector backend that reports `has_animal` + box
  with a generic `"animal"` label, replacing the COCO `yolov8s` pre-filter.
- The existing "Animals" track (gallery filter, stats, slow-mo recording,
  settings) is reused unchanged — only what the detector reports changes.
- Birds still go to the dedicated bird species classifier (owls included).
- Trade-off: specific COCO labels (cat/dog/bear) become generic "animal" until a
  later phase adds a species classifier (SpeciesNet / DeepFaune).
