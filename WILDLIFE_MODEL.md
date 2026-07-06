# Phase 1 — Wildlife detection model (MegaDetector on Hailo-8L)

Goal: detect **any** animal (squirrel, deer, fox, raccoon, …) in daytime, not
just the ~10 COCO animals the current `yolov8s_h8l.hef` knows. MegaDetector is
class-agnostic ("animal / person / vehicle"), so it catches everything; naming
the species is a later step (Phase 2, a species classifier).

## Why this can't be done on the Pi

Compiling any model to a Hailo `.hef` needs the **Hailo Dataflow Compiler
(DFC)**, which runs **only on x86-64 Linux**. The Pi (aarch64) has just the
runtime (HailoRT 4.20.0). No prebuilt MegaDetector `.hef` for the Hailo-8L is
available to download. So the `.hef` must be built on an x86 machine (or cloud
VM) and copied to the Pi.

**Version pin:** the Pi runs **HailoRT 4.20.0**. Compile with the matching
Hailo AI Software Suite / DFC release (the 2024.x train that pairs with HailoRT
4.20) or the `.hef` may not load. Check with `hailo --version` on the Pi.

## Where to compile: a Windows PC works (via WSL2)

The DFC needs Ubuntu-x86, but a normal **Windows PC can run it under WSL2**
(Windows Subsystem for Linux). Requirements to check first:

- **x86-64 Windows** (Intel/AMD) — *not* an ARM Windows machine.
- **16 GB RAM minimum, 32 GB recommended** — the compiler is memory-hungry.
- **WSL2 with Ubuntu 22.04/24.04**, Python 3.10–3.12.
- Give WSL2 enough RAM via `C:\Users\<you>\.wslconfig`:
  `[wsl2]` / `memory=18GB` / `swap=12GB`, then `wsl --shutdown` to apply.
- No Hailo device needed on the PC — compiling is CPU-only; the NPU stays on
  the Pi for running.

(Alternatively: Docker Desktop with the Hailo Suite image, or a cloud x86 VM.)

## Compile recipe — SELF-CONTAINED BRIEF FOR THE BUILD AGENT

This section is written so an agent on the x86/WSL2 machine can execute it
without any other context. Target device is a **Raspberry Pi 5 + Hailo-8L
(hailo8l)** running **HailoRT 4.20.0** — the produced `.hef` MUST load under
that runtime.

1. **Toolchain (version-pinned).** Create a free Hailo Developer Zone account.
   Install the **Hailo Dataflow Compiler whose release pairs with HailoRT
   4.20.0** (use the matching Hailo AI Software Suite Docker image, or
   `hailo-dataflow-compiler` in their venv). A mismatched DFC can emit a `.hef`
   the Pi can't load. No Hailo hardware needed on this machine.

2. **Model → ONNX.** From Microsoft's Pytorch-Wildlife / MegaDetector
   (`microsoft/CameraTraps`): use **MegaDetector v6, a COMPACT variant**
   (`MDV6-yolov9-c` or `MDV6-yolov10`, whichever exports cleanest). Export to
   **ONNX, input 640×640, opset 11–13**. MegaDetector classes are
   **1=animal, 2=person, 3=vehicle** — confirm the exact index order from the
   export and record it (the Pi needs it).

3. **Calibration images (int8 quantization).** ~256–1024 representative images.
   The Pi currently has no usable capture corpus, so **source a public
   outdoor/camera-trap set** for v1 — e.g. a sample of LILA/Caltech Camera
   Traps or COCO images containing animals + outdoor scenes. (A v2 recompile
   later can use real frames from the Pi's `timelapse/` or `captures/` once
   they accumulate, for better accuracy.)

4. **Compile.** DFC flow: `parse` ONNX → `optimize` (quantize with the
   calibration set) → `compile` targeting **`hailo8l`**. **Bake the NMS
   post-process into the HEF** (as the Pi's existing `yolov8s_h8l.hef` does —
   output `HAILO NMS BY CLASS`, UINT8 input, FLOAT32 output). This lets the Pi
   reuse the existing detection-parsing path with minimal new code. Name the
   output **`megadetector_h8l.hef`**.

5. **Verify before handing back.** Run `hailortcli parse-hef megadetector_h8l.hef`
   (works without a device). Confirm: `Architecture: HAILO8L`, input 640×640,
   and an NMS output op with the expected class count (3).

### Deliverables to hand back to the Pi

- `megadetector_h8l.hef`
- The **full `parse-hef` output** (input/output vstream names, NMS op config,
  class count).
- The **class index → label map** actually used (e.g. 0/1/2 or 1/2/3 →
  animal/person/vehicle).
- Input preprocessing expected (letterbox vs plain resize; normalization).

With those, the Pi-side backend in `objdetect.py` can be written against the
real tensor layout instead of guessed.

## Pi-side integration (done here once a .hef exists)

`objdetect.py` gets a MegaDetector backend that reports `has_animal` + box with
a generic `"animal"` label, replacing the COCO yolov8s pre-filter. The Phase 0
"Animals" track (detector routing, gallery filter, stats, settings) is reused
as-is — only what `analyze_frame` reports changes. Birds still route to the
bird classifier (owls included). Deferred until the real `.hef` is in hand so
the output-tensor parsing is written against actual model output, not guessed.

## Status

- [x] Confirmed compile is off-Pi; identified MDv6-compact as the target model.
- [x] Wrote self-contained build brief above for an agent on a Windows/WSL2 PC.
- [ ] Build agent compiles `megadetector_h8l.hef` + returns the deliverables.
- [ ] Wire MegaDetector backend into `objdetect.py` (against the real tensors).
- [ ] Phase 2: species classifier (SpeciesNet / DeepFaune) to name animals.
