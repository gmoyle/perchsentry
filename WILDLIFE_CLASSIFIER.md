# Phase 2 — Wildlife species classifier (SpeciesNet → ONNX → Pi CPU)

**Audience:** an agent on the same x86/WSL2 machine used for the MegaDetector
build (it already has Pytorch-Wildlife installed). This is fully self-contained.

## Objective

BirdBuddy (Raspberry Pi 5) already runs **MegaDetector** on the Hailo NPU to
find animals in daytime frames, and logs each non-bird animal generically as
`"animal"`. We now want to **name the species** (squirrel, deer, fox, raccoon,
…) using Google's **SpeciesNet classifier**, run on the animal crop.

Unlike the detector, the classifier runs on the **Pi CPU via ONNX Runtime**
(already installed and verified: `onnxruntime 1.27.0`, CPUExecutionProvider) —
per motion-event, so ~1–2 s inference is fine. **No Hailo/DFC compile needed.**
Your job: export the SpeciesNet classifier to ONNX and report its exact I/O so
the Pi-side code can be written against real tensors.

## Steps

1. **Install / locate SpeciesNet.** `pip install speciesnet` (Google's
   `cameratrapai`). SpeciesNet = MegaDetector + an EfficientNetV2-M **classifier**
   (~2000+ labels) + a geofence/ensemble step. We only need the **classifier**
   sub-model; the detector and ensemble are out of scope.

2. **Inspect the actual API before exporting** (don't guess — the loader
   changes between versions). Find how the package loads the classifier weights
   (PyTorch vs TF/Keras), its input size, and its preprocessing. Print the model
   class, input shape, and the label list length.

3. **Export to ONNX.**
   - If PyTorch: `torch.onnx.export(classifier, dummy, "speciesnet_classifier.onnx",
     opset_version=13, input_names=["input"], output_names=["logits"],
     dynamic_axes=None)` with a fixed batch-1 input at the model's real input size.
   - If TF/Keras: use `tf2onnx` (`python -m tf2onnx.convert --saved-model ... --opset 13`).
   - Keep batch fixed at 1.

4. **Capture the EXACT preprocessing** — this is the part most likely to break
   accuracy if wrong. Report all of:
   - Input size (EfficientNetV2-M is often 480×480, but confirm what SpeciesNet
     actually uses).
   - Resize vs resize-then-center-crop, and interpolation.
   - Normalization: mean/std (ImageNet? EfficientNet-specific?) or plain /255,
     and channel order (RGB).
   - Whether it classifies the MegaDetector **crop** (it does) — so the Pi will
     feed the detected animal box, not the whole frame.

5. **Export the label map.** The class-index → label list SpeciesNet uses.
   Labels are taxonomy strings like
   `uuid;class;order;family;genus;species;common name`. Provide the full list in
   index order as `speciesnet_labels.txt` (one per line), so the Pi can take the
   **common-name** field for display.

6. **Verify.** Run the ONNX with `onnxruntime` on a couple of clear animal
   images (fox, deer, squirrel). Apply softmax to the logits, take top-1, map to
   the label, and confirm it's sane. Include this output in the deliverables.

## Deliverables to hand back to the Pi (place in ~/speciesnet_build)

1. `speciesnet_classifier.onnx`
2. `speciesnet_labels.txt` — index-ordered labels (full taxonomy strings).
3. **Preprocessing spec** — input size, resize/crop, normalization mean/std,
   channel order.
4. **Output spec** — logits vs probabilities, and index→label confirmed by the
   verify step above.
5. **A couple of verified example outputs** (image → top-1 label + score).
6. `sha256` of the .onnx.

## Notes / gotchas

- **Skip the geofence/ensemble.** Full SpeciesNet restricts predictions to
  species plausible for a location. We're using raw top-1 for now; a later
  refinement on the Pi can add a Pacific-Northwest allowlist to suppress absurd
  predictions (e.g. an African species in a WA backyard).
- EfficientNetV2-M is ~50–200 MB in ONNX; fine to copy to the Pi.
- If SpeciesNet only ships as a large multi-part system and extracting the bare
  classifier is painful, note that and we'll reconsider (e.g. DeepFaune, or the
  full speciesnet package running on the Pi in CPU mode).

## Pi-side integration (done here once the ONNX arrives)

- New module `speciesclassify.py` (parallel to `classify.py`): load the ONNX via
  onnxruntime, preprocess the crop per the spec, softmax → top-1 → (common_name,
  confidence).
- `detector.py`: in the non-bird branch, run the species classifier on the crop
  MegaDetector already produced, and log `ANIMAL DETECTED: <species> (conf%)`
  instead of the generic `"animal"`. The Animals track (gallery/stats/settings)
  is reused unchanged.
