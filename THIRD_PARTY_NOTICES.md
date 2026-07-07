# Third-Party Notices

This project redistributes third-party model artifacts. Their licenses and
attributions are below.

## MegaDetector v6 (`models/megadetector_h8l.hef`)

`models/megadetector_h8l.hef` is a Hailo-8L–compiled build of the **MegaDetector
v6** detector, variant **MDV6-yolov9-c**, from Microsoft's AI for Good Lab.

- Project: MegaDetector — https://github.com/microsoft/MegaDetector
- Variant: MDV6-yolov9-c (compact), distributed under the **MIT License**
- Architecture: YOLOv9-c
- This file is a quantized derivative compiled for the Hailo-8L NPU (Hailo
  Dataflow Compiler). It detects `animal / person / vehicle` and is used here as
  the object-detection pre-filter.

MegaDetector is released under the MIT License:

```
MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

> Note: MegaDetector v6 ships multiple variants under different licenses (MIT,
> Apache-2.0, AGPL-3.0). This project deliberately uses the **MIT** `yolov9-c`
> variant so the compiled `.hef` can be redistributed here without copyleft
> obligations. If you rebuild with a different variant (e.g. a YOLOv10 variant,
> which is AGPL-3.0), its license terms apply instead.


## SpeciesNet (`models/speciesnet_classifier.onnx`, `models/speciesnet_labels.txt`)

`models/speciesnet_classifier.onnx` is an ONNX export of Google's SpeciesNet
classifier, model `kaggle:google/speciesnet/pyTorch/v4.0.3a/1`, produced from
the `speciesnet 5.0.5` package. `models/speciesnet_labels.txt` is the
corresponding index-ordered label map.

- Project: SpeciesNet / CameraTrapAI — https://github.com/google/cameratrapai
- License: Apache License 2.0
- Classifier type: `always_crop`, EfficientNetV2-M, 2,498 labels
- Runtime input: RGB NHWC `1x480x480x3`, float32 `0.0..1.0`

SpeciesNet source files include this notice:

```text
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Other models (not redistributed here)

The legacy bird species classifier (`models/birds_classifier.tflite`) and the
COCO YOLOv8s fallback (`/usr/share/hailo-models/yolov8s_h8l.hef`) are **not**
shipped in this repository; they are obtained separately at setup time.
