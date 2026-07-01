import csv
import numpy as np
from pathlib import Path
from ai_edge_litert.interpreter import Interpreter

MODEL_PATH = Path(__file__).parent / "models" / "birds_classifier.tflite"
LABELS_PATH = Path(__file__).parent / "models" / "birds_labels.csv"
CONFIDENCE_THRESHOLD = 0.3


def load_labels():
    with open(LABELS_PATH) as f:
        reader = csv.DictReader(f)
        return {int(row["id"]): row["name"] for row in reader}


def load_interpreter(num_threads=None):
    kwargs = {}
    if num_threads:
        kwargs["num_threads"] = num_threads
    interp = Interpreter(model_path=str(MODEL_PATH), **kwargs)
    interp.allocate_tensors()
    return interp


def classify_image(image_path, interp=None, labels=None):
    from PIL import Image

    if interp is None:
        interp = load_interpreter()
    if labels is None:
        labels = load_labels()

    input_details = interp.get_input_details()[0]
    output_details = interp.get_output_details()[0]
    h, w = input_details["shape"][1], input_details["shape"][2]

    img = Image.open(image_path).convert("RGB").resize((w, h))
    input_data = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)

    interp.set_tensor(input_details["index"], input_data)
    interp.invoke()

    output = interp.get_tensor(output_details["index"])[0]
    # Dequantize if needed
    scale, zero_point = output_details.get("quantization", (1.0, 0))
    if scale != 0:
        output = (output.astype(np.float32) - zero_point) * scale

    top_idx = int(np.argmax(output))
    confidence = float(output[top_idx])
    species = labels.get(top_idx, "unknown")

    # No hardcoded floor here - the caller's confidence_threshold setting is the
    # single source of truth so the UI slider actually controls sensitivity.
    return {"species": species, "confidence": confidence, "is_bird": top_idx != 964}


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        # Test on the most recent capture
        captures = sorted(Path(__file__).parent.glob("captures/*.jpg"))
        if not captures:
            print("No captures found.")
            sys.exit(1)
        path = captures[-1]

    result = classify_image(path)
    print(f"Image: {path}")
    if result["is_bird"]:
        print(f"Bird detected: {result['species']} ({result['confidence']:.1%} confidence)")
    elif result["species"] is None:
        print(f"No bird detected (confidence too low: {result['confidence']:.1%})")
    else:
        print(f"Background/no bird ({result['confidence']:.1%})")
