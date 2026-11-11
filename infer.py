"""
PCB Defect Detection - Inference & CSV Export Script

Outputs predictions adhering strictly to the competition format:
image_name,class_name,x_min,y_min,x_max,y_max,confidence

Rules strictly followed:
1. Coordinates in original image pixels (x_min, y_min, x_max, y_max).
2. Class names match prescribed labels.
3. Confidence score between 0 and 1.
4. Each detection occupies a separate row.
"""

import os
import sys
import glob
import argparse
import pandas as pd
from ultralytics import YOLO

# Prescribed competition class names
ALLOWED_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper"
]

def run_inference(weights_path="best.pt", input_dir="test/images", output_csv="submission.csv", conf_thresh=0.25, imgsz=1024, device="cpu"):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")
    
    # Load model
    model = YOLO(weights_path)
    
    # Supported image extensions
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG")
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))
    
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"Warning: No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to run inference on...")

    records = []
    
    # Using streaming generator: zero RAM overhead and continuous output
    results_generator = model.predict(
        source=image_files,
        conf=conf_thresh,
        imgsz=imgsz,
        device=device,
        stream=True,
        verbose=False
    )
    
    for count, r in enumerate(results_generator, 1):
        image_name = os.path.basename(r.path)
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            class_name = model.names.get(cls_id, str(cls_id))
            confidence = float(box.conf[0].item())
            
            # Bounding box in original pixels: [x_min, y_min, x_max, y_max]
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            
            records.append({
                "image_name": image_name,
                "class_name": class_name,
                "x_min": round(x_min, 2),
                "y_min": round(y_min, 2),
                "x_max": round(x_max, 2),
                "y_max": round(y_max, 2),
                "confidence": round(confidence, 4)
            })
        
        if count % 25 == 0 or count == len(image_files):
            print(f"Processed [{count}/{len(image_files)}] images...")
            sys.stdout.flush()

    df = pd.DataFrame(records, columns=[
        "image_name",
        "class_name",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "confidence"
    ])
    
    df.to_csv(output_csv, index=False)
    print(f"Successfully generated {output_csv} with {len(df)} defect predictions across {len(image_files)} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PCB Defect Detection Inference")
    parser.add_argument("--weights", type=str, default="best.pt", help="Path to best.pt trained checkpoint")
    parser.add_argument("--input", type=str, default="test/images", help="Path to directory containing test images")
    parser.add_argument("--output", type=str, default="submission.csv", help="Path to output submission CSV")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size for inference")
    parser.add_argument("--device", default="cpu", help="Device ('cpu' or '0')")
    
    args = parser.parse_args()
    run_inference(
        weights_path=args.weights,
        input_dir=args.input,
        output_csv=args.output,
        conf_thresh=args.conf,
        imgsz=args.imgsz,
        device=args.device
    )

