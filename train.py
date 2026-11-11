"""
PCB Defect Detection - Model Training Script
Target Classes: missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper
Model Architecture: YOLOv8m (Ultralytics)
Metric Target: mAP@50:90
"""

import os
import argparse
from ultralytics import YOLO

def train(data_yaml="data.yaml", epochs=50, imgsz=1024, batch=8, device=0):
    # Initialize YOLOv8 medium model (pretrained on COCO)
    model = YOLO("yolov8m.pt")
    
    # Run training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        name="pcb_defect_yolov8m",
        save=True,
        plots=True,
        patience=15  # Early stopping patience
    )
    print("Training complete! Best weights saved to runs/detect/pcb_defect_yolov8m/weights/best.pt")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PCB Defect Detection Model")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default=0, help="Device to use: 0 for GPU, or 'cpu'")
    
    args = parser.parse_args()
    train(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )

