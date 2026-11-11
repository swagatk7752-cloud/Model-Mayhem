# PCB Manufacturing Defect Detection System

## 1. Overview
This solution builds an object detection system based on **YOLOv8** to identify and localise 6 categories of manufacturing defects on Printed Circuit Boards (PCBs):
- `missing_hole`
- `mouse_bite`
- `open_circuit`
- `short`
- `spur`
- `spurious_copper`

The primary evaluation metric is **mAP@50:90**, with tie-breakers on mAP@50 and Macro Recall.

---

## 2. Environment Setup

Install required Python dependencies:
```bash
pip install -r requirements.txt
```

---

## 3. Training the Model

To train the model on GPU (e.g. Google Colab / CUDA-enabled workstation):
```bash
python train.py --data data.yaml --epochs 50 --imgsz 1024 --batch 8 --device 0
```

- High resolution (`imgsz=1024`) is selected specifically to preserve micro-features and small geometry defects (such as mouse bites and copper spurs).
- Checkpoints will automatically be saved under `runs/detect/pcb_defect_yolov8m/weights/best.pt`.

---

## 4. Inference & Generating Predictions

To generate predictions on an unseen test dataset in the exact competition CSV format:

```bash
python infer.py --weights best.pt --input path/to/unseen_test_images/ --output submission.csv --device cpu
```

### Parameters:
- `--weights`: Path to the trained checkpoint (`best.pt`).
- `--input`: Directory containing test images.
- `--output`: Output CSV destination (default: `submission.csv`).
- `--conf`: Confidence threshold (default: `0.25`).
- `--imgsz`: Inference resolution (default: `1024`).
- `--device`: `cpu` or `0` for GPU.

---

## 5. Submission CSV Format Compliance
The script guarantees strict adherence to all submission rules:
- Header: `image_name,class_name,x_min,y_min,x_max,y_max,confidence`
- Coordinates are in original pixel dimensions.
- Labels strictly conform to prescribed classes.
- One detection per row.

