# 🎯 KẾ HOẠCH CHI TIẾT: Phân loại rác thải thông minh bằng YOLOv8

## 📊 TÓM TẮT VẤN ĐỀ

**Dự án:** Hệ thống phân loại rác thải tự động trong dây chuyền xử lý công nghiệp

- **Deliverable:** Code hoàn chỉnh + Báo cáo chi tiết
- **Thời gian:** 3-4 ngày
- **Công nghệ:** YOLOv8 (Object Detection)
- **Môi trường:** Google Colab (Free GPU T4)
- **Dataset:** Nguồn công khai trên mạng
- **4 Classes:** Plastic (Nhựa), Metal (Kim loại), Paper (Giấy), Glass (Thủy tinh)

---

## 🗺️ ROADMAP TỔNG THỂ (3-4 NGÀY)

```
Ngày 1: [Task 1 + Task 2] → Dataset + Setup
Ngày 2: [Task 3] → Training Model
Ngày 3: [Task 4] → Testing + Optimization
Ngày 4: [Task 5] → Báo cáo + Hoàn thiện
```

---

# 📦 TASK 1: THU THẬP VÀ CHUẨN BỊ DATASET

## 🔑 Key Concepts

- **YOLO Format:** Annotation dạng `.txt` với format `class_id x_center y_center width height` (normalized 0-1)
- **Data Split:** Train/Valid/Test thường theo tỷ lệ 70/20/10 hoặc 80/10/10
- **Class Balance:** Đảm bảo 4 lớp (Plastic/Metal/Paper/Glass) có số lượng ảnh tương đương
- **Data Augmentation:** YOLOv8 tự động augment khi training

## 📝 Các bước thực hiện

### Bước 1.1: Tìm kiếm dataset công khai

**Nguồn đề xuất:**

- **Roboflow Universe:** https://universe.roboflow.com (search "waste classification")
- **Kaggle:** Search "waste detection dataset" hoặc "garbage classification"
- **Specific Datasets:**
  - TrashNet Dataset
  - Waste Classification Data (Kaggle)
  - TACO Dataset (Trash Annotations in Context)

### Bước 1.2: Kiểm tra và tải dataset

Chọn dataset có:

- ✅ Đủ 4 class: Plastic, Metal, Paper, Glass
- ✅ Format YOLO (hoặc có thể convert)
- ✅ Tối thiểu 500-1000 ảnh
- ✅ Annotations chất lượng cao

### Bước 1.3: Chuẩn bị cấu trúc thư mục

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

### Bước 1.4: Tạo file `data.yaml`

```yaml
train: ./dataset/train/images
val: ./dataset/valid/images
test: ./dataset/test/images

nc: 4 # number of classes
names: ["Plastic", "Metal", "Paper", "Glass"]
```

### Bước 1.5: Upload lên Google Drive

- Zip dataset và upload lên Google Drive
- Đặt tên rõ ràng: `waste_detection_dataset.zip`

## 💡 Gợi ý & Code hữu ích

**Code kiểm tra dataset:**

```python
import os
import glob

# Đếm số ảnh và labels
train_images = len(glob.glob('dataset/train/images/*'))
train_labels = len(glob.glob('dataset/train/labels/*'))
print(f"Train: {train_images} images, {train_labels} labels")

# Kiểm tra class distribution
class_counts = {0:0, 1:0, 2:0, 3:0}
for label_file in glob.glob('dataset/train/labels/*.txt'):
    with open(label_file, 'r') as f:
        for line in f:
            class_id = int(line.split()[0])
            class_counts[class_id] += 1

class_names = ['Plastic', 'Metal', 'Paper', 'Glass']
for i, count in class_counts.items():
    print(f"{class_names[i]}: {count} objects")
```

---

# ⚙️ TASK 2: THIẾT LẬP MÔI TRƯỜNG VÀ LÀM QUEN YOLOv8

## 🔑 Key Concepts

- **Ultralytics YOLOv8:** Library dễ sử dụng, high-level API
- **Pre-trained Weights:** Sử dụng transfer learning từ COCO dataset
- **Model Sizes:** YOLOv8n (nano), s (small), m (medium), l (large), x (xlarge)
- **Google Colab GPU:** T4 GPU miễn phí (12-15GB RAM)

## 📝 Các bước thực hiện

### Bước 2.1: Tạo Google Colab Notebook

- Truy cập https://colab.research.google.com
- Tạo notebook: `Waste_Classification_YOLOv8.ipynb`
- Enable GPU: `Runtime > Change runtime type > T4 GPU`

### Bước 2.2: Cài đặt YOLOv8

```python
# Install Ultralytics YOLOv8
!pip install ultralytics -q

# Import libraries
from ultralytics import YOLO
import torch
import os
from google.colab import drive
from IPython.display import Image, display

# Check GPU
print("GPU Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

### Bước 2.3: Mount Google Drive và load dataset

```python
# Mount Drive
drive.mount('/content/drive')

# Unzip dataset
!unzip -q /content/drive/MyDrive/waste_detection_dataset.zip -d /content/

# Verify structure
!ls -la dataset/
```

### Bước 2.4: Test YOLOv8 với pretrained model

```python
# Load pretrained YOLOv8n
model = YOLO('yolov8n.pt')

# Test inference trên 1 ảnh mẫu
results = model.predict(source='dataset/train/images/sample_001.jpg', save=True)

# Hiển thị kết quả
display(Image('runs/detect/predict/sample_001.jpg'))
```

## 💡 Hiểu về YOLOv8 cho người quen CNN

**So sánh YOLOv8 vs CNN truyền thống:**

- **CNN (Classification):** Input image → Output class label (1 nhãn)
- **YOLO (Detection):** Input image → Output multiple [bbox, class, confidence] (nhiều objects)

**Kiến trúc YOLOv8 đơn giản hóa:**

```
Input Image (640x640)
    ↓
Backbone (CSPDarknet) - giống CNN feature extractor
    ↓
Neck (FPN/PAN) - kết hợp features ở nhiều scales
    ↓
Head (Detection) - predict [x, y, w, h, class probabilities]
```

---

# 🎓 TASK 3: TRAINING MODEL YOLOV8

## 🔑 Key Concepts

- **Transfer Learning:** Fine-tune từ pretrained weights (COCO dataset)
- **Epochs:** Số lần model "học" toàn bộ dataset (50-100 epochs cho dataset nhỏ)
- **Image Size (imgsz):** 640x640 là chuẩn (có thể giảm xuống 416 nếu GPU yếu)
- **Batch Size:** Số ảnh xử lý cùng lúc (16-32 cho Colab T4)
- **Learning Rate:** Tốc độ học (YOLOv8 tự động tune)

## 📝 Các bước thực hiện

### Bước 3.1: Chọn model size phù hợp

**Khuyến nghị cho 3-4 ngày:**

- `yolov8n.pt` (Nano): Fastest, 3.2M params - **ĐỀ XUẤT cho newbie**
- `yolov8s.pt` (Small): Balance, 11.2M params - Nếu muốn accuracy cao hơn

### Bước 3.2: Cấu hình training parameters

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')  # hoặc yolov8s.pt

# Training
results = model.train(
    data='dataset/data.yaml',      # Path to data.yaml
    epochs=100,                      # Số epochs
    imgsz=640,                       # Image size
    batch=16,                        # Batch size (giảm xuống 8 nếu OOM)
    patience=20,                     # Early stopping patience
    save=True,                       # Save checkpoints
    device=0,                        # GPU device (0 = first GPU)
    project='runs/waste_detection',  # Save location
    name='yolov8n_waste',           # Experiment name
    exist_ok=True,                   # Overwrite existing
    pretrained=True,                 # Use pretrained weights
    optimizer='AdamW',               # Optimizer
    verbose=True,                    # Print training info
    seed=42,                         # Reproducibility

    # Augmentation (có thể bật nếu dataset nhỏ)
    hsv_h=0.015,                     # Hue augmentation
    hsv_s=0.7,                       # Saturation
    hsv_v=0.4,                       # Value
    degrees=10.0,                    # Rotation
    translate=0.1,                   # Translation
    scale=0.5,                       # Scaling
    flipud=0.0,                      # Flip up-down
    fliplr=0.5,                      # Flip left-right
    mosaic=1.0,                      # Mosaic augmentation
)
```

### Bước 3.3: Monitor training

```python
# Training sẽ tự động hiển thị:
# - Loss curves (box_loss, cls_loss, dfl_loss)
# - Metrics (Precision, Recall, mAP50, mAP50-95)
# - Validation results

# Sau khi training xong, xem results:
!ls runs/waste_detection/yolov8n_waste/

# File quan trọng:
# - weights/best.pt (model tốt nhất)
# - weights/last.pt (model cuối cùng)
# - results.csv (metrics theo epochs)
# - confusion_matrix.png
```

### Bước 3.4: Visualize training results

```python
from IPython.display import Image

# Loss curves
display(Image('runs/waste_detection/yolov8n_waste/results.png'))

# Confusion matrix
display(Image('runs/waste_detection/yolov8n_waste/confusion_matrix.png'))

# Sample predictions
display(Image('runs/waste_detection/yolov8n_waste/val_batch0_pred.jpg'))
```

## 💡 Tips để training hiệu quả

**Nếu training bị Out of Memory (OOM):**

```python
# Giảm batch size
batch=8  # hoặc 4

# Hoặc giảm image size
imgsz=416  # thay vì 640
```

**Theo dõi training từ xa (tránh Colab disconnect):**

```python
# Sao lưu weights định kỳ vào Drive
import shutil

# Sau mỗi 10 epochs, copy weights sang Drive
!cp runs/waste_detection/yolov8n_waste/weights/best.pt \
    /content/drive/MyDrive/yolov8_waste_backup.pt
```

---

# 🧪 TASK 4: TESTING, EVALUATION VÀ OPTIMIZATION

## 🔑 Key Concepts

- **mAP (mean Average Precision):** Metric chính cho object detection
  - mAP@0.5: IoU threshold = 0.5 (dễ hơn)
  - mAP@0.5:0.95: IoU từ 0.5-0.95 (khó hơn, chuẩn COCO)
- **Precision:** Tỷ lệ predictions đúng trong số predictions (độ chính xác)
- **Recall:** Tỷ lệ objects được detect trong tổng số objects (độ phủ)
- **Confusion Matrix:** Ma trận nhầm lẫn giữa các classes

## 📝 Các bước thực hiện

### Bước 4.1: Load model đã train

```python
# Load best model
model = YOLO('runs/waste_detection/yolov8n_waste/weights/best.pt')
```

### Bước 4.2: Evaluate trên test set

```python
# Validation
metrics = model.val(
    data='dataset/data.yaml',
    split='test',           # Evaluate on test set
    imgsz=640,
    batch=16,
    save_json=True,         # Save COCO format results
    save_hybrid=True,       # Save labels + predictions
    conf=0.25,              # Confidence threshold
    iou=0.6,                # IoU threshold for NMS
    plots=True              # Generate plots
)

# Print metrics
print("\n=== EVALUATION RESULTS ===")
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")

# Per-class metrics
print("\n=== PER-CLASS METRICS ===")
class_names = ['Plastic', 'Metal', 'Paper', 'Glass']
for i, name in enumerate(class_names):
    print(f"{name}: mAP@0.5 = {metrics.box.maps[i]:.4f}")
```

### Bước 4.3: Test trên ảnh thực tế

```python
# Predict trên test images
results = model.predict(
    source='dataset/test/images',
    conf=0.25,              # Confidence threshold
    iou=0.6,                # IoU for NMS
    save=True,              # Save results
    save_txt=True,          # Save labels
    save_conf=True,         # Save confidence scores
    show_labels=True,       # Show labels
    show_conf=True,         # Show confidence
    line_width=2,           # Bounding box thickness
    project='runs/test',
    name='final_results'
)

# Hiển thị một số kết quả
import glob
test_results = glob.glob('runs/test/final_results/*.jpg')[:5]
for img_path in test_results:
    display(Image(img_path))
```

### Bước 4.4: Tạo demo video/stream (Optional)

```python
# Predict trên video
results = model.predict(
    source='demo_video.mp4',
    save=True,
    stream=True,            # Stream results
    conf=0.3
)

# Hoặc webcam (nếu có)
# results = model.predict(source=0, show=True)
```

### Bước 4.5: Optimization - Tinh chỉnh threshold

```python
# Thử nghiệm với confidence threshold khác nhau
confidence_thresholds = [0.1, 0.25, 0.5, 0.7]

for conf in confidence_thresholds:
    print(f"\n=== Testing with confidence = {conf} ===")
    metrics = model.val(
        data='dataset/data.yaml',
        conf=conf,
        plots=False
    )
    print(f"mAP@0.5: {metrics.box.map50:.4f}, Precision: {metrics.box.mp:.4f}, Recall: {metrics.box.mr:.4f}")
```

## 💡 Phân tích kết quả

**Tiêu chí đánh giá tốt cho bài tập lớn:**

- ✅ **mAP@0.5 ≥ 0.70** (70%+): Khá tốt
- ✅ **mAP@0.5:0.95 ≥ 0.50** (50%+): Chấp nhận được
- ✅ **Precision ≥ 0.75**: Ít false positives
- ✅ **Recall ≥ 0.70**: Phát hiện được đa số objects

**Nếu kết quả chưa tốt:**

1. Tăng epochs (100 → 150-200)
2. Dùng model lớn hơn (yolov8n → yolov8s)
3. Tăng data augmentation
4. Kiểm tra lại chất lượng annotations

---

# 📝 TASK 5: VIẾT BÁO CÁO VÀ HOÀN THIỆN

## 🔑 Key Concepts

- **Reproducibility:** Báo cáo phải đủ chi tiết để người khác tái hiện được
- **Visualization:** Dùng biểu đồ, hình ảnh minh họa
- **Scientific Writing:** Ngôn ngữ chuyên nghiệp, rõ ràng
- **Code Documentation:** Comment code, README rõ ràng

## 📝 Các bước thực hiện

### Bước 5.1: Cấu trúc báo cáo

**Mục lục báo cáo (8-15 trang):**

```markdown
1. GIỚI THIỆU
   1.1. Bối cảnh và động lực
   1.2. Mục tiêu nghiên cứu
   1.3. Phạm vi đề tài

2. CƠ SỞ LÝ THUYẾT
   2.1. Object Detection
   2.2. YOLO (You Only Look Once)
   2.3. YOLOv8 Architecture
   2.4. Transfer Learning

3. PHƯƠNG PHÁP THỰC HIỆN
   3.1. Dataset

   - Nguồn dữ liệu
   - Thống kê dataset
   - Data preprocessing
     3.2. Môi trường thực nghiệm
     3.3. Cấu hình training
     3.4. Evaluation metrics

4. KẾT QUẢ VÀ THẢO LUẬN
   4.1. Kết quả training

   - Loss curves
   - Metrics theo epochs
     4.2. Kết quả testing
   - mAP, Precision, Recall
   - Per-class performance
   - Confusion matrix
     4.3. Visualizations
     4.4. Phân tích và nhận xét

5. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
   5.1. Tóm tắt kết quả
   5.2. Hạn chế
   5.3. Hướng phát triển

6. TÀI LIỆU THAM KHẢO

7. PHỤ LỤC
   - Source code chính
   - Thông số chi tiết
```

### Bước 5.2: Thu thập tài liệu cho báo cáo

**Artifacts cần có:**

```python
# Tạo folder báo cáo
!mkdir -p report_artifacts

# Copy các file quan trọng
!cp runs/waste_detection/yolov8n_waste/results.png report_artifacts/
!cp runs/waste_detection/yolov8n_waste/confusion_matrix.png report_artifacts/
!cp runs/waste_detection/yolov8n_waste/val_batch0_pred.jpg report_artifacts/
!cp runs/test/final_results/*.jpg report_artifacts/sample_predictions/

# Export metrics to CSV
import pandas as pd
metrics_df = pd.read_csv('runs/waste_detection/yolov8n_waste/results.csv')
metrics_df.to_excel('report_artifacts/training_metrics.xlsx', index=False)

# Zip toàn bộ
!zip -r report_artifacts.zip report_artifacts/
```

### Bước 5.3: Viết các phần chính

**Section 1: Giới thiệu (1-2 trang)**

```markdown
## 1. GIỚI THIỆU

### 1.1. Bối cảnh và động lực

Trong bối cảnh ô nhiễm môi trường ngày càng nghiêm trọng, việc phân loại
và xử lý rác thải hiệu quả trở thành vấn đề cấp thiết. Hệ thống phân loại
rác tự động sử dụng thị giác máy tính có thể giúp tối ưu hóa quy trình
tái chế trong các nhà máy xử lý rác...

### 1.2. Mục tiêu nghiên cứu

- Xây dựng mô hình deep learning phát hiện và phân loại 4 loại rác
- Đạt độ chính xác mAP@0.5 ≥ 70%
- Ứng dụng được trong môi trường thực tế (dây chuyền công nghiệp)
```

**Section 2: Cơ sở lý thuyết (2-3 trang)**

```markdown
## 2. CƠ SỞ LÝ THUYẾT

### 2.2. YOLO (You Only Look Once)

YOLO là họ mô hình object detection sử dụng approach "single-stage",
dự đoán bounding boxes và class probabilities trực tiếp từ full image
trong một lần forward pass. Điều này giúp YOLO nhanh hơn đáng kể so với
two-stage detectors như R-CNN...

[Chèn hình: YOLO architecture diagram]

### 2.3. YOLOv8 Architecture

YOLOv8 (2023) là phiên bản mới nhất với những cải tiến:

- Backbone: CSPDarknet với C2f modules
- Neck: PAN (Path Aggregation Network)
- Head: Decoupled head (classification & localization riêng biệt)
- Loss: VFL (Varifocal Loss) + CIoU loss

[Chèn công thức toán học nếu cần]
```

**Section 4: Kết quả (3-4 trang)**

```markdown
## 4. KẾT QUẢ VÀ THẢO LUẬN

### 4.1. Kết quả Training

Model được training trong 100 epochs với early stopping patience = 20.
Hình 4.1 thể hiện sự hội tụ của các loss functions và metrics.

[Chèn hình: results.png - loss curves và metrics]

Nhận xét:

- Loss giảm đều, không có dấu hiệu overfitting
- mAP@0.5 đạt 0.78 tại epoch 87 (best)
- Precision và Recall cân bằng (~0.75)

### 4.2. Kết quả Testing

Bảng 4.1: Performance metrics trên test set

| Metric       | Value |
| ------------ | ----- |
| mAP@0.5      | 0.782 |
| mAP@0.5:0.95 | 0.534 |
| Precision    | 0.756 |
| Recall       | 0.724 |
| F1-Score     | 0.740 |

[Chèn hình: Confusion Matrix]

Bảng 4.2: Per-class Performance

| Class   | mAP@0.5 | Precision | Recall |
| ------- | ------- | --------- | ------ |
| Plastic | 0.812   | 0.789     | 0.756  |
| Metal   | 0.798   | 0.812     | 0.742  |
| Paper   | 0.745   | 0.701     | 0.698  |
| Glass   | 0.773   | 0.723     | 0.701  |

Nhận xét:

- Plastic và Metal được phát hiện tốt nhất (đặc trưng rõ ràng)
- Paper có performance thấp nhất (dễ nhầm với background)
- Model hoạt động ổn định trên cả 4 classes

### 4.3. Visualizations

[Chèn 4-6 hình ảnh predictions tốt và có issues]

Hình 4.3: Một số trường hợp predictions thành công
Hình 4.4: Các trường hợp false positives và false negatives
```

### Bước 5.4: Tổ chức code và tài liệu

**Cấu trúc folder submission:**

```
submission/
├── code/
│   ├── train.py                    # Training script
│   ├── test.py                     # Testing script
│   ├── inference.py                # Inference demo
│   └── requirements.txt            # Dependencies
├── notebook/
│   └── Waste_Classification_YOLOv8.ipynb
├── report/
│   ├── report.pdf                  # Báo cáo chính
│   └── presentation.pptx           # Slides thuyết trình (nếu cần)
├── results/
│   ├── weights/
│   │   └── best.pt                 # Trained model
│   ├── plots/                      # Visualizations
│   └── metrics/                    # CSV, logs
├── dataset/
│   └── data.yaml                   # Dataset config (không upload ảnh nếu lớn)
└── README.md                       # Hướng dẫn chạy
```

**Template README.md:**

````markdown
# Waste Classification System using YOLOv8

## 📌 Overview

Hệ thống phân loại rác thải tự động sử dụng YOLOv8 cho dây chuyền xử lý công nghiệp.

## 🎯 Performance

- **mAP@0.5:** 0.782
- **Precision:** 0.756
- **Recall:** 0.724
- **Classes:** Plastic, Metal, Paper, Glass

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```
````

### 2. Download Dataset

[Link to dataset hoặc hướng dẫn]

### 3. Training

```bash
python code/train.py --data dataset/data.yaml --epochs 100
```

### 4. Testing

```bash
python code/test.py --weights results/weights/best.pt
```

### 5. Inference

```bash
python code/inference.py --source path/to/image.jpg
```

## 📊 Results

[Chèn hình kết quả chính]

## 👥 Team Members

- [Tên thành viên 1]
- [Tên thành viên 2]
- ...

## 📚 References

1. Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
2. [Dataset source]
3. [Papers...]

````

### Bước 5.5: Code scripts chính

**File: `code/train.py`**
```python
"""
Training script for Waste Classification using YOLOv8
"""
from ultralytics import YOLO
import argparse

def train(args):
    # Load model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=20,
        save=True,
        pretrained=True,
        optimizer='AdamW',
        verbose=True,
        seed=42
    )

    print("Training completed!")
    print(f"Best model saved at: {args.project}/{args.name}/weights/best.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Model size')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='0', help='Device (0 or cpu)')
    parser.add_argument('--project', type=str, default='runs/train', help='Project path')
    parser.add_argument('--name', type=str, default='waste_yolov8', help='Experiment name')

    args = parser.parse_args()
    train(args)
````

**File: `code/test.py`**

```python
"""
Testing script for Waste Classification
"""
from ultralytics import YOLO
import argparse
import json

def test(args):
    # Load model
    model = YOLO(args.weights)

    # Validate
    metrics = model.val(
        data=args.data,
        split='test',
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        save_json=True,
        plots=True
    )

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision:    {metrics.box.mp:.4f}")
    print(f"Recall:       {metrics.box.mr:.4f}")
    print("="*50)

    # Save results
    results_dict = {
        'map50': float(metrics.box.map50),
        'map': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr)
    }

    with open('test_results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)

    print("\nResults saved to test_results.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to trained weights')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.6, help='IoU threshold')

    args = parser.parse_args()
    test(args)
```

**File: `code/inference.py`**

```python
"""
Inference script for single image/video
"""
from ultralytics import YOLO
import argparse
import cv2

def inference(args):
    # Load model
    model = YOLO(args.weights)

    # Predict
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
        show=args.show,
        line_width=2,
        show_labels=True,
        show_conf=True
    )

    # Print detections
    class_names = ['Plastic', 'Metal', 'Paper', 'Glass']

    for i, r in enumerate(results):
        print(f"\n--- Image {i+1} ---")
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"{class_names[cls]}: {conf:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to weights')
    parser.add_argument('--source', type=str, required=True, help='Image/video/directory path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.6, help='IoU threshold')
    parser.add_argument('--save', action='store_true', help='Save results')
    parser.add_argument('--show', action='store_true', help='Show results')

    args = parser.parse_args()
    inference(args)
```

**File: `requirements.txt`**

```
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=10.0.0
tqdm>=4.65.0
```

## 💡 Tips viết báo cáo chuyên nghiệp

### Ngôn ngữ và trình bày:

- ✅ Sử dụng thuật ngữ tiếng Anh cho technical terms, giải thích tiếng Việt
- ✅ Trích dẫn papers và sources đầy đủ (IEEE/ACM format)
- ✅ Đánh số hình, bảng, công thức rõ ràng
- ✅ Font chữ: Times New Roman 13pt, Calibri 12pt hoặc Arial 11pt
- ✅ Line spacing: 1.5

### Visualization tips:

```python
# Tạo publication-quality figures
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300  # High resolution
plt.rcParams['font.size'] = 12

# Example: Plot training metrics
import pandas as pd

metrics_df = pd.read_csv('runs/waste_detection/yolov8n_waste/results.csv')

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Loss curves
axes[0, 0].plot(metrics_df['train/box_loss'], label='Train Box Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Box Loss Over Time')
axes[0, 0].legend()
axes[0, 0].grid(True)

# mAP curves
axes[0, 1].plot(metrics_df['metrics/mAP50(B)'], label='mAP@0.5', color='green')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('mAP')
axes[0, 1].set_title('mAP Over Time')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Precision & Recall
axes[1, 0].plot(metrics_df['metrics/precision(B)'], label='Precision', color='blue')
axes[1, 0].plot(metrics_df['metrics/recall(B)'], label='Recall', color='orange')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Precision & Recall')
axes[1, 0].legend()
axes[1, 0].grid(True)

# F1 Score
f1_scores = 2 * (metrics_df['metrics/precision(B)'] * metrics_df['metrics/recall(B)']) / \
            (metrics_df['metrics/precision(B)'] + metrics_df['metrics/recall(B)'])
axes[1, 1].plot(f1_scores, label='F1-Score', color='red')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('F1-Score')
axes[1, 1].set_title('F1-Score Over Time')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('report_artifacts/training_summary.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

# ✅ TIÊU CHÍ ĐÁNH GIÁ CUỐI CÙNG (Acceptable Output Criteria)

## 📊 Tiêu chí kỹ thuật

### 1. Model Performance (40%)

- ✅ **mAP@0.5 ≥ 0.70** (Excellent: ≥0.80)
- ✅ **mAP@0.5:0.95 ≥ 0.45** (Excellent: ≥0.55)
- ✅ **Precision ≥ 0.70**
- ✅ **Recall ≥ 0.65**
- ✅ Các class phải balanced (không có class nào quá kém)

### 2. Code Quality (20%)

- ✅ Code chạy được, không lỗi
- ✅ Có comments và docstrings rõ ràng
- ✅ Cấu trúc folder logic, dễ hiểu
- ✅ README hướng dẫn đầy đủ
- ✅ requirements.txt đầy đủ dependencies

### 3. Báo cáo (30%)

- ✅ Đầy đủ các phần: Giới thiệu, Lý thuyết, Phương pháp, Kết quả, Kết luận
- ✅ Có visualizations (loss curves, confusion matrix, predictions)
- ✅ Phân tích kết quả sâu sắc, không chỉ liệt kê số liệu
- ✅ Ngôn ngữ chuyên nghiệp, không lỗi chính tả
- ✅ Trích dẫn đầy đủ (≥5 tài liệu tham khảo)
- ✅ Format chuẩn học thuật (10-15 trang)

### 4. Presentation & Demo (10%)

- ✅ Slides trình bày rõ ràng (nếu cần thuyết trình)
- ✅ Demo inference trên ảnh thực tế
- ✅ Video demo ngắn (1-2 phút) - optional nhưng impressive

## 🎯 Checklist trước khi nộp

### Code Checklist:

- [ ] Code train.py chạy được trên Colab
- [ ] Code test.py cho ra kết quả đúng
- [ ] Code inference.py demo được trên ảnh mới
- [ ] requirements.txt đầy đủ
- [ ] README.md hướng dẫn rõ ràng
- [ ] Model weights (best.pt) được lưu

### Báo cáo Checklist:

- [ ] Trang bìa có đầy đủ thông tin (tên đề tài, nhóm, môn học)
- [ ] Mục lục có số trang
- [ ] Tất cả hình/bảng có caption và đánh số
- [ ] Tất cả hình/bảng được refer trong text
- [ ] Phần References format chuẩn
- [ ] Không có lỗi chính tả
- [ ] File PDF dưới 20MB

### Results Checklist:

- [ ] Training converged (loss giảm, không diverge)
- [ ] mAP@0.5 ≥ 0.70
- [ ] Confusion matrix reasonable (không bias quá)
- [ ] Sample predictions chất lượng tốt
- [ ] Per-class metrics acceptable

---

# 🚀 TIPS THÀNH CÔNG

## Quản lý thời gian 3-4 ngày:

**Ngày 1 (8 giờ):**

- Sáng: Tìm dataset, setup Colab, test YOLOv8 cơ bản (3h)
- Chiều: Chuẩn bị dataset, tạo data.yaml, start training (3h)
- Tối: Nghiên cứu lý thuyết cho báo cáo (2h)

**Ngày 2 (8 giờ):**

- Sáng: Monitor training, điều chỉnh nếu cần (2h)
- Chiều: Training xong, analyze results (3h)
- Tối: Viết phần Phương pháp của báo cáo (3h)

**Ngày 3 (8 giờ):**

- Sáng: Testing, optimization, tạo visualizations (4h)
- Chiều: Viết phần Kết quả và Kết luận (3h)
- Tối: Hoàn thiện code scripts (1h)

**Ngày 4 (6 giờ):**

- Sáng: Viết phần Giới thiệu và Lý thuyết (2h)
- Chiều: Review toàn bộ, fix lỗi, format báo cáo (3h)
- Tối: Final check, export PDF, chuẩn bị submission (1h)

## Tránh những sai lầm thường gặp:

❌ **KHÔNG NÊN:**

- Training với quá ít epochs (< 50)
- Bỏ qua việc validate model
- Báo cáo chỉ copy-paste lý thuyết không phân tích
- Không backup weights (Colab disconnect là mất hết)
- Dùng dataset quá nhỏ (< 300 ảnh)

✅ **NÊN:**

- Backup weights vào Drive định kỳ
- Test nhiều confidence thresholds
- Phân tích cụ thể từng class performance
- Đưa ví dụ cụ thể (ảnh predictions)
- Thừa nhận limitations và đề xuất improvements

## Resources hữu ích:

1. **YOLOv8 Documentation:** https://docs.ultralytics.com
2. **Roboflow Blog:** https://blog.roboflow.com (nhiều tutorials)
3. **Papers to cite:**
   - Original YOLO: Redmon et al., "You Only Look Once" (2016)
   - YOLOv8: Ultralytics YOLOv8 (2023)
4. **Dataset sources:**
   - Roboflow Universe: https://universe.roboflow.com
   - Kaggle Datasets: https://www.kaggle.com/datasets

---

# 📞 TROUBLESHOOTING

## Vấn đề thường gặp:

### 1. Colab Out of Memory (OOM)

**Giải pháp:**

```python
# Giảm batch size
batch=8  # hoặc 4

# Giảm image size
imgsz=416

# Restart runtime và clear cache
import torch
torch.cuda.empty_cache()
```

### 2. Colab Disconnect giữa chừng

**Giải pháp:**

```python
# Thêm vào đầu notebook:
from google.colab import drive
drive.mount('/content/drive')

# Trong training code:
# Thêm callback để save mỗi 10 epochs
callbacks = {
    'on_train_epoch_end': lambda: shutil.copy(
        'runs/.../weights/last.pt',
        '/content/drive/MyDrive/backup.pt'
    )
}
```

### 3. mAP quá thấp (< 0.50)

**Nguyên nhân & giải pháp:**

- Dataset quá nhỏ → Tìm dataset lớn hơn hoặc augment nhiều
- Annotations kém → Kiểm tra lại labels
- Training chưa đủ → Tăng epochs
- Model quá nhỏ → Dùng yolov8s thay vì yolov8n

### 4. Training không converge (loss không giảm)

**Kiểm tra:**

```python
# Verify data được load đúng
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

# Test trên 1 batch
results = model.train(data='data.yaml', epochs=1, batch=1, imgsz=640)

# Nếu chạy được → tăng dần batch và epochs
```

---

# 🎓 KẾT LUẬN

Với kế hoạch chi tiết này, bạn có thể hoàn thành bài tập lớn trong 3-4 ngày với chất lượng tốt.

**Key success factors:**

1. ⏰ **Time management:** Tuân thủ timeline chặt chẽ
2. 🔍 **Dataset quality:** Chọn dataset tốt ngay từ đầu
3. 💾 **Backup:** Luôn backup weights và code
4. 📊 **Analysis:** Phân tích kết quả sâu sắc, không chỉ liệt kê số
5. 📝 **Documentation:** Code và báo cáo rõ ràng, chuyên nghiệp

**Expected final results:**

- Trained YOLOv8 model với mAP@0.5 ~ 0.75-0.85
- Báo cáo 10-15 trang chất lượng cao
- Code base sạch sẽ, dễ reproduce
- Demo impresssive với predictions chính xác

Chúc bạn thực hiện project thành công! 🚀

---

**Lưu ý cuối:** Document này là roadmap, trong quá trình thực hiện có thể cần điều chỉnh linh hoạt dựa trên kết quả thực tế. Đừng ngần ngại experiment và tìm hiểu thêm!
