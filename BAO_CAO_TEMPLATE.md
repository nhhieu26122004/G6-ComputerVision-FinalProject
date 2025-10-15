# BÁO CÁO BÀI TẬP LỚN

## MÔN: THỊ GIÁC MÁY TÍNH

---

### THÔNG TIN ĐỀ TÀI

**Tên đề tài:** Phân loại rác thải thông minh bằng thị giác máy tính trong hệ thống dây chuyền xử lý

**Nhóm sinh viên thực hiện:** Nhóm 6

**Danh sách thành viên:**

- Nguyễn Huy Hiếu - MSSV: 22010160
- Vũ Tuấn Anh - MSSV: [MSSV2]

**Giảng viên hướng dẫn:** Nguyễn Văn Tới

**Thời gian thực hiện:** 10/2025

---

## MỤC LỤC

1. [GIỚI THIỆU](#1-giới-thiệu)

   - 1.1. Bối cảnh và động lực
   - 1.2. Mục tiêu nghiên cứu
   - 1.3. Phạm vi đề tài
   - 1.4. Cấu trúc báo cáo

2. [CƠ SỞ LÝ THUYẾT](#2-cơ-sở-lý-thuyết)

   - 2.1. Computer Vision và Object Detection
   - 2.2. Convolutional Neural Networks (CNN)
   - 2.3. YOLO (You Only Look Once)
   - 2.4. YOLOv8 Architecture
   - 2.5. Transfer Learning
   - 2.6. Evaluation Metrics

3. [PHƯƠNG PHÁP THỰC HIỆN](#3-phương-pháp-thực-hiện)

   - 3.1. Tổng quan quy trình
   - 3.2. Dataset
   - 3.3. Môi trường thực nghiệm
   - 3.4. Cấu hình mô hình
   - 3.5. Quá trình training
   - 3.6. Evaluation methodology

4. [KẾT QUẢ VÀ THẢO LUẬN](#4-kết-quả-và-thảo-luận)

   - 4.1. Kết quả training
   - 4.2. Kết quả testing
   - 4.3. Phân tích chi tiết theo từng class
   - 4.4. Visualizations
   - 4.5. So sánh và đánh giá

5. [KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN](#5-kết-luận-và-hướng-phát-triển)

   - 5.1. Tóm tắt kết quả đạt được
   - 5.2. Hạn chế của hệ thống
   - 5.3. Hướng phát triển trong tương lai

6. [TÀI LIỆU THAM KHẢO](#6-tài-liệu-tham-khảo)

7. [PHỤ LỤC](#7-phụ-lục)

---

## 1. GIỚI THIỆU

### 1.1. Bối cảnh và động lực

Trong bối cảnh ô nhiễm môi trường ngày càng trở nên nghiêm trọng, vấn đề quản lý và xử lý rác thải đang là một trong những thách thức lớn đối với các quốc gia trên toàn thế giới. Theo thống kê của Ngân hàng Thế giới (World Bank), lượng rác thải toàn cầu dự kiến sẽ tăng 70% vào năm 2050 so với mức hiện tại [1]. Việc phân loại rác thải đúng cách là bước quan trọng đầu tiên trong quy trình tái chế và xử lý rác hiệu quả.

Tuy nhiên, phân loại rác thủ công gặp nhiều hạn chế:

- **Chi phí nhân công cao:** Cần nhiều lao động với mức lương thấp
- **Hiệu suất thấp:** Tốc độ phân loại chậm, không đáp ứng được khối lượng lớn
- **Sai sót nhiều:** Con người dễ mắc sai lầm khi làm việc liên tục
- **Môi trường làm việc độc hại:** Tiếp xúc với rác thải gây ảnh hưởng sức khỏe

Với sự phát triển mạnh mẽ của trí tuệ nhân tạo (AI) và thị giác máy tính (Computer Vision), việc xây dựng hệ thống phân loại rác thải tự động đã trở nên khả thi và hiệu quả hơn bao giờ hết. Deep Learning, đặc biệt là các mô hình Object Detection như YOLO (You Only Look Once), đã chứng minh khả năng vượt trội trong việc phát hiện và phân loại đối tượng trong thời gian thực [2].

### 1.2. Mục tiêu nghiên cứu

Mục tiêu chính của đề tài này là **xây dựng hệ thống phân loại rác thải tự động sử dụng YOLOv8** để áp dụng trong dây chuyền xử lý công nghiệp. Cụ thể:

**Mục tiêu kỹ thuật:**

- Xây dựng và huấn luyện mô hình YOLOv8 để phát hiện và phân loại 4 loại rác thải phổ biến:
  - **Plastic (Nhựa):** Chai nhựa, túi nilon, hộp nhựa...
  - **Metal (Kim loại):** Lon nhôm, đồ hộp, kim loại tái chế...
  - **Paper (Giấy):** Giấy báo, carton, hộp giấy...
  - **Glass (Thủy tinh):** Chai thủy tinh, lọ thủy tinh...
- Đạt độ chính xác **mAP@0.5 ≥ 70%** trên test set
- Tối ưu hóa tốc độ inference để có thể xử lý real-time (≥ 30 FPS)

**Mục tiêu ứng dụng:**

- Tạo ra giải pháp có thể triển khai thực tế trong nhà máy xử lý rác
- Giảm chi phí nhân công và tăng hiệu suất phân loại
- Đóng góp vào việc bảo vệ môi trường thông qua tái chế hiệu quả hơn

### 1.3. Phạm vi đề tài

**Phạm vi nghiên cứu:**

- Tập trung vào 4 loại rác thải chính: Plastic, Metal, Paper, Glass
- Sử dụng YOLOv8 (phiên bản mới nhất tại thời điểm thực hiện)
- Dataset: Hình ảnh thu thập từ nguồn công khai
- Môi trường: Google Colab với GPU T4

**Ngoài phạm vi:**

- Không xây dựng phần cứng (robot, băng chuyền cơ khí)
- Không xử lý rác hữu cơ, rác nguy hại
- Không triển khai production system hoàn chỉnh

### 1.4. Cấu trúc báo cáo

Báo cáo được cấu trúc thành 5 chương chính:

- Chương 1 giới thiệu bối cảnh, mục tiêu và phạm vi
- Chương 2 trình bày cơ sở lý thuyết về YOLO và YOLOv8
- Chương 3 mô tả chi tiết phương pháp thực hiện
- Chương 4 trình bày kết quả và thảo luận
- Chương 5 kết luận và đề xuất hướng phát triển

---

## 2. CƠ SỞ LÝ THUYẾT

### 2.1. Computer Vision và Object Detection

**Computer Vision** (Thị giác máy tính) là lĩnh vực nghiên cứu giúp máy tính có khả năng "nhìn" và hiểu thông tin từ hình ảnh và video. Trong Computer Vision, **Object Detection** (Phát hiện đối tượng) là bài toán quan trọng với mục tiêu:

1. **Localization:** Xác định vị trí của đối tượng (bounding box)
2. **Classification:** Phân loại đối tượng đó thuộc class nào

Object Detection khác với Image Classification:

- **Classification:** Input image → Output label (1 nhãn cho toàn ảnh)
- **Detection:** Input image → Output multiple [bbox, class, confidence] (nhiều objects trong 1 ảnh)

**Các approach chính trong Object Detection:**

1. **Two-stage detectors:** R-CNN, Fast R-CNN, Faster R-CNN

   - Stage 1: Region Proposal (đề xuất vùng có thể chứa object)
   - Stage 2: Classification và bounding box regression
   - Ưu điểm: Accuracy cao
   - Nhược điểm: Chậm, không phù hợp real-time

2. **One-stage detectors:** YOLO, SSD, RetinaNet
   - Dự đoán trực tiếp bounding box và class trong 1 forward pass
   - Ưu điểm: Nhanh, phù hợp real-time
   - Nhược điểm: Accuracy thấp hơn (nhưng đã được cải thiện nhiều)

### 2.2. Convolutional Neural Networks (CNN)

CNN là nền tảng của hầu hết các mô hình Computer Vision hiện đại. Kiến trúc CNN bao gồm:

**1. Convolutional Layer:**

- Sử dụng filters/kernels để trích xuất features
- Mỗi filter học một pattern cụ thể (edges, textures, shapes...)
- Output: Feature maps

**2. Pooling Layer:**

- Giảm kích thước spatial (downsampling)
- Max Pooling, Average Pooling
- Mục đích: Giảm computational cost, tạo translation invariance

**3. Fully Connected Layer:**

- Kết nối mọi neuron với layer trước
- Thực hiện classification cuối cùng

**Feature Hierarchy trong CNN:**

```
Low-level features (Layer đầu)
├── Edges, corners
├── Lines, curves
│
Mid-level features (Layer giữa)
├── Textures
├── Simple shapes
│
High-level features (Layer cuối)
├── Object parts (mắt, mũi nếu là face detection)
└── Complex patterns
```

### 2.3. YOLO (You Only Look Once)

YOLO là họ mô hình object detection one-stage, được giới thiệu bởi Joseph Redmon et al. năm 2016 [2]. Ý tưởng cốt lõi:

**"You Only Look Once" - Chỉ cần nhìn 1 lần:**

- Chia ảnh thành grid (ví dụ 13x13)
- Mỗi grid cell dự đoán:
  - Bounding boxes (x, y, w, h)
  - Confidence score cho mỗi box
  - Class probabilities
- Tất cả predictions được thực hiện song song trong 1 forward pass

**Evolution của YOLO:**

- **YOLOv1 (2016):** Ý tưởng revolutionary nhưng accuracy chưa cao
- **YOLOv2/YOLO9000 (2017):** Cải thiện accuracy, multi-scale training
- **YOLOv3 (2018):** Multi-scale predictions, Darknet-53 backbone
- **YOLOv4 (2020):** Thêm nhiều tricks (CSPNet, Mish activation, Mosaic augmentation)
- **YOLOv5 (2020):** PyTorch implementation, user-friendly
- **YOLOv6, YOLOv7 (2022):** Further optimizations
- **YOLOv8 (2023):** State-of-the-art, được sử dụng trong đề tài này

### 2.4. YOLOv8 Architecture

YOLOv8 là phiên bản mới nhất (2023) được phát triển bởi Ultralytics, với những cải tiến đáng kể về cả accuracy và speed.

**Kiến trúc tổng quan:**

```
Input Image (640×640×3)
        ↓
┌───────────────────┐
│  BACKBONE         │ ← Feature Extraction
│  (CSPDarknet)     │
│  - C2f modules    │
│  - SPPF           │
└───────────────────┘
        ↓
┌───────────────────┐
│  NECK             │ ← Multi-scale Feature Fusion
│  (PAN-FPN)        │
│  - Bottom-up      │
│  - Top-down       │
└───────────────────┘
        ↓
┌───────────────────┐
│  HEAD             │ ← Detection Head
│  (Decoupled)      │
│  - Classification │
│  - Localization   │
└───────────────────┘
        ↓
Output: [x, y, w, h, confidence, class_probs]
```

**Chi tiết các thành phần:**

**1. Backbone (CSPDarknet với C2f):**

- Trách nhiệm: Trích xuất features từ input image
- **C2f (Cross Stage Partial with 2 convolutions):**
  - Cải tiến từ C3 của YOLOv5
  - Tăng gradient flow, giảm parameters
  - Kết hợp features từ nhiều scales
- **SPPF (Spatial Pyramid Pooling Fast):**
  - Pool features ở nhiều scales
  - Tăng receptive field mà không tăng computational cost

**2. Neck (PAN - Path Aggregation Network):**

- Kết hợp features từ backbone ở nhiều resolutions
- **Top-down pathway:** Truyền semantic information từ high-level
- **Bottom-up pathway:** Truyền localization information từ low-level
- Mục đích: Phát hiện tốt cả objects nhỏ và lớn

**3. Head (Decoupled Detection Head):**

- Tách riêng classification và localization tasks
- **Classification branch:** Dự đoán class probabilities
- **Localization branch:** Dự đoán bounding box coordinates
- Decoupled design giúp cải thiện accuracy

**Loss Functions trong YOLOv8:**

1. **Classification Loss - VFL (Varifocal Loss):**

   - Cải tiến từ Focal Loss
   - Giảm ảnh hưởng của easy samples
   - Focus vào hard samples

2. **Localization Loss - CIoU (Complete IoU):**

   ```
   CIoU = IoU - ρ²(b, b_gt)/c² - αv

   Trong đó:
   - IoU: Intersection over Union
   - ρ: Euclidean distance giữa center points
   - c: Diagonal length của smallest enclosing box
   - α: Trade-off parameter
   - v: Đo aspect ratio consistency
   ```

3. **Distribution Focal Loss (DFL):**
   - Cải thiện độ chính xác của bounding box
   - Model học distribution thay vì single value

**Model Sizes:**

| Model   | Parameters | FLOPs  | mAP@0.5:0.95 | Speed (ms) |
| ------- | ---------- | ------ | ------------ | ---------- |
| YOLOv8n | 3.2M       | 8.7G   | 37.3%        | 1.2        |
| YOLOv8s | 11.2M      | 28.6G  | 44.9%        | 2.3        |
| YOLOv8m | 25.9M      | 78.9G  | 50.2%        | 4.5        |
| YOLOv8l | 43.7M      | 165.2G | 52.9%        | 6.8        |
| YOLOv8x | 68.2M      | 257.8G | 53.9%        | 10.1       |

_Trong đề tài này, nhóm sử dụng **YOLOv8n** (nano) để cân bằng giữa accuracy và training time._

### 2.5. Transfer Learning

**Transfer Learning** là kỹ thuật sử dụng model đã được train trên dataset lớn (source domain) và fine-tune trên dataset nhỏ hơn (target domain).

**Tại sao cần Transfer Learning?**

- Dataset nhỏ → Dễ overfitting nếu train from scratch
- Training from scratch tốn thời gian và tài nguyên
- Pretrained model đã học được các features cơ bản (edges, shapes...)

**Trong dự án này:**

- **Pretrained weights:** YOLOv8 pretrained trên COCO dataset (80 classes, 330K images)
- **Fine-tuning:** Chỉ cần điều chỉnh detection head cho 4 classes (Plastic, Metal, Paper, Glass)
- **Freezing strategy:**
  - Option 1: Freeze backbone, chỉ train head (fast nhưng accuracy thấp)
  - Option 2: Train toàn bộ với learning rate nhỏ (recommended)

### 2.6. Evaluation Metrics

Các metrics đánh giá Object Detection:

**1. IoU (Intersection over Union):**

```
IoU = Area of Overlap / Area of Union
    = |A ∩ B| / |A ∪ B|

Trong đó:
- A: Predicted bounding box
- B: Ground truth bounding box
```

- IoU = 1.0: Perfect match
- IoU ≥ 0.5: Thường được coi là "correct detection"

**2. Precision và Recall:**

```
Precision = TP / (TP + FP)
          = Tỷ lệ predictions đúng trong tổng số predictions

Recall = TP / (TP + FN)
       = Tỷ lệ objects được detect trong tổng số objects thật

Trong đó:
- TP (True Positive): Detect đúng object đúng class
- FP (False Positive): Detect nhầm (object không tồn tại hoặc sai class)
- FN (False Negative): Miss object (không detect được)
```

**3. Average Precision (AP):**

- Area under Precision-Recall curve
- AP@0.5: Tính AP với IoU threshold = 0.5
- AP@0.75: Tính AP với IoU threshold = 0.75 (khó hơn)

**4. mAP (mean Average Precision):**

```
mAP = (AP_class1 + AP_class2 + ... + AP_classN) / N
```

- **mAP@0.5:** Average của AP@0.5 across all classes (COCO style)
- **mAP@0.5:0.95:** Average của AP với IoU từ 0.5 đến 0.95 (step 0.05)
  - Metric khó hơn, comprehensive hơn
  - Standard trong COCO benchmark

**5. F1-Score:**

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

- Harmonic mean của Precision và Recall
- Cân bằng giữa hai metrics

**Confusion Matrix:**

- Ma trận N×N (N = số classes)
- Hàng: True labels
- Cột: Predicted labels
- Diagonal: Correct predictions
- Off-diagonal: Confusions giữa các classes

---

## 3. PHƯƠNG PHÁP THỰC HIỆN

### 3.1. Tổng quan quy trình

Quy trình thực hiện dự án được chia thành 5 giai đoạn chính:

```
[1. Dataset Collection]
        ↓
[2. Data Preparation]
        ↓
[3. Model Training]
        ↓
[4. Model Evaluation]
        ↓
[5. Testing & Deployment]
```

**Timeline:**

- Giai đoạn 1-2: Ngày 1 (6-8 giờ)
- Giai đoạn 3: Ngày 2 (training overnight)
- Giai đoạn 4-5: Ngày 3 (4-6 giờ)

### 3.2. Dataset

#### 3.2.1. Nguồn dữ liệu

**Dataset được sử dụng:** [Tên dataset - ví dụ: "Waste Detection Dataset from Roboflow"]

**Nguồn:** [URL - ví dụ: https://universe.roboflow.com/...]

**Đặc điểm dataset:**

- **Tổng số ảnh:** [Số lượng] images
- **Format:** YOLO format (`.txt` annotations)
- **Resolution:** [WxH] pixels trung bình
- **Nguồn ảnh:** [Mô tả - ví dụ: Ảnh chụp từ conveyor belts, waste facilities...]

#### 3.2.2. Thống kê dataset

**Phân chia dữ liệu:**

| Split      | Số ảnh | Tỷ lệ | Mục đích                   |
| ---------- | ------ | ----- | -------------------------- |
| Training   | [XXX]  | 70%   | Huấn luyện model           |
| Validation | [XXX]  | 20%   | Tune hyperparameters       |
| Test       | [XXX]  | 10%   | Đánh giá final performance |

**Phân bố theo class:**

| Class   | Train | Valid | Test | Tổng  |
| ------- | ----- | ----- | ---- | ----- |
| Plastic | [XX]  | [XX]  | [XX] | [XXX] |
| Metal   | [XX]  | [XX]  | [XX] | [XXX] |
| Paper   | [XX]  | [XX]  | [XX] | [XXX] |
| Glass   | [XX]  | [XX]  | [XX] | [XXX] |

_[Chèn biểu đồ bar chart phân bố classes]_

**Nhận xét:**

- [Ví dụ: Dataset khá balanced, các classes có số lượng tương đương]
- [Ví dụ: Paper class có ít ảnh nhất, có thể cần augmentation thêm]

#### 3.2.3. Cấu trúc dataset

Dataset được tổ chức theo chuẩn YOLO:

```
dataset/
├── train/
│   ├── images/
│   │   ├── img_0001.jpg
│   │   ├── img_0002.jpg
│   │   └── ...
│   └── labels/
│       ├── img_0001.txt
│       ├── img_0002.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

**Format annotation (`.txt` file):**

```
class_id x_center y_center width height
```

- Tất cả values được normalize về [0, 1]
- `class_id`: 0=Plastic, 1=Metal, 2=Paper, 3=Glass

**File `data.yaml`:**

```yaml
train: ./dataset/train/images
val: ./dataset/valid/images
test: ./dataset/test/images

nc: 4 # number of classes
names: ["Plastic", "Metal", "Paper", "Glass"]
```

#### 3.2.4. Data Preprocessing

**Preprocessing steps:**

1. **Resize:** Tất cả ảnh resize về 640×640 (YOLOv8 input size)
2. **Normalization:** Pixel values normalize về [0, 1]
3. **Data Augmentation:** (tự động bởi YOLOv8)
   - Random scaling (±50%)
   - Translation (±10%)
   - Rotation (±10°)
   - Horizontal flip (50%)
   - Mosaic augmentation
   - HSV color jittering

_[Chèn hình: Ví dụ augmented images]_

### 3.3. Môi trường thực nghiệm

**Hardware:**

- **Platform:** Google Colab
- **GPU:** Tesla T4 (15GB VRAM)
- **RAM:** 12GB
- **Storage:** Google Drive (50GB)

**Software:**

- **Python:** 3.10
- **PyTorch:** 2.0.1
- **CUDA:** 11.8
- **Ultralytics:** 8.0.200
- **OS:** Ubuntu 22.04 (Colab environment)

**Thư viện chính:**

```
ultralytics==8.0.200
torch==2.0.1
torchvision==0.15.0
opencv-python==4.8.0
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
```

### 3.4. Cấu hình mô hình

**Model chọn:** YOLOv8n (nano)

**Lý do chọn YOLOv8n:**

- ✅ Lightweight (3.2M parameters) → Train nhanh trên Colab
- ✅ Inference speed cao (~ 1.2ms) → Phù hợp real-time
- ✅ Đủ accuracy cho bài toán 4 classes
- ✅ Phù hợp với thời gian 3-4 ngày

**Pretrained weights:** `yolov8n.pt` (trained on COCO dataset)

**Hyperparameters:**

| Parameter        | Value  | Mô tả                                |
| ---------------- | ------ | ------------------------------------ |
| `epochs`         | 100    | Số epochs training                   |
| `batch_size`     | 16     | Batch size (giới hạn bởi GPU memory) |
| `imgsz`          | 640    | Input image size                     |
| `optimizer`      | AdamW  | Optimizer với weight decay           |
| `lr0`            | 0.01   | Initial learning rate                |
| `lrf`            | 0.01   | Final learning rate (lr0 × lrf)      |
| `momentum`       | 0.937  | SGD momentum / Adam beta1            |
| `weight_decay`   | 0.0005 | L2 regularization                    |
| `warmup_epochs`  | 3      | Warmup epochs                        |
| `patience`       | 20     | Early stopping patience              |
| `conf_threshold` | 0.25   | Confidence threshold for inference   |
| `iou_threshold`  | 0.6    | IoU threshold for NMS                |

**Augmentation parameters:**

| Parameter   | Value | Mô tả                           |
| ----------- | ----- | ------------------------------- |
| `hsv_h`     | 0.015 | Hue augmentation                |
| `hsv_s`     | 0.7   | Saturation augmentation         |
| `hsv_v`     | 0.4   | Value augmentation              |
| `degrees`   | 10.0  | Rotation (±degrees)             |
| `translate` | 0.1   | Translation (±fraction)         |
| `scale`     | 0.5   | Scaling (±fraction)             |
| `flipud`    | 0.0   | Vertical flip probability       |
| `fliplr`    | 0.5   | Horizontal flip probability     |
| `mosaic`    | 1.0   | Mosaic augmentation probability |

### 3.5. Quá trình training

#### 3.5.1. Training code

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Training
results = model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    save=True,
    device=0,
    project='runs/waste_detection',
    name='yolov8n_exp1',
    exist_ok=True,
    pretrained=True,
    optimizer='AdamW',
    verbose=True,
    seed=42,

    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
)
```

#### 3.5.2. Training process monitoring

**Metrics được theo dõi:**

- **Loss functions:**
  - `box_loss`: Localization loss
  - `cls_loss`: Classification loss
  - `dfl_loss`: Distribution focal loss
- **Performance metrics:**
  - `Precision`
  - `Recall`
  - `mAP@0.5`
  - `mAP@0.5:0.95`

**Early stopping:**

- Monitor: `mAP@0.5` trên validation set
- Patience: 20 epochs
- Nếu không improve sau 20 epochs → Stop training

#### 3.5.3. Training time

**Thời gian training:**

- **Tổng thời gian:** [XX] giờ [YY] phút
- **Thời gian/epoch:** ~[ZZ] giây
- **Best model:** Epoch [N] với mAP@0.5 = [X.XXX]

### 3.6. Evaluation methodology

**Evaluation protocol:**

1. **Validation during training:**

   - Validate sau mỗi epoch
   - Metrics: mAP@0.5, mAP@0.5:0.95, Precision, Recall

2. **Final testing:**

   - Load best model (`best.pt`)
   - Evaluate trên test set
   - Generate confusion matrix, PR curves

3. **Confidence threshold tuning:**

   - Test với nhiều confidence thresholds: [0.1, 0.25, 0.5, 0.7]
   - Chọn threshold tối ưu dựa trên F1-score

4. **Qualitative analysis:**
   - Visualize predictions trên test images
   - Phân tích failure cases

---

## 4. KẾT QUẢ VÀ THẢO LUẬN

### 4.1. Kết quả training

#### 4.1.1. Training convergence

Training process kéo dài [XX] epochs (early stopped tại epoch [YY]).

_[Chèn hình: Loss curves - box_loss, cls_loss, dfl_loss theo epochs]_

**Hình 4.1:** Loss curves trong quá trình training

**Quan sát:**

- Loss giảm đều và ổn định, không có dấu hiệu overfitting
- `box_loss` hội tụ nhanh trong 20 epochs đầu
- `cls_loss` giảm chậm hơn, cho thấy classification là challenging hơn
- Validation loss track gần với training loss → Model generalize tốt

#### 4.1.2. Performance metrics evolution

_[Chèn hình: Precision, Recall, mAP@0.5, mAP@0.5:0.95 theo epochs]_

**Hình 4.2:** Evolution của performance metrics

**Bảng 4.1:** Best metrics achieved during training

| Metric       | Value  | Epoch |
| ------------ | ------ | ----- |
| mAP@0.5      | [0.XX] | [YY]  |
| mAP@0.5:0.95 | [0.XX] | [YY]  |
| Precision    | [0.XX] | [YY]  |
| Recall       | [0.XX] | [YY]  |
| F1-Score     | [0.XX] | [YY]  |

**Nhận xét:**

- [Ví dụ: mAP@0.5 đạt 0.78 tại epoch 87, vượt mục tiêu 0.70]
- [Ví dụ: Precision và Recall cân bằng (~0.75), cho thấy model không bias]
- [Ví dụ: mAP@0.5:0.95 = 0.534, acceptable cho real-world application]

### 4.2. Kết quả testing

#### 4.2.1. Overall performance

Model được evaluate trên test set ([XXX] images) với confidence threshold = 0.25.

**Bảng 4.2:** Final test results

| Metric         | Value   | Notes                                 |
| -------------- | ------- | ------------------------------------- |
| mAP@0.5        | [0.XX]  | Main metric (target ≥ 0.70) ✅        |
| mAP@0.5:0.95   | [0.XX]  | COCO-style metric                     |
| Precision      | [0.XX]  | Tỷ lệ predictions đúng                |
| Recall         | [0.XX]  | Tỷ lệ objects được detect             |
| F1-Score       | [0.XX]  | Harmonic mean của P và R              |
| Inference time | [X.X]ms | Average per image (640×640) on T4 GPU |
| FPS            | [XXX]   | Frames per second                     |

**Đánh giá:**

- ✅ **mAP@0.5 ≥ 0.70:** [Đạt/Không đạt] mục tiêu
- ✅ **Real-time capability:** [XXX] FPS >> 30 FPS required
- ✅ **Precision-Recall balance:** F1-score = [0.XX] cho thấy cân bằng tốt

#### 4.2.2. Confusion Matrix

_[Chèn hình: Confusion Matrix 4×4]_

**Hình 4.3:** Confusion matrix trên test set

**Phân tích:**

- **Diagonal (correct predictions):** [Mô tả - ví dụ: Plastic 85%, Metal 82%, Paper 76%, Glass 79%]
- **Main confusions:**
  - [Ví dụ: Paper ↔ Plastic: 12% - Do màu sắc tương tự]
  - [Ví dụ: Glass ↔ Background: 8% - Do glass trong suốt, khó detect]
- **Background FP:** [X%] - False detections trên background

### 4.3. Phân tích chi tiết theo từng class

**Bảng 4.3:** Per-class performance metrics

| Class   | AP@0.5 | AP@0.5:0.95 | Precision | Recall | F1     | Sample Count |
| ------- | ------ | ----------- | --------- | ------ | ------ | ------------ |
| Plastic | [0.XX] | [0.XX]      | [0.XX]    | [0.XX] | [0.XX] | [XXX]        |
| Metal   | [0.XX] | [0.XX]      | [0.XX]    | [0.XX] | [0.XX] | [XXX]        |
| Paper   | [0.XX] | [0.XX]      | [0.XX]    | [0.XX] | [0.XX] | [XXX]        |
| Glass   | [0.XX] | [0.XX]      | [0.XX]    | [0.XX] | [0.XX] | [XXX]        |

_[Chèn biểu đồ: Bar chart so sánh AP@0.5 của 4 classes]_

**Hình 4.4:** So sánh Average Precision giữa các classes

#### 4.3.1. Plastic (Nhựa)

**Performance:** AP@0.5 = [0.XX]

**Đặc điểm:**

- [Ví dụ: Plastic có đặc trưng rõ ràng: màu sắc đa dạng, hình dạng đặc trưng (chai, túi)]
- [Ví dụ: Dễ phân biệt với các class khác]

**Typical predictions:**
_[Chèn 2-3 hình ảnh predictions tốt cho Plastic]_

**Failure cases:**

- [Ví dụ: Nhầm với Paper khi plastic bags màu trắng]
- [Ví dụ: Miss detection với plastic trong suốt]

#### 4.3.2. Metal (Kim loại)

**Performance:** AP@0.5 = [0.XX]

**Đặc điểm:**

- [Ví dụ: Metal có texture và reflection đặc trưng]
- [Ví dụ: Hình dạng thường là lon, hộp đều đặn]

**Typical predictions:**
_[Chèn 2-3 hình predictions cho Metal]_

**Failure cases:**

- [Ví dụ: Nhầm với glass khi có ánh sáng phản chiếu tương tự]

#### 4.3.3. Paper (Giấy)

**Performance:** AP@0.5 = [0.XX]

**Đặc điểm:**

- [Ví dụ: Paper có performance thấp nhất do texture phức tạp]
- [Ví dụ: Dễ nhầm với background (cùng màu sắc nhạt)]

**Typical predictions:**
_[Chèn hình predictions cho Paper]_

**Failure cases:**

- [Ví dụ: Giấy bị nhăn, rách → khó detect]
- [Ví dụ: Cardboard boxes lớn → bounding box không chính xác]

#### 4.3.4. Glass (Thủy tinh)

**Performance:** AP@0.5 = [0.XX]

**Đặc điểm:**

- [Ví dụ: Glass challenging do tính trong suốt]
- [Ví dụ: Phụ thuộc nhiều vào lighting conditions]

**Typical predictions:**
_[Chèn hình predictions cho Glass]_

**Failure cases:**

- [Ví dụ: Glass bottles trong suốt → low confidence]
- [Ví dụ: Nhầm với metal khi có label kim loại trên glass]

### 4.4. Visualizations

#### 4.4.1. Successful predictions

_[Chèn 6-8 hình: Predictions thành công với high confidence]_

**Hình 4.5:** Các trường hợp model predictions chính xác

**Quan sát:**

- Model detect tốt khi objects có:
  - Lighting đều
  - Góc nhìn rõ ràng
  - Không bị occlusion
  - Đặc trưng class rõ ràng

#### 4.4.2. Failure cases

_[Chèn 4-6 hình: False positives, False negatives, Wrong classifications]_

**Hình 4.6:** Các trường hợp model predictions sai

**Phân loại failure cases:**

1. **False Positives (FP):**

   - [Ví dụ: Detect background objects như plastic]
   - [Nguyên nhân: Texture tương tự với training data]

2. **False Negatives (FN):**

   - [Ví dụ: Miss objects nhỏ hoặc bị che khuất]
   - [Nguyên nhân: Occlusion, small object size]

3. **Wrong Classification:**

   - [Ví dụ: Plastic được classify thành Paper]
   - [Nguyên nhân: Màu sắc và texture tương tự]

4. **Localization Errors:**
   - [Ví dụ: Bounding box không chính xác (IoU < 0.5)]
   - [Nguyên nhân: Object shape phức tạp]

#### 4.4.3. Precision-Recall Curves

_[Chèn hình: PR curves cho 4 classes]_

**Hình 4.7:** Precision-Recall curves

**Diễn giải:**

- Curves càng gần góc trên bên phải càng tốt
- [Class X] có PR curve tốt nhất → Easy to detect and classify
- [Class Y] có PR curve thấp → Need improvement

### 4.5. So sánh và đánh giá

#### 4.5.1. So sánh với baseline

**Bảng 4.4:** So sánh với các approaches khác

| Approach           | mAP@0.5 | Inference Time | Notes                     |
| ------------------ | ------- | -------------- | ------------------------- |
| **YOLOv8n (Ours)** | [0.XX]  | [X.X]ms        | Balanced speed & accuracy |
| YOLOv5s (baseline) | [0.XX]  | [X.X]ms        | Older version             |
| YOLOv8s            | [0.XX]  | [X.X]ms        | Higher accuracy, slower   |
| Faster R-CNN       | [0.XX]  | [XX]ms         | High accuracy, very slow  |

_Note: Baseline numbers từ papers/previous works_

**Nhận xét:**

- YOLOv8n đạt good trade-off giữa accuracy và speed
- [Ví dụ: Chậm hơn YOLOv5n 15% nhưng accuracy tăng 8%]

#### 4.5.2. Ablation study (Optional)

Nếu có thời gian, test các configurations khác:

**Bảng 4.5:** Ablation experiments

| Configuration         | mAP@0.5 | Change  |
| --------------------- | ------- | ------- |
| Baseline (all tricks) | [0.XX]  | -       |
| - Mosaic augmentation | [0.XX]  | [-X.X%] |
| - Transfer learning   | [0.XX]  | [-X.X%] |
| + Larger image size   | [0.XX]  | [+X.X%] |

#### 4.5.3. Confidence threshold analysis

_[Chèn hình: Precision-Recall-F1 vs Confidence threshold]_

**Hình 4.8:** Impact của confidence threshold

**Bảng 4.6:** Metrics ở các confidence thresholds

| Conf | Precision | Recall | F1     | mAP@0.5 |
| ---- | --------- | ------ | ------ | ------- |
| 0.10 | [0.XX]    | [0.XX] | [0.XX] | [0.XX]  |
| 0.25 | [0.XX]    | [0.XX] | [0.XX] | [0.XX]  |
| 0.50 | [0.XX]    | [0.XX] | [0.XX] | [0.XX]  |
| 0.70 | [0.XX]    | [0.XX] | [0.XX] | [0.XX]  |

**Optimal threshold:** [0.XX] (maximize F1-score)

**Trade-off:**

- Confidence thấp → High Recall, Low Precision (nhiều FP)
- Confidence cao → High Precision, Low Recall (nhiều FN)

#### 4.5.4. Strengths & Weaknesses

**Điểm mạnh:**

- ✅ [Ví dụ: Accuracy cao trên 3/4 classes (Plastic, Metal, Glass)]
- ✅ [Ví dụ: Speed nhanh, đáp ứng real-time (> 50 FPS)]
- ✅ [Ví dụ: Robust với lighting conditions khác nhau]
- ✅ [Ví dụ: Model size nhỏ (6MB), dễ deploy]

**Điểm yếu:**

- ❌ [Ví dụ: Paper class có accuracy thấp nhất]
- ❌ [Ví dụ: Struggle với transparent/translucent objects (glass, clear plastic)]
- ❌ [Ví dụ: Performance giảm khi objects bị occlusion nặng]
- ❌ [Ví dụ: Small objects (< 32×32 pixels) khó detect]

---

## 5. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 5.1. Tóm tắt kết quả đạt được

Trong dự án này, nhóm đã thực hiện thành công việc **xây dựng hệ thống phân loại rác thải tự động sử dụng YOLOv8** với các kết quả chính:

**Kết quả kỹ thuật:**

- ✅ **Model:** YOLOv8n được fine-tune trên dataset [tên dataset]
- ✅ **Performance:** mAP@0.5 = [0.XX], vượt target 0.70
- ✅ **Speed:** [XXX] FPS trên Tesla T4 GPU, đáp ứng real-time
- ✅ **Deployment:** Model size nhỏ (6MB), dễ dàng deploy

**Kết quả về mặt học thuật:**

- Hiểu sâu về Object Detection và kiến trúc YOLOv8
- Nắm vững quy trình training, evaluation, và optimization
- Biết phân tích kết quả và troubleshoot issues

**Kết quả ứng dụng:**

- Tạo ra proof-of-concept có thể triển khai thực tế
- Đóng góp vào giải quyết vấn đề môi trường qua automation

### 5.2. Hạn chế của hệ thống

**Hạn chế kỹ thuật:**

1. **Accuracy chưa đồng đều giữa các classes:**

   - Paper class có performance thấp nhất ([0.XX])
   - Cần thêm data hoặc model lớn hơn

2. **Challenging scenarios:**

   - Transparent objects (glass bottles, clear plastic)
   - Heavy occlusion (objects chồng lên nhau)
   - Poor lighting conditions

3. **Dataset limitations:**
   - Dataset chỉ [XXX] ảnh, có thể chưa đủ đa dạng
   - Chủ yếu từ controlled environment, chưa test real-world đủ

**Hạn chế về deployment:**

- Chỉ test trên GPU, chưa optimize cho CPU/Edge devices
- Chưa có user interface hoặc API wrapper
- Chưa integrate với hardware (conveyor belt, robot arm)

### 5.3. Hướng phát triển trong tương lai

**Cải thiện model (Short-term):**

1. **Tăng accuracy:**

   - Thu thập thêm data, đặc biệt cho Paper class
   - Thử YOLOv8s hoặc YOLOv8m (model lớn hơn)
   - Advanced augmentation (CutOut, MixUp, CutMix)
   - Ensemble multiple models

2. **Tối ưu performance:**

   - Model quantization (FP32 → INT8)
   - Pruning và knowledge distillation
   - TensorRT optimization cho inference speed

3. **Mở rộng classes:**
   - Thêm các loại rác: Organic, E-waste, Hazardous waste
   - Multi-label classification (object có thể thuộc nhiều categories)

**Triển khai thực tế (Medium-term):**

1. **System integration:**

   - Xây dựng API service (FastAPI, Flask)
   - Tạo web interface để monitoring
   - Integrate với PLC/robot arm control

2. **Edge deployment:**

   - Deploy trên Raspberry Pi, Jetson Nano
   - ONNX export cho cross-platform
   - Mobile deployment (iOS/Android)

3. **Production features:**
   - Logging và monitoring system
   - Continuous learning từ production data
   - A/B testing framework

**Nghiên cứu nâng cao (Long-term):**

1. **3D Detection:**

   - Sử dụng depth cameras (RGB-D)
   - Estimate object volume và weight

2. **Multi-modal learning:**

   - Kết hợp vision với sensors khác (NIR, X-ray)
   - Improve accuracy cho challenging materials

3. **Robotic manipulation:**
   - Grasping point detection
   - Trajectory planning cho robot arm
   - Closed-loop control system

**Societal impact:**

- Deploy hệ thống ở waste treatment facilities thực tế
- Đo lường impact: tốc độ xử lý, chi phí giảm, tỷ lệ tái chế tăng
- Mở rộng ra các khu vực khác (bãi rác, households)

### 5.4. Bài học kinh nghiệm

**Technical lessons:**

- Transfer learning rất hiệu quả, quan trọng phải có pretrained weights tốt
- Data quality > quantity: annotations chính xác quan trọng hơn số lượng ảnh
- Monitoring training process kỹ để detect overfitting sớm
- Ablation study giúp hiểu rõ contribution của từng component

**Project management lessons:**

- Timeline planning quan trọng, đặc biệt với thời gian training
- Backup code và weights thường xuyên (Colab dễ disconnect)
- Documentation trong quá trình làm, không để cuối mới viết báo cáo

**Collaboration lessons:**

- Chia task rõ ràng giữa các thành viên
- Regular sync-up để check progress
- Version control (Git) rất quan trọng

---

## 6. TÀI LIỆU THAM KHẢO

[1] World Bank. (2018). "What a Waste 2.0: A Global Snapshot of Solid Waste Management to 2050." World Bank Publications.

[2] Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). "You Only Look Once: Unified, Real-Time Object Detection." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 779-788.

[3] Jocher, G., Chaurasia, A., & Qiu, J. (2023). "Ultralytics YOLOv8." GitHub repository. https://github.com/ultralytics/ultralytics

[4] Lin, T. Y., Maire, M., Belongie, S., et al. (2014). "Microsoft COCO: Common Objects in Context." In European Conference on Computer Vision (ECCV), pp. 740-755.

[5] Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). "YOLOv4: Optimal Speed and Accuracy of Object Detection." arXiv preprint arXiv:2004.10934.

[6] [Dataset citation - ví dụ: "Waste Detection Dataset." Roboflow Universe, 2023. https://universe.roboflow.com/...]

[7] He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778.

[8] Tan, M., Pang, R., & Le, Q. V. (2020). "EfficientDet: Scalable and Efficient Object Detection." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10781-10790.

[9] Pan, S. J., & Yang, Q. (2010). "A Survey on Transfer Learning." IEEE Transactions on Knowledge and Data Engineering, 22(10), 1345-1359.

[10] Shorten, C., & Khoshgoftaar, T. M. (2019). "A Survey on Image Data Augmentation for Deep Learning." Journal of Big Data, 6(1), 1-48.

_[Thêm các tài liệu khác nếu có]_

---

## 7. PHỤ LỤC

### Phụ lục A: Source Code

#### A.1. Training Script (`train.py`)

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
        seed=42,

        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
    )

    print("Training completed!")
    print(f"Best model: {args.project}/{args.name}/weights/best.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--name', type=str, default='waste_yolov8')

    args = parser.parse_args()
    train(args)
```

#### A.2. Testing Script (`test.py`)

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
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.6)

    args = parser.parse_args()
    test(args)
```

#### A.3. Inference Script (`inference.py`)

```python
"""
Inference script for single image/video
"""
from ultralytics import YOLO
import argparse

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
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.6)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--show', action='store_true')

    args = parser.parse_args()
    inference(args)
```

### Phụ lục B: Dataset Statistics

**Bảng B.1:** Detailed dataset statistics

| Metric                    | Value  |
| ------------------------- | ------ |
| Total images              | [XXXX] |
| Total annotations         | [XXXX] |
| Average objects per image | [X.XX] |
| Min objects per image     | [X]    |
| Max objects per image     | [XX]   |
| Average image resolution  | [WxH]  |
| Dataset size (GB)         | [X.XX] |

**Bảng B.2:** Object size distribution

| Size Category | Percentage | Definition (pixels) |
| ------------- | ---------- | ------------------- |
| Small         | [XX%]      | < 32×32             |
| Medium        | [XX%]      | 32×32 to 96×96      |
| Large         | [XX%]      | > 96×96             |

### Phụ lục C: Hyperparameter Tuning Log

**Bảng C.1:** Experiments với different hyperparameters

| Exp | Model   | Batch | LR    | Epochs | mAP@0.5 | Notes        |
| --- | ------- | ----- | ----- | ------ | ------- | ------------ |
| 1   | YOLOv8n | 16    | 0.01  | 100    | [0.XX]  | Baseline     |
| 2   | YOLOv8n | 32    | 0.01  | 100    | [0.XX]  | Larger batch |
| 3   | YOLOv8n | 16    | 0.005 | 100    | [0.XX]  | Lower LR     |
| 4   | YOLOv8s | 16    | 0.01  | 100    | [0.XX]  | Larger model |

### Phụ lục D: Error Analysis

**Bảng D.1:** Error breakdown

| Error Type           | Count | Percentage | Main Causes            |
| -------------------- | ----- | ---------- | ---------------------- |
| False Positive       | [XX]  | [XX%]      | Background similarity  |
| False Negative       | [XX]  | [XX%]      | Occlusion, small size  |
| Wrong Classification | [XX]  | [XX%]      | Inter-class similarity |
| Localization Error   | [XX]  | [XX%]      | Complex shapes         |

---

## KẾT THÚC BÁO CÁO

**Ngày hoàn thành:** [DD/MM/YYYY]

**Chữ ký các thành viên:**

- Nguyễn Huy Hiếu: Hiếu
- Vũ Tuấn Anh: \***\*\_\_\_\*\***

**Xác nhận của giảng viên hướng dẫn:**

---

[Họ tên - Chữ ký]
