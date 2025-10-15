# 📘 HƯỚNG DẪN XỬ LÝ DATASET - XÓA CARDBOARD VÀ RANDOM TRASH

## 🎯 Mục tiêu

Xóa 2 classes không cần thiết (Cardboard và Random Trash) khỏi dataset, chỉ giữ lại **4 classes chính**:

- **Plastic** (ID 0)
- **Metal** (ID 1)
- **Paper** (ID 2)
- **Glass** (ID 3)

---

## 📋 Yêu cầu trước khi bắt đầu

### 1. Dataset đã download và extract

Cấu trúc dataset cần có dạng:

```
dataset/
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── labels/
│       ├── img_001.txt
│       ├── img_002.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── test/ (optional)
    ├── images/
    └── labels/
```

### 2. Python đã cài đặt

```bash
python --version
# Cần Python 3.6 trở lên
```

---

## 🚀 CÁCH SỬ DỤNG

### Bước 1: Kiểm tra dataset gốc (TRƯỚC KHI XỬ LÝ)

```bash
python verify_dataset.py
```

**Output mẫu:**

```
📂 Nhập đường dẫn tới dataset: dataset

=============================================================
🔍 VERIFY DATASET
=============================================================

📊 TRAIN
-------------------------------------------------------------
📁 Files:
   - Images: 800
   - Labels: 800
   ✅ Images và labels khớp nhau!

🎯 Objects:
   - Tổng: 2546 objects

📦 Class Distribution:
   - cardboard (ID 0):  403 ( 15.8%)
   - glass     (ID 1):  511 ( 20.1%)
   - metal     (ID 2):  411 ( 16.1%)
   - paper     (ID 3):  603 ( 23.7%)
   - plastic   (ID 4):  480 ( 18.9%)
   - random_tr (ID 5):  138 (  5.4%)
```

→ Xem class distribution trước khi xử lý

---

### Bước 2: Xử lý dataset - Xóa 2 classes

```bash
python process_dataset.py
```

**Script sẽ hỏi:**

```
📂 Nhập đường dẫn tới dataset folder: dataset
```

→ Nhập: `dataset` (hoặc đường dẫn tương ứng)

**Script sẽ:**

1. **Dry run** đầu tiên - Hiển thị preview những gì sẽ thay đổi
2. Hỏi xác nhận: `⚠️  Bạn có chắc muốn THỰC HIỆN xóa/sửa files? (yes/no):`
3. **Tạo backup** tự động vào `dataset_backup/`
4. **Xử lý** dataset:
   - Remap class IDs: 1→3, 2→1, 3→2, 4→0
   - Xóa annotations của Cardboard (0) và Random Trash (5)
   - Xóa ảnh không còn annotations hợp lệ
5. **Tạo file `data.yaml`** mới với 4 classes

**Output mẫu:**

```
=============================================================
🔍 BƯỚC 1: DRY RUN (Xem trước, không thực sự xóa/sửa)
=============================================================

📊 THỐNG KÊ: TRAIN
=============================================================

📁 Files:
  - Tổng files: 800
  - Giữ lại: 750 (93.8%)
  - Xóa: 50 (6.2%)

🎯 Objects:
  - Trước: 2546 objects
  - Sau: 2005 objects
  - Giảm: 541 objects

📦 Class distribution TRƯỚC XỬ LÝ:
  - cardboard      (ID 0):  403 objects ( 15.8%)
  - glass          (ID 1):  511 objects ( 20.1%)
  - metal          (ID 2):  411 objects ( 16.1%)
  - paper          (ID 3):  603 objects ( 23.7%)
  - plastic        (ID 4):  480 objects ( 18.9%)
  - random_trash   (ID 5):  138 objects (  5.4%)

✅ Class distribution SAU XỬ LÝ (4 classes):
  - Plastic        (ID 0):  480 objects ( 23.9%)
  - Metal          (ID 1):  411 objects ( 20.5%)
  - Paper          (ID 2):  603 objects ( 30.1%)
  - Glass          (ID 3):  511 objects ( 25.5%)

=============================================================
📊 TỔNG KẾT
=============================================================
Tổng objects TRƯỚC: 2546
Tổng objects SAU: 2005
Sẽ XÓA: 541 objects (21.2%)

=============================================================
⚠️  Bạn có chắc muốn THỰC HIỆN xóa/sửa files? (yes/no): yes

💾 Đang tạo backup...
✅ Đã backup vào: dataset_backup

=============================================================
🔧 BƯỚC 2: THỰC HIỆN XỬ LÝ
=============================================================

⚙️  Đang xử lý: train...
  Xóa: dataset/train/images/img_123.jpg
  Xóa: dataset/train/images/img_456.jpg
  ...
✅ Hoàn thành: train

⚙️  Đang xử lý: valid...
✅ Hoàn thành: valid

=============================================================
📝 Tạo file data.yaml
=============================================================

✅ Đã tạo file: dataset/data.yaml

=============================================================
✅ HOÀN THÀNH!
=============================================================

📂 Dataset đã được xử lý: dataset
💾 Backup gốc tại: dataset_backup
📝 File config: dataset/data.yaml

🚀 Bạn có thể bắt đầu training ngay!
```

---

### Bước 3: Verify dataset sau khi xử lý

```bash
python verify_dataset.py
```

**Kiểm tra:**

- ✅ Chỉ còn 4 class IDs: 0, 1, 2, 3
- ✅ Class distribution hợp lý (balanced)
- ✅ Không có invalid class IDs
- ✅ File `data.yaml` đã được tạo

**Output mẫu:**

```
📊 TRAIN
-------------------------------------------------------------
📁 Files:
   - Images: 750
   - Labels: 750
   ✅ Images và labels khớp nhau!

🎯 Objects:
   - Tổng: 2005 objects

📦 Class Distribution:
   - Plastic    (ID 0):  480 ( 23.9%)
   - Metal      (ID 1):  411 ( 20.5%)
   - Paper      (ID 2):  603 ( 30.1%)
   - Glass      (ID 3):  511 ( 25.5%)

=============================================================
📝 Kiểm tra data.yaml
=============================================================
✅ Tìm thấy: dataset/data.yaml

Nội dung:
# Dataset configuration for YOLOv8
train: dataset/train/images
val: dataset/valid/images
test: dataset/test/images

nc: 4
names: ['Plastic', 'Metal', 'Paper', 'Glass']
```

---

## ⚠️ LƯU Ý QUAN TRỌNG

### 1. Backup tự động

Script sẽ **TỰ ĐỘNG TẠO BACKUP** trước khi xử lý:

```
dataset/          ← Bản xử lý (dùng để training)
dataset_backup/   ← Bản gốc (giữ để phòng hờ)
```

### 2. Class mapping

**QUAN TRỌNG:** Kiểm tra lại class order trong dataset gốc của bạn!

Trong script, tôi giả định:

```python
ORIGINAL_CLASSES = {
    0: 'cardboard',
    1: 'glass',
    2: 'metal',
    3: 'paper',
    4: 'plastic',
    5: 'random_trash'
}
```

**Nếu class order của bạn KHÁC, cần sửa file `process_dataset.py`:**

Mở file `process_dataset.py`, tìm dòng `ORIGINAL_CLASSES` và sửa cho đúng với dataset của bạn.

**Cách check class order:**

Xem file `data.yaml` gốc từ Roboflow hoặc chạy:

```python
# Đọc 1 vài label files và xem class IDs
with open('dataset/train/labels/img_001.txt', 'r') as f:
    for line in f:
        class_id = int(line.split()[0])
        print(f"Class ID: {class_id}")
```

### 3. Test trên một phần nhỏ trước

Nếu không chắc chắn, test trên một phần nhỏ:

```bash
# Copy một phần dataset ra test
cp -r dataset/train dataset_test_train
# Chỉ giữ 10 ảnh đầu
cd dataset_test_train/images
ls | tail -n +11 | xargs rm

# Test script
python process_dataset.py
# Nhập: dataset_test
```

---

## 🔧 Troubleshooting

### Lỗi: "Không tìm thấy folder"

```
❌ KHÔNG tìm thấy folder: dataset/train/images
```

**Giải pháp:**

- Kiểm tra lại đường dẫn
- Dataset cần có đúng cấu trúc: `dataset/train/images` và `dataset/train/labels`

### Lỗi: "Invalid class IDs"

```
❌ CẢNH BÁO: Tìm thấy class IDs không hợp lệ: [6, 7]
```

**Giải pháp:**

- Class order trong dataset không giống với script
- Sửa `ORIGINAL_CLASSES` trong `process_dataset.py`

### Lỗi: Permission denied

```
PermissionError: [Errno 13] Permission denied
```

**Giải pháp:**

- Đóng các chương trình đang mở dataset (VSCode, File Explorer)
- Chạy với quyền admin (Windows) hoặc sudo (Linux/Mac)

---

## 📊 Kết quả mong đợi

**Sau khi xử lý, dataset sẽ có:**

| Split | Images (ước tính) | Objects (ước tính) |
| ----- | ----------------- | ------------------ |
| Train | ~750              | ~2000              |
| Valid | ~200              | ~500               |
| Test  | ~100              | ~250               |

**Class distribution (balanced):**

- Plastic: ~24%
- Paper: ~30%
- Glass: ~25%
- Metal: ~21%

→ **RẤT TỐT cho training!**

---

## 🚀 Bước tiếp theo

Sau khi xử lý xong:

1. ✅ **Upload dataset lên Google Drive**

```bash
# Zip dataset
zip -r waste_dataset.zip dataset/

# Upload file zip lên Google Drive
# Hoặc kéo thả vào Drive web interface
```

2. ✅ **Chuyển sang Task 2: Setup Google Colab**

- Tạo Colab notebook mới
- Mount Google Drive
- Unzip dataset
- Cài đặt YOLOv8
- Bắt đầu training!

---

## 📞 Cần trợ giúp?

Nếu gặp vấn đề:

1. Chạy `verify_dataset.py` để xem thống kê
2. Check file backup: `dataset_backup/`
3. Đọc lại log output từ script
4. Hỏi AI assistant để được support!

---

**Good luck! 🎉**
