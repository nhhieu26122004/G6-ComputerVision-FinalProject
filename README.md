# 🎓 Bài Tập Lớn: Phân loại Rác Thải Thông Minh - YOLOv8

## 📚 Tổng quan

Repository này chứa **kế hoạch chi tiết và template báo cáo** cho bài tập lớn môn Thị Giác Máy Tính với đề tài:

> **"Phân loại rác thải thông minh bằng thị giác máy tính trong hệ thống dây chuyền xử lý"**

**Công nghệ:** YOLOv8 (Object Detection)  
**4 Classes:** Plastic (Nhựa), Metal (Kim loại), Paper (Giấy), Glass (Thủy tinh)  
**Thời gian:** 3-4 ngày  
**Môi trường:** Google Colab (GPU T4)

---

## 📁 Cấu trúc Files

```
📦 g6-computervision-finalproject/
├── 📄 README.md                          ← File này (hướng dẫn tổng quan)
├── 📄 PLAN_BAI_TAP_LON_YOLOV8.md        ← KẾ HOẠCH CHI TIẾT (đọc đầu tiên!)
└── 📄 BAO_CAO_TEMPLATE.md               ← TEMPLATE BÁO CÁO (điền vào khi làm)
```

---

## 🚀 Hướng dẫn sử dụng

### Bước 1: Đọc kế hoạch chi tiết 📖

**File:** `PLAN_BAI_TAP_LON_YOLOV8.md`

File này chứa:

- ✅ **Roadmap 3-4 ngày** với timeline cụ thể
- ✅ **5 Tasks chi tiết:**
  - Task 1: Thu thập & chuẩn bị Dataset
  - Task 2: Setup môi trường & làm quen YOLOv8
  - Task 3: Training Model
  - Task 4: Testing & Evaluation
  - Task 5: Viết báo cáo & hoàn thiện
- ✅ **Code examples** đầy đủ cho từng task
- ✅ **Tips & tricks** để thành công
- ✅ **Troubleshooting** các vấn đề thường gặp

**👉 BẮT ĐẦU TỪ FILE NÀY!**

### Bước 2: Thực hiện theo từng task 💻

Làm theo trình tự trong kế hoạch:

**Ngày 1:** Task 1 + Task 2 (Dataset + Setup)  
**Ngày 2:** Task 3 (Training - chạy qua đêm nếu cần)  
**Ngày 3:** Task 4 (Testing & Optimization)  
**Ngày 4:** Task 5 (Viết báo cáo)

### Bước 3: Viết báo cáo 📝

**File:** `BAO_CAO_TEMPLATE.md`

Khi có kết quả từ training/testing:

1. Mở file `BAO_CAO_TEMPLATE.md`
2. Điền vào các phần:
   - `[XXX]` - Các con số từ kết quả của bạn
   - `[Mô tả]` - Phân tích và nhận xét của bạn
   - Chèn hình ảnh vào vị trí `*[Chèn hình: ...]*`
3. Export sang PDF để nộp

**Template bao gồm:**

- ✅ Cấu trúc báo cáo chuẩn học thuật (7 chương)
- ✅ Các phần lý thuyết đã viết sẵn (CNN, YOLO, YOLOv8...)
- ✅ Placeholder cho kết quả của bạn
- ✅ Source code trong Phụ lục

---

## 📊 Kết quả mong đợi

Sau khi hoàn thành, bạn sẽ có:

### 1. Code hoàn chỉnh 💻

```
code/
├── train.py          # Training script
├── test.py           # Testing script
├── inference.py      # Demo script
└── requirements.txt  # Dependencies
```

### 2. Trained Model 🤖

- `best.pt` - Model weights tốt nhất
- **Target:** mAP@0.5 ≥ 0.70
- **Speed:** Real-time (≥30 FPS)

### 3. Báo cáo chuyên nghiệp 📄

- **Format:** PDF, 10-15 trang
- **Nội dung:** Đầy đủ từ Giới thiệu đến Kết luận
- **Visualizations:** Loss curves, Confusion matrix, Predictions

### 4. Kết quả demo 🎬

- Sample predictions trên test images
- (Optional) Video demo

---

## 🎯 Quick Start (TL;DR)

Nếu muốn bắt đầu ngay:

### 1️⃣ Setup Google Colab

```python
# Tạo notebook mới, enable GPU T4
!pip install ultralytics -q
from ultralytics import YOLO
```

### 2️⃣ Download Dataset

- **Nguồn đề xuất:** https://universe.roboflow.com
- **Search:** "waste classification" hoặc "garbage detection"
- **Yêu cầu:** ≥500 images, YOLO format, 4 classes

### 3️⃣ Train Model

```python
model = YOLO('yolov8n.pt')
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

### 4️⃣ Test & Evaluate

```python
metrics = model.val(data='data.yaml', split='test')
print(f"mAP@0.5: {metrics.box.map50:.4f}")
```

### 5️⃣ Viết báo cáo

- Dùng template `BAO_CAO_TEMPLATE.md`
- Điền kết quả và phân tích

---

## 📚 Tài liệu tham khảo nhanh

### YOLOv8 Documentation

- **Official Docs:** https://docs.ultralytics.com
- **GitHub:** https://github.com/ultralytics/ultralytics
- **Tutorials:** https://blog.roboflow.com

### Dataset Sources

- **Roboflow Universe:** https://universe.roboflow.com
- **Kaggle Datasets:** https://www.kaggle.com/datasets
- **Waste Classification:** Search "waste detection", "trash classification"

### Papers to Cite

1. **YOLOv1:** Redmon et al. (2016) - "You Only Look Once"
2. **YOLOv8:** Ultralytics (2023) - GitHub repository
3. **COCO Dataset:** Lin et al. (2014) - Common Objects in Context

---

## ⚠️ Lưu ý quan trọng

### Tránh những sai lầm này:

❌ **KHÔNG:**

- Training với quá ít epochs (< 50)
- Quên backup weights (Colab disconnect = mất hết)
- Copy-paste lý thuyết không hiểu
- Nộp báo cáo không có phân tích kết quả
- Dùng dataset quá nhỏ (< 300 ảnh)

✅ **NÊN:**

- Backup weights vào Google Drive định kỳ
- Hiểu rõ từng bước đang làm gì
- Phân tích cụ thể, đưa ví dụ minh họa
- Test nhiều confidence thresholds
- Thừa nhận limitations và đề xuất improvements

### Tips thành công:

1. **Quản lý thời gian:** Tuân thủ timeline 3-4 ngày
2. **Dataset quality > quantity:** Annotations chính xác quan trọng hơn
3. **Monitor training:** Check loss curves, metrics thường xuyên
4. **Document as you go:** Đừng để cuối mới viết báo cáo
5. **Collaborate:** Chia task rõ ràng giữa các thành viên

---

## 🆘 Troubleshooting

### Vấn đề thường gặp:

**1. Colab Out of Memory (OOM)**

```python
# Giải pháp: Giảm batch size
batch=8  # hoặc 4
```

**2. Training quá chậm**

```python
# Giải pháp: Giảm image size
imgsz=416  # thay vì 640
```

**3. mAP quá thấp (< 0.50)**

- Kiểm tra annotations có đúng không
- Tăng epochs (100 → 150-200)
- Thử model lớn hơn (yolov8s)
- Tăng data augmentation

**4. Colab disconnect**

```python
# Giải pháp: Backup weights vào Drive
!cp runs/.../weights/best.pt /content/drive/MyDrive/backup.pt
```

Xem thêm chi tiết trong `PLAN_BAI_TAP_LON_YOLOV8.md` → Section "Troubleshooting"

---

## 📞 Liên hệ & Support

Nếu gặp khó khăn:

1. Đọc kỹ phần **Troubleshooting** trong file kế hoạch
2. Search lỗi trên Google/Stack Overflow
3. Tham khảo Ultralytics docs: https://docs.ultralytics.com
4. Hỏi giảng viên hoặc nhóm khác

---

## ✅ Checklist Trước Khi Nộp

### Code:

- [ ] `train.py` chạy được
- [ ] `test.py` cho kết quả đúng
- [ ] `inference.py` demo được
- [ ] `requirements.txt` đầy đủ
- [ ] README.md có hướng dẫn chạy

### Model:

- [ ] `best.pt` được lưu
- [ ] mAP@0.5 ≥ 0.70 (hoặc gần đạt)
- [ ] Training converged (loss giảm)

### Báo cáo:

- [ ] Đầy đủ 7 chương
- [ ] Có hình/bảng minh họa (≥10 figures)
- [ ] Có số liệu kết quả thật (không fake)
- [ ] Có phân tích cụ thể (không chỉ liệt kê)
- [ ] Trích dẫn ≥5 tài liệu tham khảo
- [ ] Không lỗi chính tả
- [ ] Format PDF chuẩn

### Demo (Optional):

- [ ] Sample predictions chất lượng cao
- [ ] Video demo (1-2 phút)

---

## 🎉 Lời kết

Chúc bạn thực hiện project thành công! 🚀

Với kế hoạch chi tiết và template báo cáo này, bạn hoàn toàn có thể hoàn thành xuất sắc bài tập lớn trong 3-4 ngày.

**Remember:**

- 📖 Đọc kỹ `PLAN_BAI_TAP_LON_YOLOV8.md` trước khi bắt đầu
- 💻 Follow từng task một cách tuần tự
- 📝 Dùng `BAO_CAO_TEMPLATE.md` để viết báo cáo
- 💾 Backup thường xuyên!

Good luck! 💪

---

**Last updated:** [15/10/2025]  
**Version:** 1.0  
**Môn:** Thị Giác Máy Tính  
**Học kỳ:** 1-1-25  
**Thành viên:**

- Nguyễn Huy Hiếu - 22010160
- Vũ Tuấn Anh -
