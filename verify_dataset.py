"""
Script để verify dataset sau khi xử lý
Kiểm tra class distribution, số lượng ảnh, labels
"""

import os
import glob
from collections import defaultdict


def verify_dataset(dataset_root):
    """Verify dataset structure và thống kê"""

    print("="*60)
    print("🔍 VERIFY DATASET")
    print("="*60)

    class_names = ['Plastic', 'Metal', 'Paper', 'Glass']

    for split in ['train', 'valid', 'test']:
        print(f"\n📊 {split.upper()}")
        print("-"*60)

        image_dir = f"{dataset_root}/{split}/images"
        label_dir = f"{dataset_root}/{split}/labels"

        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            print(f"⚠️  Không tìm thấy: {split}/")
            continue

        # Đếm files
        image_files = glob.glob(f"{image_dir}/*")
        label_files = glob.glob(f"{label_dir}/*.txt")

        print(f"📁 Files:")
        print(f"   - Images: {len(image_files)}")
        print(f"   - Labels: {len(label_files)}")

        # Kiểm tra mismatch
        image_stems = set([os.path.splitext(os.path.basename(f))[0]
                          for f in image_files])
        label_stems = set([os.path.splitext(os.path.basename(f))[0]
                          for f in label_files])

        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems

        if missing_labels:
            print(f"   ⚠️  Ảnh thiếu labels: {len(missing_labels)}")
        if missing_images:
            print(f"   ⚠️  Labels thiếu ảnh: {len(missing_images)}")

        if not missing_labels and not missing_images:
            print(f"   ✅ Images và labels khớp nhau!")

        # Thống kê classes
        class_counts = defaultdict(int)
        total_objects = 0
        empty_files = 0

        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()

                if not lines:
                    empty_files += 1
                    continue

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                        total_objects += 1

        print(f"\n🎯 Objects:")
        print(f"   - Tổng: {total_objects} objects")

        if empty_files > 0:
            print(f"   ⚠️  Empty label files: {empty_files}")

        print(f"\n📦 Class Distribution:")
        for class_id in range(4):
            count = class_counts[class_id]
            percentage = count / total_objects * 100 if total_objects > 0 else 0
            class_name = class_names[class_id]
            print(
                f"   - {class_name:10s} (ID {class_id}): {count:4d} ({percentage:5.1f}%)")

        # Kiểm tra class IDs invalid
        invalid_classes = [cid for cid in class_counts.keys() if cid not in [
            0, 1, 2, 3]]
        if invalid_classes:
            print(
                f"\n   ❌ CẢNH BÁO: Tìm thấy class IDs không hợp lệ: {invalid_classes}")
            print(f"      Chỉ nên có class IDs: 0, 1, 2, 3")

    # Kiểm tra data.yaml
    print(f"\n{'='*60}")
    print("📝 Kiểm tra data.yaml")
    print("="*60)

    yaml_path = f"{dataset_root}/data.yaml"
    if os.path.exists(yaml_path):
        print(f"✅ Tìm thấy: {yaml_path}")
        with open(yaml_path, 'r') as f:
            print("\nNội dung:")
            print(f.read())
    else:
        print(f"❌ KHÔNG tìm thấy: {yaml_path}")
        print("   Cần tạo file data.yaml!")

    print("\n" + "="*60)
    print("✅ VERIFY HOÀN TẤT")
    print("="*60)


if __name__ == "__main__":
    dataset_root = input("📂 Nhập đường dẫn tới dataset: ").strip()

    if not os.path.exists(dataset_root):
        print(f"❌ Không tìm thấy: {dataset_root}")
    else:
        verify_dataset(dataset_root)
