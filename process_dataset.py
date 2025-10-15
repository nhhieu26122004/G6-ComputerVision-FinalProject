"""
Script để xử lý dataset: Xóa Cardboard và Random Trash, chỉ giữ 4 classes
Author: AI Assistant
Date: 2025-10-15
"""

import os
import glob
import shutil
from pathlib import Path

# ============================================
# CONFIGURATION
# ============================================

# Mapping class names sang IDs (theo dataset gốc)
# CHECKED: Class order từ file data.yaml
ORIGINAL_CLASSES = {
    0: 'Metal',          # GIỮ -> remap to 1
    1: 'Paper',          # GIỮ -> remap to 2
    2: 'Plastic',        # GIỮ -> remap to 0
    3: 'Random Trash',   # SẼ XÓA
    4: 'cardboard',      # SẼ XÓA
    5: 'glass',          # GIỮ -> remap to 3
}

# Classes muốn giữ lại (4 classes)
# Remap theo thứ tự mới: Plastic(0), Metal(1), Paper(2), Glass(3)
KEEP_CLASSES = {
    2: 0,  # Plastic (old ID 2) -> class 0
    0: 1,  # Metal (old ID 0) -> class 1
    1: 2,  # Paper (old ID 1) -> class 2
    5: 3,  # glass (old ID 5) -> class 3
}

NEW_CLASS_NAMES = ['Plastic', 'Metal', 'Paper', 'Glass']

# ============================================
# FUNCTIONS
# ============================================


def process_labels(label_dir, image_dir, dry_run=True):
    """
    Xử lý labels: remap class IDs và xóa các ảnh không có annotations hợp lệ

    Args:
        label_dir: Đường dẫn tới folder labels
        image_dir: Đường dẫn tới folder images
        dry_run: Nếu True, chỉ in ra thống kê mà không thực sự xóa/sửa
    """
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))

    stats = {
        'total_files': len(label_files),
        'kept_files': 0,
        'removed_files': 0,
        'total_objects_before': 0,
        'total_objects_after': 0,
        'class_counts_before': {i: 0 for i in range(6)},
        'class_counts_after': {i: 0 for i in range(4)}
    }

    for label_file in label_files:
        new_lines = []
        file_has_valid_objects = False

        # Đọc file label gốc
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:  # Invalid format
                    continue

                old_class = int(parts[0])
                stats['total_objects_before'] += 1
                stats['class_counts_before'][old_class] += 1

                # Kiểm tra xem class này có được giữ không
                if old_class in KEEP_CLASSES:
                    new_class = KEEP_CLASSES[old_class]
                    new_line = f"{new_class} {' '.join(parts[1:])}\n"
                    new_lines.append(new_line)
                    file_has_valid_objects = True
                    stats['total_objects_after'] += 1
                    stats['class_counts_after'][new_class] += 1

        # Xử lý file
        if file_has_valid_objects:
            # Giữ file, cập nhật với class IDs mới
            stats['kept_files'] += 1

            if not dry_run:
                with open(label_file, 'w') as f:
                    f.writelines(new_lines)
        else:
            # Xóa file label và ảnh tương ứng
            stats['removed_files'] += 1

            if not dry_run:
                # Xóa label file
                os.remove(label_file)

                # Tìm và xóa ảnh tương ứng
                label_name = Path(label_file).stem
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    img_file = os.path.join(image_dir, label_name + ext)
                    if os.path.exists(img_file):
                        os.remove(img_file)
                        print(f"  Xóa: {img_file}")
                        break

    return stats


def print_stats(stats, split_name):
    """In thống kê"""
    print(f"\n{'='*60}")
    print(f"📊 THỐNG KÊ: {split_name.upper()}")
    print(f"{'='*60}")

    print(f"\n📁 Files:")
    print(f"  - Tổng files: {stats['total_files']}")
    print(
        f"  - Giữ lại: {stats['kept_files']} ({stats['kept_files']/stats['total_files']*100:.1f}%)")
    print(
        f"  - Xóa: {stats['removed_files']} ({stats['removed_files']/stats['total_files']*100:.1f}%)")

    print(f"\n🎯 Objects:")
    print(f"  - Trước: {stats['total_objects_before']} objects")
    print(f"  - Sau: {stats['total_objects_after']} objects")
    print(
        f"  - Giảm: {stats['total_objects_before'] - stats['total_objects_after']} objects")

    print(f"\n📦 Class distribution TRƯỚC XỬ LÝ:")
    for class_id, count in stats['class_counts_before'].items():
        class_name = ORIGINAL_CLASSES.get(class_id, 'unknown')
        percentage = count / stats['total_objects_before'] * \
            100 if stats['total_objects_before'] > 0 else 0
        print(
            f"  - {class_name:15s} (ID {class_id}): {count:4d} objects ({percentage:5.1f}%)")

    print(f"\n✅ Class distribution SAU XỬ LÝ (4 classes):")
    for class_id, count in stats['class_counts_after'].items():
        class_name = NEW_CLASS_NAMES[class_id]
        percentage = count / stats['total_objects_after'] * \
            100 if stats['total_objects_after'] > 0 else 0
        print(
            f"  - {class_name:15s} (ID {class_id}): {count:4d} objects ({percentage:5.1f}%)")


def create_data_yaml(dataset_root, output_file='data.yaml'):
    """Tạo file data.yaml mới"""
    content = f"""# Dataset configuration for YOLOv8
# Generated by process_dataset.py

# Paths (relative to this file)
train: {dataset_root}/train/images
val: {dataset_root}/valid/images
test: {dataset_root}/test/images

# Number of classes
nc: 4

# Class names
names: ['Plastic', 'Metal', 'Paper', 'Glass']

# Class descriptions
# - Plastic: Plastic bottles, bags, containers
# - Metal: Aluminum cans, metal containers
# - Paper: Paper, documents (cardboard merged into this)
# - Glass: Glass bottles, containers
"""

    with open(output_file, 'w') as f:
        f.write(content)

    print(f"\n✅ Đã tạo file: {output_file}")

# ============================================
# MAIN EXECUTION
# ============================================


def main():
    print("="*60)
    print("🗑️  XÓA CARDBOARD VÀ RANDOM TRASH - CHỈ GIỮ 4 CLASSES")
    print("="*60)

    # Nhập đường dẫn dataset
    dataset_root = input(
        "\n📂 Nhập đường dẫn tới dataset folder (ví dụ: ./dataset hoặc dataset): ").strip()

    if not os.path.exists(dataset_root):
        print(f"❌ KHÔNG tìm thấy folder: {dataset_root}")
        return

    # Kiểm tra cấu trúc
    required_dirs = [
        f"{dataset_root}/train/images",
        f"{dataset_root}/train/labels",
        f"{dataset_root}/valid/images",
        f"{dataset_root}/valid/labels",
    ]

    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ KHÔNG tìm thấy folder: {dir_path}")
            print(
                "   Dataset cần có cấu trúc: dataset/train/images, dataset/train/labels, ...")
            return

    print(f"\n✅ Dataset folder OK: {dataset_root}")

    # Xác nhận class mapping
    print("\n📋 CLASS MAPPING:")
    print("   XÓA:")
    print("   - Random Trash (ID 3)")
    print("   - cardboard (ID 4)")
    print("\n   GIỮ LẠI và REMAP:")
    for old_id, new_id in sorted(KEEP_CLASSES.items()):
        old_name = ORIGINAL_CLASSES[old_id]
        new_name = NEW_CLASS_NAMES[new_id]
        print(
            f"   - {old_name:15s} (ID {old_id}) -> {new_name:15s} (ID {new_id})")

    # DRY RUN đầu tiên
    print("\n" + "="*60)
    print("🔍 BƯỚC 1: DRY RUN (Xem trước, không thực sự xóa/sửa)")
    print("="*60)

    total_stats = {
        'train': None,
        'valid': None,
        'test': None
    }

    for split in ['train', 'valid', 'test']:
        label_dir = f"{dataset_root}/{split}/labels"
        image_dir = f"{dataset_root}/{split}/images"

        if os.path.exists(label_dir):
            stats = process_labels(label_dir, image_dir, dry_run=True)
            print_stats(stats, split)
            total_stats[split] = stats

    # Tổng kết
    total_before = sum(s['total_objects_before']
                       for s in total_stats.values() if s)
    total_after = sum(s['total_objects_after']
                      for s in total_stats.values() if s)

    print("\n" + "="*60)
    print("📊 TỔNG KẾT")
    print("="*60)
    print(f"Tổng objects TRƯỚC: {total_before}")
    print(f"Tổng objects SAU: {total_after}")
    print(
        f"Sẽ XÓA: {total_before - total_after} objects ({(total_before - total_after)/total_before*100:.1f}%)")

    # Xác nhận thực hiện
    print("\n" + "="*60)
    confirm = input(
        "\n⚠️  Bạn có chắc muốn THỰC HIỆN xóa/sửa files? (yes/no): ").strip().lower()

    if confirm != 'yes':
        print("\n❌ Đã hủy. Không có file nào bị thay đổi.")
        return

    # Backup trước khi xử lý
    print("\n💾 Đang tạo backup...")
    backup_dir = f"{dataset_root}_backup"
    if not os.path.exists(backup_dir):
        shutil.copytree(dataset_root, backup_dir)
        print(f"✅ Đã backup vào: {backup_dir}")
    else:
        print(f"⚠️  Backup đã tồn tại: {backup_dir}")

    # THỰC HIỆN
    print("\n" + "="*60)
    print("🔧 BƯỚC 2: THỰC HIỆN XỬ LÝ")
    print("="*60)

    for split in ['train', 'valid', 'test']:
        label_dir = f"{dataset_root}/{split}/labels"
        image_dir = f"{dataset_root}/{split}/images"

        if os.path.exists(label_dir):
            print(f"\n⚙️  Đang xử lý: {split}...")
            stats = process_labels(label_dir, image_dir, dry_run=False)
            print(f"✅ Hoàn thành: {split}")

    # Tạo data.yaml mới
    print("\n" + "="*60)
    print("📝 Tạo file data.yaml")
    print("="*60)
    create_data_yaml(dataset_root, f"{dataset_root}/data.yaml")

    # Hoàn thành
    print("\n" + "="*60)
    print("✅ HOÀN THÀNH!")
    print("="*60)
    print(f"\n📂 Dataset đã được xử lý: {dataset_root}")
    print(f"💾 Backup gốc tại: {backup_dir}")
    print(f"📝 File config: {dataset_root}/data.yaml")
    print("\n🚀 Bạn có thể bắt đầu training ngay!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Đã hủy bởi người dùng.")
    except Exception as e:
        print(f"\n\n❌ LỖI: {e}")
        import traceback
        traceback.print_exc()
