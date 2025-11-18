import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import cv2
import numpy as np
import albumentations as A
from typing import List, Tuple

# -------------------------------
# 配置参数
# -------------------------------
IMG_DIR = 'E:\\PytorchTest\\PytorchTest\\dataset\\images'          # 原始图像目录
LABEL_DIR = 'E:\\PytorchTest\\PytorchTest\\dataset\\labels'         # 原始标签目录（YOLOv8格式）
AUG_IMG_DIR = 'E:\\PytorchTest\\PytorchTest\\dataset\\aug_images'   # 增强后图像保存目录
AUG_LABEL_DIR = 'E:\\PytorchTest\\PytorchTest\\dataset\\aug_labels' # 增强后标签保存目录
IMG_SIZE = 640                         # YOLOv8 默认输入尺寸
IMG_HEIGHT = 3648
IMG_WIDTH = 5472
NUM_AUG_SAMPLES = 20                  # 每张图像生成的增强样本数

# -------------------------------
# 数据增强变换（适用于目标检测）
# -------------------------------
def get_detection_transforms():
    return A.Compose([
        A.Resize(IMG_HEIGHT, IMG_WIDTH),                           # 统一分辨率
        A.HorizontalFlip(p=0.5),                                   # 水平翻转
        A.RandomBrightnessContrast(p=0.2),                        # 亮度对比度扰动
        A.HueSaturationValue(p=0.2),                              # 色彩抖动
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),                                                           # 平移、缩放、旋转
        A.RandomResizedCrop(
            IMG_HEIGHT,
            IMG_WIDTH,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=0.5
        ),                                                           # 随机裁剪
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            fill_value=0,
            p=0.2
        ),                                                           # 随机遮挡
    ], bbox_params=A.BboxParams(
        format='yolo',           # 输入边界框格式（归一化到0-1）
        label_fields=['class_ids']
    ))

# -------------------------------
# 读取 YOLOv8 格式标注
# -------------------------------
def read_yolo_labels(label_path: str) -> Tuple[List[float], List[List[float]]]:
    """返回类别列表和边界框列表（归一化）"""
    if not os.path.exists(label_path):
        return [], []
    with open(label_path, 'r') as f:
        lines = f.readlines()
    class_ids = []
    bboxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_ids.append(int(parts[0]))
        bboxes.append([float(x) for x in parts[1:]])
    return class_ids, bboxes

# -------------------------------
# 保存 YOLOv8 格式标注
# -------------------------------
def write_yolo_labels(label_path: str, class_ids: List[int], bboxes: List[List[float]]):
    """保存类别和边界框到文件"""
    with open(label_path, 'w') as f:
        for cls_id, bbox in zip(class_ids, bboxes):
            f.write(f"{cls_id} {' '.join(map(str, bbox))}\n")

# -------------------------------
# 批量增强数据集
# -------------------------------
def augment_dataset():
    os.makedirs(AUG_IMG_DIR, exist_ok=True)
    os.makedirs(AUG_LABEL_DIR, exist_ok=True)

    transform = get_detection_transforms()
    image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in image_files:
        img_path = os.path.join(IMG_DIR, img_file)
        label_path = os.path.join(LABEL_DIR, img_file.rsplit('.', 1)[0] + '.txt')
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        class_ids, bboxes = read_yolo_labels(label_path)

        for i in range(NUM_AUG_SAMPLES):
            # 应用增强
            augmented = transform(image=image, bboxes=bboxes, class_ids=class_ids)
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_class_ids = augmented['class_ids']

            # 保存增强后的图像和标注
            base_name = img_file.rsplit('.', 1)[0]
            aug_img_name = f"{base_name}_aug_{i}.jpg"
            aug_label_name = f"{base_name}_aug_{i}.txt"
            cv2.imwrite(os.path.join(AUG_IMG_DIR, aug_img_name), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            write_yolo_labels(os.path.join(AUG_LABEL_DIR, aug_label_name), aug_class_ids, aug_bboxes)

            print(f"Saved: {aug_img_name}, {aug_label_name}")

# -------------------------------
# 运行增强
# -------------------------------
if __name__ == "__main__":
    augment_dataset()