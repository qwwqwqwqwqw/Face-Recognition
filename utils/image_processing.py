# utils/image_processing.py
import cv2
import numpy as np
import os # Though not directly used in process_image_local, good to have for utils related to paths if added later

def process_image_local(img_path: str, target_size: int = 64, 
                        mean_rgb: list[float] = [0.485, 0.456, 0.406], 
                        std_rgb: list[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    对单张输入图像进行预处理，为模型提取特征做准备。

    处理步骤:
    1. 加载图像。
    2. 缩放到 `target_size`。
    3. BGR 转 RGB。
    4. 归一化到 [0, 1]。
    5. 标准化 (减均值，除以标准差)。
    6. HWC 转 CHW。
    7. 增加批次维度 (batch_size=1)。

    Args:
        img_path (str): 输入图像的文件路径。
        target_size (int, optional): 图像将被缩放到的目标正方形尺寸。默认为 64。
        mean_rgb (list[float], optional): RGB三通道的均值。默认为 ImageNet 常用均值。
        std_rgb (list[float], optional): RGB三通道的标准差。默认为 ImageNet 常用标准差。

    Returns:
        np.ndarray: 预处理后的图像数据 (1, 3, target_size, target_size)，float32类型。

    Raises:
        FileNotFoundError: 如果图像文件无法读取。
    """
    img = cv2.imread(img_path)
    if img is None: raise FileNotFoundError(f"错误: 无法读取图像文件 {img_path}")
    img = cv2.resize(img, (target_size, target_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    mean = np.array(mean_rgb, dtype='float32').reshape((1, 1, 3))
    std = np.array(std_rgb, dtype='float32').reshape((1, 1, 3))
    img_normalized = (img - mean) / std
    img_chw = img_normalized.transpose((2, 0, 1))
    img_expanded = np.expand_dims(img_chw, axis=0)
    return img_expanded.astype('float32') 