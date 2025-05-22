# utils/image_processing.py
#import cv2
#import numpy as np
import os # Though not directly used in process_image_local, good to have for utils related to paths if added later

#def process_image_local(img_path: str, target_size: int = 64,
#                        mean_rgb: list[float] = [0.485, 0.456, 0.406],
#                       std_rgb: list[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
#   """
#   对单张输入图像进行预处理，为模型提取特征做准备。
#
#   处理步骤:
#   4. 归一化到 [0, 1]。
#    5. 标准化 (减均值，除以标准差)。
#    6. HWC 转 CHW。
#    7. 增加批次维度 (batch_size=1)。

#    Args:
#        img_path (str): 输入图像的文件路径。
#        target_size (int, optional): 图像将被缩放到的目标正方形尺寸。默认为 64。
#        mean_rgb (list[float], optional): RGB三通道的均值。默认为 ImageNet 常用均值。
#        std_rgb (list[float], optional): RGB三通道的标准差。默认为 ImageNet 常用标准差。

#    Returns:
#        np.ndarray: 预处理后的图像数据 (1, 3, target_size, target_size)，float32类型。

#   FileNotFoundError: 如果图像文件无法读取。
#    """
#   img = cv2.imread(img_path)
#if img is None: raise FileNotFoundError(f"错误: 无法读取图像文件 {img_path}")

    #img = cv2.resize(img, (target_size, target_size))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = img.astype('float32') / 255.0
    #mean = np.array(mean_rgb, dtype='float32').reshape((1, 1, 3))
    #std = np.array(std_rgb, dtype='float32').reshape((1, 1, 3))
    #img_normalized = (img - mean) / std
    #img_chw = img_normalized.transpose((2, 0, 1))
    #img_expanded = np.expand_dims(img_chw, axis=0)
    #return img_expanded.astype('float32')

# 在文件头部新增：

from facenet_pytorch import MTCNN
import numpy as np
import cv2

# 在全局创建一个 MTCNN 检测器（默认输入 PIL 或 numpy 都行）
_mtcnn = MTCNN(keep_all=False, device='cpu')  # 如果有 GPU，可 device='cuda'

def process_image_local(img_path: str, target_size: int = 64,
                        mean_rgb: list[float] = [0.485, 0.456, 0.406],
                        std_rgb: list[float] = [0.229, 0.224, 0.225],
                        crop_margin: float = 0.2) -> np.ndarray:
    """
    1) 读原图
    2) 用 MTCNN 检测人脸框，选最大框
    3) 按 margin 扩大 bbox，裁剪
    4) resize → RGB → 归一化 → CHW → 扩 batch
    """
    # 1. 读取
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像 {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. 检测
    # MTCNN 返回 PIL.Image 裁剪后或 boxes.tensors
    boxes, _ = _mtcnn.detect(img_rgb)
    if boxes is None or len(boxes) == 0:
        raise RuntimeError("未检测到人脸")
    # 选最大框
    x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    h, w, _ = img_rgb.shape
    # 3. 扩 margin
    bw, bh = x2-x1, y2-y1
    mx = bw * crop_margin; my = bh * crop_margin
    x1c = max(int(x1 - mx), 0)
    y1c = max(int(y1 - my), 0)
    x2c = min(int(x2 + mx), w)
    y2c = min(int(y2 + my), h)
    face = img_rgb[y1c:y2c, x1c:x2c]

    # 4. resize, normalize
    face_resized = cv2.resize(face, (target_size, target_size))
    face_norm = face_resized.astype('float32') / 255.0
    mean = np.array(mean_rgb, dtype=np.float32).reshape(1,1,3)
    std  = np.array(std_rgb, dtype=np.float32).reshape(1,1,3)
    face_norm = (face_norm - mean) / std
    # HWC->CHW
    face_chw = face_norm.transpose(2,0,1)
    # add batch
    return np.expand_dims(face_chw, axis=0).astype('float32')

