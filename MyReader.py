# coding:utf-8
import os
import cv2
import random
import numpy as np
import paddle
from paddle.io import Dataset, DataLoader
import paddle.vision.transforms as T

class ImageAugmentation:
    """图像数据增强和预处理类。

    该类封装了多种常见的图像增强技术，如随机翻转、颜色抖动、亮度调整等，
    以及图像的缩放和归一化操作。
    它区分训练模式和评估/测试模式，在训练时应用数据增强以提高模型泛化能力，
    在评估/测试时仅进行必要的尺寸调整和归一化。
    """

    def __init__(self, image_size: int | tuple[int, int] = 64,
                 mean_value: list[float] = [0.5, 0.5, 0.5],
                 std_value: list[float] = [0.5, 0.5, 0.5],
                 is_train: bool = True):
        """
        ImageAugmentation 初始化函数。

        Args:
            image_size (int or tuple, optional): 目标图像尺寸。如果是整数，则假设为正方形 (H=W=image_size)。
                                               如果是元组 (H, W)，则分别指定高度和宽度。
                                               默认为 64。
            mean_value (list[float], optional): 图像归一化时使用的均值 (按通道，通常为RGB顺序)。
                                             长度应为3。默认为 [0.5, 0.5, 0.5]。
            std_value (list[float], optional): 图像归一化时使用的标准差 (按通道，通常为RGB顺序)。
                                             长度应为3。默认为 [0.5, 0.5, 0.5]。
            is_train (bool, optional): 是否处于训练模式。默认为 True。
        """
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.mean_value = mean_value
        self.std_value = std_value
        self.is_train = is_train

        # Initialize transforms to None; they will be created in process_image
        # within the worker process scope on the first call for that worker.
        self.train_transforms = None
        self.eval_transforms = None
        self.color_jitter = None # Also initialize individual transforms to None
        self.random_flip = None


    def process_image(self, image_path):
        """
        读取、处理并增强图像。Transforms are initialized on the first call in the worker process.

        Args:
            image_path (str): 图像文件的完整路径或相对路径。

        Returns:
            numpy.ndarray: 处理后的图像数据，格式为 CHW (Channels, Height, Width)。
        """
        # Initialize transforms on the first call within the worker process scope
        # This ensures T is available when transforms are created and used in the worker.
        if self.is_train and self.train_transforms is None:
            try:
                self.color_jitter = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
                self.random_flip = T.RandomHorizontalFlip(prob=0.5)
                self.train_transforms = T.Compose([
                    # Apply ColorJitter and RandomFlip BEFORE converting to float
                    self.color_jitter,
                    self.random_flip,
                    T.Resize(self.image_size),
                    # Normalization and transpose will be applied manually AFTER transforms
                ])
            except ImportError:
                 print("Error: paddle.vision.transforms not available when initializing train transforms!")
                 raise # Re-raise the error if import fails

        elif not self.is_train and self.eval_transforms is None:
            try:
                self.eval_transforms = T.Compose([
                    T.Resize(self.image_size),
                    # Normalization and transpose will be applied manually AFTER transforms
                ])
            except ImportError:
                 print("Error: paddle.vision.transforms not available when initializing eval transforms!")
                 raise # Re-raise the error if import fails


        full_img_path = image_path

        try:
            # Check if file exists
            if not os.path.exists(full_img_path):
                raise FileNotFoundError(f"图像文件未找到: {full_img_path}")

            # Use OpenCV to read image in color mode (tries to convert grayscale to 3 channels)
            img = cv2.imread(full_img_path, cv2.IMREAD_COLOR)

            # Check if image read was successful
            if img is None:
                raise IOError(f"无法读取图像文件: {full_img_path}")

            # Convert image from BGR (OpenCV default) to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # img is now HWC, RGB, uint8 (0-255)

            # --- Handle grayscale images (already in RGB now) ---
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
            elif len(img.shape) == 3 and img.shape[2] == 1:
                 img = np.stack([np.squeeze(img, axis=-1)] * 3, axis=-1)
            # --- Added block end ---

            # Apply transforms (ColorJitter, RandomFlip, Resize) on uint8 image
            if self.is_train:
                transformed_img = self.train_transforms(img) # Apply transforms on uint8 img
            else:
                transformed_img = self.eval_transforms(img)   # Apply transforms on uint8 img

            # Convert image to float and normalize to [0, 1] after transforms that expect uint8
            # Ensure transformed_img is numpy array and convert to float [0, 1]
            transformed_img_float = transformed_img.astype(np.float32) / 255.0 # Now HWC, RGB, float [0,1]


            # Normalize and Transpose CHW
            # Normalize after converting to float and other transforms
            mean = np.array(self.mean_value).reshape((1, 1, 3)) # Reshape for HWC
            std = np.array(self.std_value).reshape((1, 1, 3))   # Reshape for HWC
            img_normalized = (transformed_img_float - mean) / std # Apply normalization

            # HWC to CHW
            img_chw = img_normalized.transpose((2, 0, 1))

            # Convert to Paddle Tensor if needed by the rest of the pipeline.
            # The DataLoader should handle batching and conversion to Tensor.
            return img_chw

        except Exception as e:
            # Capture and re-raise the exception with more context
            raise Exception(f"图像处理失败: {image_path}, 原因: {str(e)}")

# Assuming MyDataset class is defined below this or in the same file
class MyDataset(Dataset):
    """
    自定义数据集类，用于加载图像数据和对应的标签。
    """
    def __init__(self, data_list_file: str, image_augmentor: ImageAugmentation, data_root_path: str = ''):
        """
        初始化数据集。

        Args:
            data_list_file (str): 包含图像路径和标签的数据列表文件路径。
                                  每行格式应为: 图像相对路径\t类别ID。
            image_augmentor (ImageAugmentation): 图像增强和预处理类的实例。
            data_root_path (str, optional): 数据集根目录路径。如果 data_list_file
                                            中的路径是相对路径，则需要提供此根路径。
                                            默认为空字符串，表示 data_list_file 中的路径是绝对路径
                                            或相对于当前工作目录。
        """
        super(MyDataset, self).__init__()
        self.data_list_file = data_list_file
        self.image_augmentor = image_augmentor
        self.data_root_path = data_root_path
        self.data_list = self._load_data_list()

    def _load_data_list(self):
        """加载数据列表文件内容。"""
        data = []
        try:
            with open(self.data_list_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('\t')
                    if len(parts) == 2:
                        image_relative_path, label = parts
                        # Construct full image path
                        full_image_path = os.path.join(self.data_root_path, image_relative_path)
                        try:
                            label = int(label)
                            data.append((full_image_path, label))
                        except ValueError:
                            print(f"警告: 跳过无效标签行: {line}")
                    else:
                        print(f"警告: 跳过格式不正确的行: {line}")
            if not data:
                 print(f"警告: 数据列表文件 {self.data_list_file} 中没有找到有效数据。")
        except FileNotFoundError:
            print(f"错误: 数据列表文件未找到: {self.data_list_file}")
            # Depending on how critical this is, you might want to exit or raise an error
            raise # Re-raise the error so train.py knows the data list is missing
        except Exception as e:
            print(f"错误: 读取数据列表文件 {self.data_list_file} 时发生异常: {e}")
            raise # Re-raise other exceptions during file reading

        mode_str = 'train' if self.image_augmentor.is_train else ('eval' if 'test' in self.data_list_file.lower() else 'unknown')
        print(f"成功从 '{self.data_list_file}' 加载 {len(data)} 个样本用于模式 '{mode_str}'。")
        return data


    def __len__(self):
        """返回数据集中的样本总数。"""
        return len(self.data_list)

    def __getitem__(self, index):
        """根据索引获取一个样本。"""
        full_img_path, label = self.data_list[index]

        # Process image using the ImageAugmentation instance
        # The process_image method will handle reading, augmentation, resizing, normalization, and CHW conversion
        processed_image = self.image_augmentor.process_image(full_img_path)

        # Convert label to Paddle Tensor
        label_tensor = paddle.to_tensor(label, dtype='int64')

        return processed_image, label_tensor

# Assuming create_data_loader function is defined below this or in the same file
# This function is likely called from train.py

# Example of how create_data_loader might be implemented (based on train.py's call):
def create_data_loader(config, mode='train'):
    """
    根据配置和模式创建数据加载器。

    Args:
        config: 配置对象 (由 config_utils.py 加载)。
        mode (str): 数据加载器模式 ('train', 'eval', 'acceptance').

    Returns:
        paddle.io.DataLoader: 创建的数据加载器。
    """
    print(f"正在创建模式 '{mode}' 的数据加载器...")

    # Determine data list file path and data root path
    # Based on train.py config (data_dir: data, class_name: face) and previous outputs,
    # the list files are expected in data/face/
    dataset_list_file_dir = os.path.join(config.data_dir, config.class_name)
    data_list_file_name = config.dataset_params[f'{mode}_list'] # e.g., 'trainer.list'
    actual_data_list_file_path = os.path.join(dataset_list_file_dir, data_list_file_name)

    # The data_root_path for the dataset should be the directory that, when joined with
    # relative paths from the list file (e.g., 'dilireba/...'), gives the full image path.
    # Since CreateDataList.py with --data_root data/face --output_dir data/face put lists
    # in data/face and paths like 'dilireba/...', the data root for the dataset is 'data/face'.
    # This corresponds to dataset_list_file_dir.
    data_root_for_dataset = dataset_list_file_dir

    if not os.path.exists(actual_data_list_file_path):
        # This error should be less likely now if CreateDataList.py was run correctly
        raise FileNotFoundError(f"模式 '{mode}' 的数据列表文件 '{actual_data_list_file_path}' 未找到。请确保它存在于 '{dataset_list_file_dir}' 目录下，或先运行 CreateDataList.py 生成。")


    image_augmentor = ImageAugmentation(
        image_size=config.image_size,
        mean_value=config.dataset_params.mean,
        std_value=config.dataset_params.std,
        is_train=(mode == 'train')
    )

    dataset = MyDataset(
        data_list_file=actual_data_list_file_path,
        image_augmentor=image_augmentor,
        data_root_path=data_root_for_dataset # Pass the data root to the dataset
    )

    # Determine batch size and shuffle
    batch_size = config.batch_size if mode == 'train' else config.batch_size # Using same batch size for eval/infer, adjust if needed
    shuffle = (mode == 'train') # Shuffle only for training data
    drop_last = (mode == 'train') # Drop the last incomplete batch only during training

    # Create DataLoader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.dataset_params.num_workers,
        drop_last=drop_last,
        # use_shared_memory=True # Potentially improve performance, might interact with multiprocessing
    )

    print(f"为模式 '{mode}' 创建 DataLoader 成功。Batch size: {batch_size}, Shuffle: {shuffle}, Num workers: {config.dataset_params.num_workers}, Drop last: {drop_last}")

    return data_loader


# Add test functions here if needed, like the ones in the original snippet
# if __name__ == '__main__':
#     # Add test logic here
#     pass