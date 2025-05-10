# coding:utf-8
import os
import cv2
import random
import numpy as np
import paddle
from paddle.io import Dataset, DataLoader

class FaceDataset(Dataset):
    """人脸数据集类，继承自paddle.io.Dataset"""
    def __init__(self, data_list_file, image_size=64, mode='train'):
        """
        FaceDataset 初始化函数
        Args:
            data_list_file (str): 数据列表文件的路径。文件每行格式应为：图像路径\t标签ID
            image_size (int): 图像预处理后的目标尺寸 (高度和宽度相同)
            mode (str): 数据集模式，'train' 或 'test'。'train'模式下会进行数据增强。
        """
        super(FaceDataset, self).__init__()
        self.image_paths_and_labels = [] # 用于存储 (图像路径, 标签) 对
        self.image_size = image_size
        self.mode = mode.lower() # 转换为小写以统一处理
        
        # 检查并读取数据列表文件
        if not os.path.exists(data_list_file):
            raise FileNotFoundError(f"错误: 找不到数据列表文件 -> {data_list_file}")
        
        with open(data_list_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split() # 使用空格或制表符分割
                if len(parts) == 2:
                    image_path = parts[0]
                    try:
                        label = int(parts[1])
                        self.image_paths_and_labels.append((image_path, label))
                    except ValueError:
                        print(f"警告: 无效的标签格式 '{parts[1]}' 在文件 {data_list_file} 的行: '{line.strip()}'。已跳过此行。")
                elif line.strip(): # 如果行不为空但格式不正确
                    print(f"警告: 无效的数据格式在文件 {data_list_file} 的行: '{line.strip()}'。预期格式: '图像路径 标签ID'。已跳过此行。")
    
    def __getitem__(self, idx):
        """
        根据索引获取单个样本 (图像和标签)
        Args:
            idx (int): 样本的索引
        Returns:
            tuple: (image_tensor, label_id) 处理后的图像张量和对应的标签ID
        """
        image_path, label = self.image_paths_and_labels[idx]
        
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                # 如果图像读取失败，打印错误并返回一个随机数据以避免训练中断
                print(f"警告: 无法读取图像 {image_path}。将使用随机数据替代。")
                # 生成一个符合预期的随机图像张量和原始标签
                random_img_tensor = np.random.rand(3, self.image_size, self.image_size).astype('float32')
                return random_img_tensor, label
            
            # 图像预处理
            img_resized = cv2.resize(img, (self.image_size, self.image_size))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) # OpenCV默认BGR，转换为RGB
            
            # 数据增强 (仅在训练模式下进行)
            if self.mode == 'train':
                # 随机水平翻转
                if random.random() > 0.5:
                    img_rgb = cv2.flip(img_rgb, 1) # 1表示水平翻转
                
                # 随机调整亮度和对比度
                # 此处可根据需要添加更多数据增强方法，如旋转、裁剪、色彩抖动等
                if random.random() > 0.5:
                    # alpha 控制对比度 (1.0为原始，<1降低, >1增加)
                    # beta 控制亮度 (0为原始，<0降低, >0增加)
                    alpha = 0.8 + random.random() * 0.4  # 范围 [0.8, 1.2]
                    beta = 20 * (random.random() - 0.5)    # 范围 [-10, 10]
                    img_rgb = cv2.convertScaleAbs(img_rgb, alpha=alpha, beta=beta)
            
            # 归一化到 [0, 1]
            img_normalized = img_rgb.astype('float32') / 255.0
            
            # 标准化 (使用ImageNet的均值和标准差)
            # mean = [0.485, 0.456, 0.406]
            # std = [0.229, 0.224, 0.225]
            # (img_normalized - mean) / std
            # paddle.vision.transforms.Normalize(mean=mean, std=std) 也可以实现此功能
            # 这里手动实现以保持与项目中其他部分的一致性
            mean_vals = np.array([0.485, 0.456, 0.406], dtype='float32')
            std_vals = np.array([0.229, 0.224, 0.225], dtype='float32')
            img_standardized = (img_normalized - mean_vals) / std_vals
            
            # 转换图像格式 HWC (高, 宽, 通道) -> CHW (通道, 高, 宽)
            img_tensor_chw = img_standardized.transpose((2, 0, 1))
            
            # 确保返回的是float32类型的numpy数组，DataLoader会自动转换为Tensor
            return img_tensor_chw.astype('float32'), label
            
        except Exception as e:
            # 捕获其他潜在错误，并返回随机数据
            print(f"处理图像 {image_path} (索引 {idx}) 时发生意外错误: {e}。将使用随机数据替代。")
            random_img_tensor = np.random.rand(3, self.image_size, self.image_size).astype('float32')
            return random_img_tensor, label
    
    def __len__(self):
        """返回数据集中样本的总数"""
        return len(self.image_paths_and_labels)

def create_dataloader(data_list_file, image_size=64, batch_size=32, mode='train', num_workers=0):
    """
    创建PaddlePaddle数据加载器 (DataLoader)
    Args:
        data_list_file (str): 数据列表文件的路径
        image_size (int): 图像预处理后的目标尺寸
        batch_size (int): 每个批次加载的样本数量
        mode (str): 数据集模式，'train' 或 'test'
        num_workers (int): 用于数据加载的子进程数量。默认为0，表示在主进程中加载数据。
                           在Windows上设为0通常更稳定，Linux上可以尝试增加以加速。
    Returns:
        paddle.io.DataLoader: 配置好的数据加载器实例
    """
    dataset = FaceDataset(data_list_file, image_size, mode)
    
    # 根据模式确定是否随机打乱数据
    # 训练模式 (mode='train') 时通常需要打乱数据，测试模式则不需要
    shuffle_data = (mode.lower() == 'train')
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,                    # PaddlePaddle Dataset实例
        batch_size=batch_size,      # 每个批次的大小
        shuffle=shuffle_data,       # 是否在每个epoch开始时打乱数据顺序
        num_workers=num_workers,    # 加载数据的子进程数
        drop_last=False             # 是否丢弃最后一个不完整的批次。训练时可以设为True，测试时通常为False以评估所有样本。
                                    # 当前项目中训练和测试均设为False，以处理所有数据。
    )
    
    return dataloader 