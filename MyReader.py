# coding:utf-8
import os
import cv2
import random
import numpy as np
import paddle
from paddle.io import Dataset, DataLoader

class FaceDataset(Dataset):
    """人脸数据集"""
    def __init__(self, data_list, image_size=64, mode='train'):
        """
        初始化数据集
        Args:
            data_list (str): 数据列表文件路径
            image_size (int): 图像大小
            mode (str): 模式，'train'或'test'
        """
        super(FaceDataset, self).__init__()
        self.data_list = []
        self.image_size = image_size
        self.mode = mode
        
        # 读取数据列表
        if not os.path.exists(data_list):
            raise FileNotFoundError(f"找不到数据列表文件: {data_list}")
        
        with open(data_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 2:  # 确保格式正确
                    continue
                image_path = parts[0]
                label = int(parts[1])
                self.data_list.append((image_path, label))
    
    def __getitem__(self, idx):
        """
        获取单个样本
        Args:
            idx (int): 索引
        Returns:
            tuple: (image, label) 图像和标签
        """
        image_path, label = self.data_list[idx]
        
        # 读取图像
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 数据增强 (仅训练模式)
            if self.mode == 'train':
                # 随机水平翻转
                if random.random() > 0.5:
                    img = cv2.flip(img, 1)
                
                # 随机亮度和对比度调整
                if random.random() > 0.5:
                    alpha = 0.9 + random.random() * 0.2  # 0.9-1.1
                    beta = 10 * (random.random() - 0.5)  # -5 to 5
                    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            
            # 归一化
            img = img.astype('float32') / 255.0
            
            # 标准化
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img - mean) / std
            
            # 转置为CHW格式
            img = img.transpose((2, 0, 1))
            
            # 确保返回float32类型
            img = img.astype('float32')
            
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            # 返回一个随机生成的图像
            img = np.random.rand(3, self.image_size, self.image_size).astype('float32')
        
        return img, label
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data_list)

def create_dataloader(data_list, image_size=64, batch_size=32, mode='train'):
    """
    创建数据加载器
    Args:
        data_list (str): 数据列表文件路径
        image_size (int): 图像大小
        batch_size (int): 批大小
        mode (str): 模式，'train'或'test'
    Returns:
        DataLoader: 数据加载器
    """
    dataset = FaceDataset(data_list, image_size, mode)
    
    # 训练模式使用随机采样，测试模式使用顺序采样
    shuffle = (mode == 'train')
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False
    )
    
    return dataloader 