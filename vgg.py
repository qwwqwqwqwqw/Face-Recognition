# coding:utf-8
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class VGGFace(nn.Layer):
    """基于VGG的人脸识别网络"""
    
    def __init__(self, num_classes, dropout_rate=0.5):
        """
        初始化VGG网络
        
        Args:
            num_classes (int): 分类数量
            dropout_rate (float): Dropout比率
        """
        super(VGGFace, self).__init__()
        
        # 第一个卷积块
        self.conv1_1 = nn.Conv2D(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2D(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        
        # 第二个卷积块
        self.conv2_1 = nn.Conv2D(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2D(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        
        # 第三个卷积块
        self.conv3_1 = nn.Conv2D(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2D(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2D(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2D(kernel_size=2, stride=2)
        
        # 第四个卷积块
        self.conv4_1 = nn.Conv2D(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2D(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2D(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2D(kernel_size=2, stride=2)
        
        # 第五个卷积块
        self.conv5_1 = nn.Conv2D(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2D(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2D(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2D(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc6 = nn.Linear(512 * 2 * 2, 512)  # 假设输入为64x64，经过5次下采样变为2x2
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc7 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 特征输出层，用于人脸对比
        self.feature_out = nn.Linear(512, 512)
        
        # 分类层，用于人脸识别
        self.fc8 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        """前向传播"""
        # 确保输入是float32类型
        x = paddle.cast(x, 'float32')
        
        # 卷积块1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        
        # 卷积块2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        
        # 卷积块3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)
        
        # 卷积块4
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool4(x)
        
        # 卷积块5
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool5(x)
        
        # 展平
        x = paddle.reshape(x, [x.shape[0], -1])
        
        # 全连接层
        x = F.relu(self.fc6(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc7(x))
        x = self.dropout2(x)
        
        # 特征输出，用于人脸对比
        feature = self.feature_out(x)
        
        # 分类输出，用于人脸识别
        logits = self.fc8(x)
        
        return feature, logits 