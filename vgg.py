# coding:utf-8
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class VGGFace(nn.Layer):
    """基于VGG16结构的人脸识别网络模型"""
    
    def __init__(self, num_classes, dropout_rate=0.5):
        """
        VGGFace 初始化函数
        Args:
            num_classes (int): 分类数量，即有多少个人需要识别
            dropout_rate (float): 在全连接层之后使用的Dropout比率，防止过拟合
        """
        super(VGGFace, self).__init__()
        
        # 第一个卷积块 (conv1)
        # 卷积层1_1: 输入3通道 (RGB), 输出64通道, 卷积核3x3, padding为1保持原尺寸
        self.conv1_1 = nn.Conv2D(3, 64, 3, padding=1)
        # 卷积层1_2: 输入64通道, 输出64通道, 卷积核3x3, padding为1
        self.conv1_2 = nn.Conv2D(64, 64, 3, padding=1)
        # 最大池化层1: 卷积核2x2, 步长2, 特征图尺寸减半
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        
        # 第二个卷积块 (conv2)
        self.conv2_1 = nn.Conv2D(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2D(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        
        # 第三个卷积块 (conv3)
        self.conv3_1 = nn.Conv2D(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2D(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2D(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2D(kernel_size=2, stride=2)
        
        # 第四个卷积块 (conv4)
        self.conv4_1 = nn.Conv2D(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2D(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2D(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2D(kernel_size=2, stride=2)
        
        # 第五个卷积块 (conv5)
        self.conv5_1 = nn.Conv2D(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2D(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2D(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2D(kernel_size=2, stride=2)
        
        # 全连接层 (Fully Connected Layers)
        # 假设输入图像为64x64，经过5次步长为2的最大池化后，特征图尺寸变为 64 / (2^5) = 64 / 32 = 2x2
        # 所以，第五个池化层输出的特征图数量为512，尺寸为2x2。展平后的维度为 512 * 2 * 2。
        self.fc6 = nn.Linear(512 * 2 * 2, 512) 
        self.dropout1 = nn.Dropout(dropout_rate) # Dropout层
        
        self.fc7 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(dropout_rate) # Dropout层
        
        # 特征输出层 (用于人脸对比)
        # 这一层输出的512维向量可以作为人脸的紧凑特征表示
        self.feature_out = nn.Linear(512, 512) 
        
        # 分类层 (用于人脸识别)
        # 最终的分类器，输出每个类别的原始得分 (logits)
        self.fc8 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        """模型的前向传播过程"""
        # 确保输入数据类型为float32，这是大多数深度学习框架卷积等操作的要求
        x = paddle.cast(x, 'float32')
        
        # 卷积块1
        x = F.relu(self.conv1_1(x)) # 卷积后通常接ReLU激活函数
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
        
        # 展平操作 (Flatten)
        # 将多维的特征图展平成一维向量，以输入到全连接层
        # x.shape[0] 是批大小 (batch_size)
        # -1 表示自动计算该维度的大小，这里是 512*2*2
        x = paddle.reshape(x, [x.shape[0], -1])
        
        # 全连接层
        x = F.relu(self.fc6(x))
        x = self.dropout1(x) # 应用dropout
        
        x = F.relu(self.fc7(x))
        x = self.dropout2(x) # 应用dropout
        
        # 特征输出 (用于人脸对比)
        # 这里的 'feature' 可以用于计算两个人脸的相似度
        feature = self.feature_out(x)
        
        # 分类输出 (用于人脸识别)
        # 这里的 'logits' 是未经softmax激活的原始分类得分
        # 在计算损失函数 (如CrossEntropyLoss) 时，通常直接使用logits
        # 在推理预测时，会对logits应用softmax得到概率
        logits = self.fc8(feature)
        
        return feature, logits