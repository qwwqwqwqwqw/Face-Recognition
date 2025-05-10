import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class ConvBNLayer(nn.Layer):
    '''卷积+批归一化层'''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        '''
        ConvBNLayer 初始化函数
        Args:
            in_channels (int): 输入特征图的通道数
            out_channels (int): 输出特征图的通道数
            kernel_size (int or tuple): 卷积核大小
            stride (int or tuple): 卷积步长
            padding (int or tuple): 填充大小
        '''
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias_attr=False  # 批归一化层包含了偏置，所以卷积层不需要偏置
        )
        self.bn = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        '''前向传播'''
        return self.bn(self.conv(x))

class Shortcut(nn.Layer):
    '''ResNet的残差连接shortcut模块'''
    def __init__(self, in_channels, out_channels, stride):
        '''
        Shortcut 初始化函数
        Args:
            in_channels (int): 输入特征图的通道数
            out_channels (int): 输出特征图的通道数 (可能与输入不同)
            stride (int or tuple): 卷积步长，用于当shortcut路径需要下采样时
        '''
        super(Shortcut, self).__init__()
        # 当输入输出通道数不同，或者步长不为1时（即需要下采样），使用1x1卷积调整维度
        if in_channels != out_channels or stride != 1:
            self.conv = ConvBNLayer(
                in_channels,
                out_channels,
                kernel_size=1,  # 1x1卷积
                stride=stride,
                padding=0
            )
        else:
            self.conv = None # 输入输出通道数相同且不需要下采样，则直接相加

    def forward(self, x):
        '''前向传播'''
        if self.conv is not None:
            return self.conv(x) # 应用1x1卷积调整
        else:
            return x # 直接返回输入

class BasicBlock(nn.Layer):
    '''ResNet的基础残差块'''
    def __init__(self, in_channels, out_channels, stride):
        '''
        BasicBlock 初始化函数
        Args:
            in_channels (int): 输入特征图的通道数
            out_channels (int): 输出特征图的通道数
            stride (int or tuple): 第一个卷积层的步长，用于控制是否下采样
        '''
        super(BasicBlock, self).__init__()
        # 第一个卷积层，可能进行下采样
        self.conv1 = ConvBNLayer(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )
        # 第二个卷积层，步长固定为1
        self.conv2 = ConvBNLayer(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        # 残差连接
        self.shortcut = Shortcut(in_channels, out_channels, stride)

    def forward(self, x):
        '''前向传播'''
        residual = x  # 保存输入作为残差项
        out = self.conv1(x)
        out = F.relu(out)  # 第一个卷积和BN后接ReLU
        out = self.conv2(out)
        # 注意：第二个卷积和BN后不立即接ReLU

        short = self.shortcut(residual) # 计算shortcut路径的输出
        out = out + short  # 主路径输出与shortcut路径输出相加
        out = F.relu(out)  # 相加后再接ReLU
        return out

class ResNetFace(nn.Layer):
    '''基于ResNet的人脸识别网络模型'''
    def __init__(self, num_classes, nf=32, n=3):
        '''
        ResNetFace 初始化函数
        Args:
            num_classes (int): 分类数量
            nf (int): 初始卷积层输出通道数 (filter数量)
            n (int): 每个残差块组中BasicBlock的数量 (对应旧版resnet.py中的n)
                     旧版resnet.py中 depth=20, n=(depth-2)//6 = 3
        '''
        super(ResNetFace, self).__init__()
        self.num_classes = num_classes

        # 初始卷积层 + BN + ReLU
        self.conv1 = ConvBNLayer(
            in_channels=3,  # 输入图像通道数
            out_channels=nf,
            kernel_size=3,
            stride=1,
            padding=1
        )
        # 注意：旧版 resnet.py 中有 ipt_bn = ipt - 128.0 的预处理，
        # 在新版中这部分通常在数据加载（MyReader.py）或forward开始时处理。

        # 残差块组
        # 第1组: nf -> nf*2, stride=2 (下采样)
        # 对应旧版: conv2 = layer_warp(basicblock, conv1, nf, nf*2, n, 2)
        self.layer1 = self._make_layer(BasicBlock, nf, nf * 2, n, stride=2)
        # 第2组: nf*2 -> nf*4, stride=2 (下采样)
        # 对应旧版: conv3 = layer_warp(basicblock, conv2, nf*2, nf*4, n, 2)
        self.layer2 = self._make_layer(BasicBlock, nf * 2, nf * 4, n, stride=2)
        # 第3组: nf*4 -> nf*8, stride=2 (下采样)
        # 对应旧版: conv4 = layer_warp(basicblock, conv3, nf*4, nf*8, n, 2)
        self.layer3 = self._make_layer(BasicBlock, nf * 4, nf * 8, n, stride=2)

        # 全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=1)

        # 特征层 (用于人脸对比)
        # 输入维度是最后一个残差块组的输出通道数 (nf*8)
        # 输出维度参考旧版设定为512
        self.feature_out = nn.Linear(nf * 8, 512)

        # 分类层 (用于人脸识别)
        # 输入维度是特征层的输出维度 (512)
        self.classifier = nn.Linear(512, num_classes)

    def _make_layer(self, block_type, in_channels, out_channels, blocks, stride=1):
        '''
        辅助函数，用于构建包含多个残差块的层
        Args:
            block_type (nn.Layer): 残差块的类型 (例如 BasicBlock)
            in_channels (int): 该层的输入通道数
            out_channels (int): 该层的输出通道数
            blocks (int): 该层包含的残差块数量
            stride (int): 该层第一个残差块的步长 (用于控制是否下采样)
        Returns:
            nn.Sequential: 包含多个残差块的序列模块
        '''
        layers = []
        # 第一个残差块，可能需要改变输入通道数或进行下采样 (通过stride控制)
        layers.append(block_type(in_channels, out_channels, stride))
        # 后续的残差块，输入输出通道数不变，步长为1
        for _ in range(1, blocks):
            layers.append(block_type(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        '''前向传播'''
        # 确保输入是float32类型
        x = paddle.cast(x, 'float32')
        # 如果需要，在此处进行数据预处理，例如减去均值 (如旧版的 x = x - 128.0)
        # 但更推荐在数据加载器 (MyReader.py) 中完成预处理。

        x = self.conv1(x)
        x = F.relu(x)  # 初始卷积和BN后接ReLU

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # 全局平均池化
        x = self.global_avg_pool(x)
        # 展平操作，将 [N, C, H, W] 展平为 [N, C*H*W]
        # H和W此时应为1，所以变为 [N, C]
        x = paddle.flatten(x, start_axis=1)

        # 特征层
        feature = self.feature_out(x)

        # 分类层
        logits = self.classifier(feature)  # 输出的是未经softmax的原始预测值 (logits)

        return feature, logits

if __name__ == '__main__':
    # 测试 ResNetFace 模型
    num_classes_test = 10  # 假设测试时有10个类别
    # 使用与旧版resnet.py中 n=(depth-2)//6 (depth=20时n=3) 和 nf=32 一致的参数
    model = ResNetFace(num_classes=num_classes_test, nf=32, n=3)
    model.eval()  # 设置为评估模式

    # 模拟一个输入批次 (batch_size=2, channels=3, height=64, width=64)
    dummy_input = paddle.randn([2, 3, 64, 64])

    # 前向传播
    feature, logits = model(dummy_input)

    print("输入形状:", dummy_input.shape)
    print("特征输出形状:", feature.shape)
    print("Logits 输出形状:", logits.shape)

    # 检查形状是否符合预期
    # 对于 nf=32, 最后一个残差块组输出通道数为 nf*8 = 32*8 = 256
    # feature_out 层是 nn.Linear(256, 512)
    assert feature.shape == [2, 512]  # 预期形状: [batch_size, 特征维度]
    assert logits.shape == [2, num_classes_test]  # 预期形状: [batch_size, 类别数量]

    print("New ResNetFace model preliminary test successful!") 