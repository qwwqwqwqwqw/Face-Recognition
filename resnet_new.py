import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class ConvBNLayer(nn.Layer):
    '''卷积+批归一化层'''
    def __init__(self, ch_in, ch_out, filter_size=3, stride=1, groups=1, padding=0, act=None):
        '''
        ConvBNLayer 初始化函数
        Args:
            ch_in (int): 输入特征图的通道数
            ch_out (int): 输出特征图的通道数
            filter_size (int or tuple): 卷积核大小
            stride (int or tuple): 卷积步长
            padding (int or tuple): 填充大小
            act (callable): 激活函数
        '''
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False  # 批归一化层包含了偏置，所以卷积层不需要偏置
        )
        self.bn = nn.BatchNorm2D(ch_out)
        self.act = act

    def forward(self, x):
        '''前向传播'''
        x = self.conv(x)
        x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x

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
                ch_in=in_channels,
                ch_out=out_channels,
                filter_size=1,  # 1x1卷积
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
    expansion = 1
    def __init__(self, ch_in, ch_out, stride, shortcut):
        '''
        BasicBlock 初始化函数
        Args:
            ch_in (int): 输入特征图的通道数
            ch_out (int): 输出特征图的通道数
            stride (int or tuple): 第一个卷积层的步长，用于控制是否下采样
            shortcut (callable): 残差连接的shortcut函数
        '''
        super(BasicBlock, self).__init__()
        # 第一个卷积层，可能进行下采样
        self.conv1 = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=3,
            stride=stride,
            padding=1
        )
        # 第二个卷积层，步长固定为1
        self.conv2 = ConvBNLayer(
            ch_in=ch_out,
            ch_out=ch_out,
            filter_size=3,
            stride=1,
            padding=1
        )
        # 残差连接
        self.shortcut = shortcut

    def forward(self, x):
        '''前向传播'''
        identity = x  # 保存输入作为残差项
        out = self.conv1(x)
        out = F.relu(out)  # 第一个卷积和BN后接ReLU
        out = self.conv2(out)
        # 注意：第二个卷积和BN后不立即接ReLU

        if self.shortcut:
            identity = self.shortcut(x)
        
        out = out + identity  # 主路径输出与shortcut路径输出相加
        out = F.relu(out)  # 相加后再接ReLU
        return out

class ResNetFace(nn.Layer):
    '''基于ResNet的人脸识别网络模型'''
    def __init__(self, nf=32, n=3, feature_dim=512, name="resnet"):
        '''
        ResNetFace 初始化函数 (仅作为特征提取器)
        Args:
            nf (int): ResNet初始卷积核数量 (如32, 64)
            n (int): 每个残差块组中BasicBlock的数量 (通常为2, 3, 4等，对应ResNet18/34/50的配置)
            feature_dim (int): 输出特征的维度，例如512
            name (str): 网络名称
        '''
        super(ResNetFace, self).__init__(name)
        self.feature_dim = feature_dim
        
        # 初始卷积层
        self.conv1 = ConvBNLayer(
            ch_in=3, 
            ch_out=nf, 
            filter_size=3, 
            stride=1, 
            padding=1, 
            act=F.relu) # 初始卷积后加ReLU

        # 残差块组
        # 根据n的值确定每个阶段的块数，这里为了简化，n直接代表块数
        # ResNet18/34: n 通常是一个列表如 [2,2,2,2] 或 [3,4,6,3]
        # 这里n是单个值，假设每个stage的block数相同或通过n推断
        # 为了更灵活，可以让n成为一个列表，如 [n0, n1, n2, n3]
        # 简单起见，我们先用一个统一的n值，并假定有3个stage
        
        # Stage 1
        self.layer1 = self._make_layer(BasicBlock, nf, nf, n, stride=1)
        # Stage 2
        self.layer2 = self._make_layer(BasicBlock, nf * BasicBlock.expansion, nf * 2, n, stride=2)
        # Stage 3
        self.layer3 = self._make_layer(BasicBlock, nf * 2 * BasicBlock.expansion, nf * 4, n, stride=2)
        # Stage 4 (可选，如果需要更深的网络)
        self.layer4 = self._make_layer(BasicBlock, nf * 4 * BasicBlock.expansion, nf * 8, n, stride=2)
        
        # 根据最后一个stage的输出通道计算展平前的特征数量
        # 如果有layer4, in_c = nf * 8 * BasicBlock.expansion
        # 如果只有layer3, in_c = nf * 4 * BasicBlock.expansion
        # 假设图像大小64x64, 经过3次stride=2的下采样 (conv1的stride=1, layer2, layer3, layer4的stride=2)
        # 64 -> 32 (layer2) -> 16 (layer3) -> 8 (layer4)
        # 所以最终特征图大小是 8x8
        
        # 使用AdaptiveAvgPool2D来处理不同输入尺寸，并得到固定的池化输出
        self.pool = nn.AdaptiveAvgPool2D(1)
        
        in_c = nf * 8 * BasicBlock.expansion # 假设有layer4
        
        # 特征提取的全连接层
        self.feature_output_layer = nn.Linear(in_c, self.feature_dim)

    def _make_layer(self, block, ch_in, ch_out, count, stride):
        '''
        辅助函数，用于构建包含多个残差块的层
        Args:
            block (nn.Layer): 残差块的类型 (例如 BasicBlock)
            ch_in (int): 该层的输入通道数
            ch_out (int): 该层的输出通道数
            count (int): 该层包含的残差块数量
            stride (int): 该层第一个残差块的步长 (用于控制是否下采样)
        Returns:
            nn.Sequential: 包含多个残差块的序列模块
        '''
        layers = []
        # 第一个残差块，可能需要改变输入通道数或进行下采样 (通过stride控制)
        shortcut = Shortcut(ch_in, ch_out * block.expansion, stride) if stride != 1 or ch_in != ch_out * block.expansion else None
        layers.append(block(ch_in, ch_out, stride, shortcut))
        # 后续的残差块，输入输出通道数不变，步长为1
        for _ in range(1, count):
            layers.append(block(ch_out * block.expansion, ch_out, 1, None)) # shortcut为None因为通道数和尺寸不变
        return nn.Sequential(*layers)

    def forward(self, x):
        '''前向传播'''
        # 确保输入是float32类型
        x = paddle.cast(x, 'float32')
        # 如果需要，在此处进行数据预处理，例如减去均值 (如旧版的 x = x - 128.0)
        # 但更推荐在数据加载器 (MyReader.py) 中完成预处理。

        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # 应用第四个stage

        # 全局平均池化
        x = self.pool(x)
        # 展平操作，将 [N, C, H, W] 展平为 [N, C*H*W]
        # H和W此时应为1，所以变为 [N, C]
        x = paddle.flatten(x, start_axis=1)

        # 特征层
        feature = self.feature_output_layer(x)

        return feature # 只返回特征

# ArcFace Loss的头部，负责计算最终的loss
class ArcFaceHead(nn.Layer):
    """
    ArcFace Loss 计算头
    Args:
        in_features (int): 输入特征的维度 (例如 backbone 输出的512维特征)
        out_features (int): 分类数量 (num_classes)
        margin1 (float): ArcFace的m1参数 (cos(m1*theta + m2) - m3)
        margin2 (float): ArcFace的m2参数
        margin3 (float): ArcFace的m3参数
        scale (float): ArcFace的尺度因子s
    """
    def __init__(self, in_features, out_features, margin1=1.0, margin2=0.5, margin3=0.0, scale=64.0):
        super(ArcFaceHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m1 = margin1
        self.m2 = margin2
        self.m3 = margin3
        self.s = scale

        # ArcFace的权重矩阵 W
        # W的形状是 [in_features, out_features]
        self.weight = self.create_parameter(
            shape=[in_features, out_features],
            dtype='float32',
            default_initializer=nn.initializer.XavierNormal() # 使用Xavier初始化
        )

    def forward(self, features, label):
        """
        前向传播计算ArcFace loss
        Args:
            features (Tensor): 从骨干网络提取的特征，形状 [batch_size, in_features]
            label (Tensor): 真实标签，形状 [batch_size]
        Returns:
            loss (Tensor): 计算得到的ArcFace loss
            softmax_output (Tensor): 经过ArcFace处理后的softmax概率，可用于评估
        """
        # 1. L2归一化输入特征 X
        # features.shape = [batch_size, in_features]
        features_norm = F.normalize(features, p=2, axis=1)
        
        # 2. L2归一化权重矩阵 W
        # self.weight.shape = [in_features, out_features]
        weight_norm = F.normalize(self.weight, p=2, axis=0) # 对每个类别的权重向量(列向量)进行归一化
        
        # 3. 计算余弦相似度 (X_norm * W_norm)
        # cosine_logits.shape = [batch_size, out_features]
        cosine_logits = paddle.matmul(features_norm, weight_norm)
        
        # 4. 应用 margin_cross_entropy 函数计算loss和softmax输出
        loss, softmax_output = F.margin_cross_entropy(
            logits=cosine_logits, 
            label=label, 
            margin1=self.m1, 
            margin2=self.m2, 
            margin3=self.m3, 
            scale=self.s,
            return_softmax=True, # 需要返回softmax输出来计算准确率
            reduction='mean'     # 对loss进行平均
        )
        
        return loss, softmax_output

if __name__ == '__main__':
    print("开始测试 ResNetFace 特征提取器...")
    # 假设输入图像是 64x64x3
    test_image = paddle.randn([4, 3, 64, 64], dtype='float32')
    
    # 实例化模型
    # ResNetFace现在只输出特征，不需要num_classes
    resnet_model = ResNetFace(nf=32, n=2, feature_dim=256) # 使用较小的n和feature_dim进行测试
    resnet_model.eval() # 设置为评估模式
    
    # 前向传播
    with paddle.no_grad():
        features = resnet_model(test_image)
    
    print(f"ResNetFace 输出特征形状: {features.shape}") # 预期: [4, feature_dim]
    if features.shape[0] == 4 and features.shape[1] == 256:
        print("ResNetFace (特征提取器模式) 初步测试成功!")
    else:
        print(f"ResNetFace (特征提取器模式) 测试失败! 输出形状与预期不符。")

    print("\n开始测试 ArcFaceHead...")
    batch_s = 4
    feat_dim = 256 # 与上面ResNetFace输出的feature_dim一致
    num_cls = 10   # 假设有10个类别

    # 模拟特征输入和标签
    sim_features = paddle.randn([batch_s, feat_dim], dtype='float32')
    sim_labels = paddle.randint(0, num_cls, [batch_s], dtype='int64')

    # 实例化ArcFaceHead
    arcface_head = ArcFaceHead(in_features=feat_dim, out_features=num_cls)
    arcface_head.eval()

    # 前向传播
    with paddle.no_grad():
        arc_loss, arc_softmax = arcface_head(sim_features, sim_labels)

    print(f"ArcFaceHead Loss: {arc_loss.item()}")
    print(f"ArcFaceHead Softmax 输出形状: {arc_softmax.shape}") # 预期: [batch_s, num_cls]
    
    if arc_softmax.shape[0] == batch_s and arc_softmax.shape[1] == num_cls and arc_loss.item() > 0:
        print("ArcFaceHead 初步测试成功!")
    else:
        print("ArcFaceHead 测试失败!")

    # 模拟联合测试
    print("\n开始测试 ResNetFace + ArcFaceHead 联合...")
    resnet_backbone = ResNetFace(nf=32, n=2, feature_dim=feat_dim)
    arcface_head_combined = ArcFaceHead(in_features=feat_dim, out_features=num_cls)
    
    resnet_backbone.eval()
    arcface_head_combined.eval()

    with paddle.no_grad():
        extracted_features = resnet_backbone(test_image) # test_image: [4, 3, 64, 64]
        final_loss, final_softmax = arcface_head_combined(extracted_features, sim_labels) # sim_labels: [4]

    print(f"联合测试 Loss: {final_loss.item()}")
    print(f"联合测试 Softmax 输出形状: {final_softmax.shape}")
    if final_softmax.shape[0] == test_image.shape[0] and final_softmax.shape[1] == num_cls and final_loss.item() > 0:
        print("ResNetFace + ArcFaceHead 联合初步测试成功!")
    else:
        print("ResNetFace + ArcFaceHead 联合测试失败!")

    # 以下是旧的 ResNetFace 测试代码，已被上面的新测试取代
    # if __name__ == '__main__':
    # print("开始测试新版ResNetFace模型...")
    #     # 假设输入图像是 64x64x3，类别数为5
    # test_image = paddle.randn([2, 3, 64, 64], dtype='float32')
    # num_classes_test = 5
    # feature_dim_test = 128 # 测试时使用小一点的特征维度
    
    #     # 实例化模型
    #     # model = ResNetFace(num_classes=num_classes_test, nf=32, n=2, feature_dim=feature_dim_test) # 使用较小的n进行测试
    # model = ResNetFace(num_classes=num_classes_test, nf=32, n=2) # 使用默认feature_dim=512
    # model.eval() # 设置为评估模式
    
    #     # 前向传播
    # with paddle.no_grad():
    #         features, logits = model(test_image)
    
    # print(f"输入图像形状: {test_image.shape}")
    # print(f"输出特征形状: {features.shape}") # 预期: [2, feature_dim]
    # print(f"输出Logits形状: {logits.shape}") # 预期: [2, num_classes_test]
    
    # if features.shape[1] == 512 and logits.shape[1] == num_classes_test: # 检查默认feature_dim
    # print("New ResNetFace model preliminary test successful!")
    # else:
    # print(f"New ResNetFace model test failed. Output shapes are not as expected. Feature dim: {features.shape[1]}, Logit classes: {logits.shape[1]}") 