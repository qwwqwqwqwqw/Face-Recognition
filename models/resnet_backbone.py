# resnet_backbone.py
# 该模块实现了 ResNet (Residual Network) 的骨干网络结构。
# ResNet 通过引入残差学习单元来解决深度神经网络训练中的梯度消失和模型退化问题，
# 使得构建非常深的网络成为可能。
# 此处定义的 ResNetFace 类专门用作人脸特征提取器。

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class ConvBNLayer(nn.Layer):
    """卷积 + 批归一化 + 可选激活函数 的组合层。

    这是一个辅助类，将标准的卷积操作、批归一化和激活函数（如果提供）封装在一起，
    简化了网络构建的代码。批归一化层有助于加速训练收敛并提高模型的泛化能力。
    """
    def __init__(self, ch_in: int, ch_out: int,
                 filter_size: int | tuple[int, int] = 3,
                 stride: int | tuple[int, int] = 1,
                 groups: int = 1,
                 padding: int | tuple[int, int] | str = 0,
                 act: callable = None):
        """
        ConvBNLayer 初始化函数。

        Args:
            ch_in (int): 输入特征图的通道数。
            ch_out (int): 输出特征图的通道数。
            filter_size (int or tuple, optional): 卷积核的尺寸。可以是一个整数（方核）或一个包含两个整数的元组 (kh, kw)。
                                              默认为 3。
            stride (int or tuple, optional): 卷积操作的步长。默认为 1。
            groups (int, optional): 分组卷积的组数。默认为 1 (标准卷积)。
            padding (int or tuple or str, optional): 填充大小。可以是整数、元组或字符串（如 'SAME', 'VALID'）。
                                                 默认为 0 (无填充)。对于 filter_size=3, padding=1 通常用于保持特征图尺寸不变。
            act (callable, optional): 激活函数。例如 `F.relu` 或 `F.sigmoid`。
                                      如果为 `None`，则不应用激活函数。默认为 `None`。
        """
        super(ConvBNLayer, self).__init__()
        # 定义卷积层
        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False  # 批归一化层会学习偏置，因此卷积层通常不使用偏置
        )
        # 定义批归一化层
        self.bn = nn.BatchNorm2D(ch_out)
        # 激活函数
        self.act = act

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """前向传播过程。"""
        x = self.conv(x)
        x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x

class Shortcut(nn.Layer):
    """ResNet的残差连接（Shortcut Connection）模块。

    在ResNet中，残差块的输出是其输入（恒等映射）与卷积变换结果的和。
    当输入和卷积变换结果的维度（通道数或空间尺寸）不一致时，
    需要通过此 Shortcut 模块对输入进行调整（通常是使用1x1卷积）以匹配维度，
    然后才能进行元素相加。
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int | tuple[int, int]):
        """
        Shortcut 初始化函数。

        Args:
            in_channels (int): 输入特征图的通道数 (即残差块的输入通道数)。
            out_channels (int): 期望的输出通道数 (应与残差块主路径卷积变换后的输出通道数一致)。
            stride (int or tuple): 应用于 shortcut 路径的卷积步长。
                                   当残差块的主路径进行了下采样（即stride > 1）时，
                                   shortcut 路径也需要进行相应的下采样以匹配空间尺寸。
        """
        super(Shortcut, self).__init__()
        # 检查是否需要进行维度调整：
        # 1. 输入通道数 (in_channels) 与主路径输出通道数 (out_channels) 是否不同。
        # 2. 是否需要下采样 (stride != 1)。
        self.needs_projection = (in_channels != out_channels or stride != 1)

        if self.needs_projection:
            # 如果需要调整，则使用一个 ConvBNLayer (1x1卷积) 来改变通道数和/或进行下采样。
            # 注意：这里的 ConvBNLayer 不带激活函数。
            self.conv = ConvBNLayer(
                ch_in=in_channels,
                ch_out=out_channels,
                filter_size=1,  # 1x1卷积，仅用于调整通道和尺寸
                stride=stride,  # 与主路径的第一个卷积层步长一致
                padding=0       # 1x1卷积通常不需要填充
            )
        else:
            self.conv = None # 如果不需要调整，则 shortcut 是一个恒等映射

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """前向传播过程。"""
        if self.conv is not None:
            # 如果定义了卷积层 (即需要维度调整)，则应用它
            return self.conv(x)
        else:
            # 否则，直接返回输入 (恒等映射)
            return x

class BasicBlock(nn.Layer):
    """ResNet的基础残差块 (Basic Residual Block)。

    对应于ResNet论文中较浅层网络（如ResNet-18, ResNet-34）使用的残差单元。
    它主要由两个3x3的卷积层组成。
    """
    expansion = 1 # 对于BasicBlock，输入通道数等于输出通道数（在块内部），因此扩展因子为1。

    def __init__(self, ch_in: int, ch_out: int, stride: int | tuple[int, int], shortcut: Shortcut | None):
        """
        BasicBlock 初始化函数。

        Args:
            ch_in (int): 输入特征图的通道数。
            ch_out (int): 块内卷积层的输出通道数。
                          对于BasicBlock，最终块的输出通道数也是 ch_out * expansion。
            stride (int or tuple): 第一个卷积层的步长。如果大于1，则该块会执行下采样。
            shortcut (Shortcut | None): 用于处理残差连接的Shortcut实例。
                                     如果为None，表示输入和输出维度一致且无需调整（通常在后续块中）。
                                     但更规范的做法是外部计算好shortcut并传入。
        """
        super(BasicBlock, self).__init__()
        # 第一个卷积层 (ConvBNLayer: Conv + BN + ReLU)
        # 如果stride > 1，这个卷积层会执行下采样
        self.conv1 = ConvBNLayer(
            ch_in=ch_in,
            ch_out=ch_out,
            filter_size=3,
            stride=stride,
            padding=1, # 3x3卷积配合padding=1可以保持空间尺寸（当stride=1时）
            act=F.relu # 第一个卷积后通常有ReLU激活
        )
        # 第二个卷积层 (ConvBNLayer: Conv + BN, ReLU在与shortcut相加后应用)
        self.conv2 = ConvBNLayer(
            ch_in=ch_out, # 输入通道是上一个卷积的输出通道
            ch_out=ch_out * self.expansion, # 输出通道乘以扩展因子
            filter_size=3,
            stride=1,   # 第二个卷积层步长固定为1
            padding=1,
            act=None    # 注意：第二个卷积和BN之后不立即接ReLU
        )
        # 残差连接路径
        self.shortcut = shortcut

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """前向传播过程。"""
        identity = x  # 保存输入，作为残差连接的恒等路径

        # 主路径的卷积变换
        out = self.conv1(x)
        # F.relu(out) # conv1内部已带relu
        out = self.conv2(out)

        # 处理残差连接
        if self.shortcut is not None:
            identity = self.shortcut(x) # 如果需要，对恒等路径进行维度调整

        # 将主路径输出与shortcut路径输出相加
        out = out + identity
        # 在相加之后应用ReLU激活函数
        out = F.relu(out)
        return out

# 注意: BottleneckBlock (用于ResNet-50/101/152) 在此文件中未实现。
# 如果需要更深的网络，可以类似地定义BottleneckBlock，它使用1x1, 3x3, 1x1的卷积序列，
# 并且通常有 expansion = 4。

class ResNetFace(nn.Layer):
    """基于ResNet架构的人脸特征提取器。

    该网络通过堆叠多个残差块来构建深度模型，旨在学习用于人脸识别的高判别力特征。
    网络的深度和宽度可以通过参数 `nf` (初始滤波器数量) 和 `n` (每个阶段的块数) 控制。
    最终输出一个固定维度的特征向量，由 `feature_dim` 参数指定。
    """

    def __init__(self, nf: int = 32, n: int | list[int] = 3, feature_dim: int = 512, name: str = "resnet_face"):
        """
        ResNetFace 初始化函数。

        Args:
            nf (int, optional): ResNet初始卷积层 (conv1) 的输出通道数（滤波器数量）。
                               控制网络的整体宽度。默认为 32。
            n (int or list[int], optional):
                如果为整数，表示每个残差阶段（stage）中 `BasicBlock` 的数量。
                如果为列表 (例如 `[2,2,2,2]` for ResNet-18 like, or `[3,4,6,3]` for ResNet-34 like)，
                则列表中的每个元素分别指定对应阶段的块数。当前实现简化为使用单个整数 `n`
                用于多个（当前为4个）阶段，或需要修改 `_make_layer` 调用以支持列表 `n`。
                为了演示，这里假设 `n` 是一个整数，并且有4个主要的残差阶段。
                默认为 3。
            feature_dim (int, optional): 网络最终输出的特征向量的维度。
                                       例如，常用于人脸识别的特征维度有 128, 256, 512等。
                                       默认为 512。
            name (str, optional): 网络的名称。默认为 "resnet_face"。
        """
        super(ResNetFace, self).__init__(name)
        self.feature_dim = feature_dim
        block_type = BasicBlock # 当前只使用BasicBlock

        # 初始卷积层：3通道输入 (RGB图像), nf通道输出
        # 通常包含一个较大的卷积核和步长，以快速降低空间分辨率并提取初步特征。
        # 但这里使用较小的filter_size=3, stride=1, padding=1，更像VGG的初始层，下采样主要由后续stage完成。
        self.conv1 = ConvBNLayer(
            ch_in=3,
            ch_out=nf,
            filter_size=3, # 原论文中通常是7x7, stride=2
            stride=1,      # 如果这里stride=1, 第一次下采样将在第一个stage的第一个block中进行
            padding=1,
            act=F.relu
        )
        # 可选的初始池化层 (原ResNet在conv1后有MaxPool)
        # self.pool1 = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        # 残差块组 (Stages)
        # 每个 _make_layer 调用构建一个阶段，包含多个残差块。
        # ch_in: 当前阶段的输入通道数
        # ch_out: 当前阶段内部卷积层的基准输出通道数 (实际输出会乘以 block.expansion)
        # count: 当前阶段包含的残差块数量 (来自参数 n)
        # stride: 当前阶段第一个残差块的步长 (如果为2，则该阶段执行下采样)

        # Stage 1: 输入 nf, 输出 nf * expansion (BasicBlock.expansion=1, 所以是 nf)
        # stride=1 表示此阶段不执行额外的空间下采样 (如果conv1的stride=1)
        self.layer1 = self._make_layer(block_type, nf, nf, n if isinstance(n, int) else n[0], stride=1)
        current_channels = nf * block_type.expansion

        # Stage 2: 输入 current_channels, 输出 (nf*2) * expansion
        # stride=2 表示此阶段开始时执行空间下采样
        self.layer2 = self._make_layer(block_type, current_channels, nf * 2, n if isinstance(n, int) else n[1], stride=2)
        current_channels = (nf * 2) * block_type.expansion

        # Stage 3: 输入 current_channels, 输出 (nf*4) * expansion
        # stride=2 表示此阶段继续空间下采样
        self.layer3 = self._make_layer(block_type, current_channels, nf * 4, n if isinstance(n, int) else n[2], stride=2)
        current_channels = (nf * 4) * block_type.expansion

        # Stage 4: 输入 current_channels, 输出 (nf*8) * expansion
        # stride=2 表示此阶段进一步空间下采样
        self.layer4 = self._make_layer(block_type, current_channels, nf * 8, n if isinstance(n, int) else n[3], stride=2)
        current_channels = (nf * 8) * block_type.expansion

        # 全局自适应平均池化层 (Adaptive Average Pooling)
        # 将每个通道的特征图池化为一个单一的值，输出形状为 [N, C, 1, 1]。
        # 这使得网络对输入图像的空间尺寸具有一定的鲁棒性。
        self.pool_adaptive = nn.AdaptiveAvgPool2D(output_size=1) # 输出尺寸为1x1

        # 全连接层 (Fully Connected Layer) / 特征输出层
        # 输入维度是最后一个残差阶段的输出通道数 (current_channels)。
        # 输出维度是预定义的特征维度 (self.feature_dim)。
        self.feature_output_layer = nn.Linear(current_channels, self.feature_dim)

    def _make_layer(self, block: type[BasicBlock],
                    ch_in: int, ch_out: int,
                    count: int, stride: int) -> nn.Sequential:
        """
        辅助函数，用于构建包含多个指定类型残差块的序列（一个ResNet阶段）。

        Args:
            block (type[BasicBlock]): 要使用的残差块的类 (例如 `BasicBlock`)。
            ch_in (int): 该阶段的输入通道数。
            ch_out (int): 该阶段中残差块内部卷积层的目标输出通道数。
                         实际输出通道数会是 `ch_out * block.expansion`。
            count (int): 该阶段包含的残差块数量。
            stride (int): 该阶段第一个残差块的卷积步长。如果为2，则该阶段执行下采样。

        Returns:
            nn.Sequential: 一个包含 `count` 个残差块的 `paddle.nn.Sequential` 容器。
        """
        layers = []
        # 第一个残差块：可能需要调整输入通道数或进行下采样 (通过stride和ch_in, ch_out的比较来决定shortcut类型)
        # Shortcut的输出通道数需要是 ch_out * block.expansion
        shortcut_block1 = Shortcut(ch_in, ch_out * block.expansion, stride)
        layers.append(block(ch_in, ch_out, stride, shortcut_block1))

        # 后续的残差块：输入通道数是 ch_out * block.expansion，输出通道数也是 ch_out * block.expansion
        # 步长固定为1，因此 shortcut 通常是恒等映射 (如果block内部不改变通道数)
        for _ in range(1, count):
            # 对于后续块，输入通道是前一个块的输出通道 (ch_out * block.expansion)
            # shortcut为None表示输入输出维度一致，不需要projection
            # (更严谨地，应始终创建Shortcut对象，它内部会判断是否需要projection)
            shortcut_others = Shortcut(ch_out * block.expansion, ch_out * block.expansion, 1)
            layers.append(block(ch_out * block.expansion, ch_out, 1, shortcut_others))

        return nn.Sequential(*layers)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """前向传播过程，最终输出提取到的人脸特征向量。"""
        # 确保输入是float32类型，这是PaddlePaddle模型通常期望的
        x = paddle.cast(x, 'float32')

        # 数据预处理（例如减均值、归一化等）通常在数据加载器 (MyReader.py) 中完成。
        # 如果需要在此处进行，可以添加相应操作。
        # 例如: x = (x - self.mean_value) / self.std_value

        # 初始卷积和激活
        x = self.conv1(x)
        # if hasattr(self, 'pool1'): x = self.pool1(x) # 如果有初始池化层

        # 通过各个残差阶段
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局自适应平均池化
        # 将 [N, C, H, W] 池化为 [N, C, 1, 1]
        x = self.pool_adaptive(x)

        # 展平操作 (Flatten)
        # 将 [N, C, 1, 1] 展平为 [N, C]，以便输入到全连接层
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)

        # 通过最终的全连接层得到特征输出
        feature = self.feature_output_layer(x)

        return feature # 作为特征提取器，只返回特征向量

if __name__ == '__main__':
    # 这是一个简单的单元测试/使用示例，用于验证ResNetFace模型是否能正确实例化和运行。
    print("开始测试 ResNetFace 特征提取器...")

    # 模拟一批输入图像 (batch_size=4, channels=3, height=64, width=64)
    # 图像尺寸可以根据实际情况调整，AdaptiveAvgPool2D使其具有一定灵活性
    test_image_batch = paddle.randn([4, 3, 64, 64], dtype='float32')

    # 实例化模型参数示例
    # nf: 初始滤波器数量，可以尝试不同的值如 32, 64
    # n: 每个阶段的块数。简单起见用整数，或用列表如 [2,2,2,2] (类ResNet18)
    # feature_dim: 期望的输出特征维度
    model_params = {
        'nf': 32,
        'n': 2,  # 每个stage有2个BasicBlock (类似更浅的ResNet)
        'feature_dim': 256 # 输出256维特征
    }
    resnet_model_test = ResNetFace(**model_params)

    # 将模型设置为评估模式，这会影响BatchNorm和Dropout等层的行为（如果存在Dropout）
    resnet_model_test.eval()

    # 执行前向传播
    # 使用 paddle.no_grad() 上下文管理器，在评估/推理时关闭梯度计算，以节省内存和计算
    with paddle.no_grad():
        extracted_features = resnet_model_test(test_image_batch)

    print(f"输入图像批次形状: {test_image_batch.shape}")
    print(f"ResNetFace ({model_params}) 输出特征形状: {extracted_features.shape}")

    # 验证输出形状是否符合预期
    expected_shape = (test_image_batch.shape[0], model_params['feature_dim'])
    if extracted_features.shape == expected_shape:
        print(f"ResNetFace (特征提取器模式，参数: {model_params}) 初步测试成功!")
    else:
        print(f"ResNetFace (特征提取器模式，参数: {model_params}) 测试失败! 输出形状 {extracted_features.shape} 与预期 {expected_shape} 不符。")

    # 更多测试可以包括：
    # - 尝试不同的 nf, n, feature_dim 组合
    # - 检查模型参数是否正确加载和更新（如果涉及训练）
    # - 传入不同尺寸的输入图像，验证AdaptiveAvgPool2D的作用