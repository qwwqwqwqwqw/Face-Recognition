import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# VGG网络配置字典
# 键代表不同的VGG变体 (如VGG11, VGG13, VGG16, VGG19)
# 值是一个列表，其中数字代表卷积层的输出通道数，'M' 代表一个最大池化层。
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGGBackbone(nn.Layer):
    """VGG骨干网络实现，用作特征提取器。

    该网络基于经典的VGG架构，移除了原始VGG网络末尾的分类全连接层，
    替换为一个自适应平均池化层和一个用于输出指定维度特征的线性层。
    这使其更适合作为通用的特征提取骨干，特别是当输入图像尺寸可变时。
    """

    def __init__(self, vgg_name: str = 'VGG16', dropout_rate: float = 0.5, feature_out_dim: int = 512):
        """
        VGGBackbone 初始化函数。

        Args:
            vgg_name (str, optional): 指定要使用的VGG网络变体。必须是 `cfg` 字典中的一个键。
                                      默认为 'VGG16'。
            dropout_rate (float, optional): 在特征输出层之前的Dropout比率。默认为 0.5。
            feature_out_dim (int, optional): 最终输出特征向量的维度。默认为 512。
        """
        super(VGGBackbone, self).__init__()
        if vgg_name not in cfg:
            raise ValueError(f"不支持的VGG变体名称: {vgg_name}。支持的包括: {list(cfg.keys())}")
        
        self.vgg_name = vgg_name # 保存vgg_name，可能在其他地方需要
        # 1. 构建VGG的卷积层序列 (特征提取主体)
        self.features = self._make_layers(cfg[vgg_name])
        
        #池化层:
        # 比喻: 就像对探测结果进行"总结和压缩"。它不关心精确的位置，只关心在某个区域内有没有探测到某个特征。
        #怎么做的？ 在特征图上划分出一个个小区域（比如 2x2 的窗口），然后对每个区域内的数字进行总结。
        #最大池化 (MaxPool2D): 取区域内的最大值。这有助于保留最强的特征响应，同时减小尺寸。
        #平均池化 (AvgPool2D): 取区域内的平均值。提供更平滑的特征表示。
        #自适应平均池化 (AdaptiveAvgPool2D): 我的 ResNetFace 用到了这个，很重要！ 
        #    它不像前两者需要指定固定的窗口大小和步长，而是指定输出特征图的目标尺寸（通常是 1x1）。比如 nn.AdaptiveAvgPool2D(1) 会把输入的任意尺寸的特征图（例如 [通道数, H, W]）都平均池化成 [通道数, 1, 1]。
        #参数: 最大池化和平均池化需要 kernel_size (窗口大小) 和 stride。自适应池化需要 output_size。
        # 输出: 一个尺寸更小的特征图 Tensor，形状是 [batch_size, 通道数, 变小的高度, 变小的宽度]。也是一个 4 维 Tensor。
        # 池化层不会改变通道数。

        # 2. 自适应平均池化层
        # 将每个通道的特征图池化为一个 2x2 的小特征图 (或者其他自定义尺寸)。
        # 这有助于处理不同输入图像尺寸，并为后续的全连接层提供固定大小的输入。
        # 原VGG在进入FC层前，特征图尺寸固定为7x7 (对于224x224输入)。
        # AdaptiveAvgPool2D((H_out, W_out)) 可以池化到任意指定的输出尺寸。
        self.adaptive_pool_output_size = (2, 2) # 可以考虑未来也让这个可配置
        self.adaptive_pool = nn.AdaptiveAvgPool2D(self.adaptive_pool_output_size) # 输出2x2的特征图
        
        # 3. 特征输出层 (类似原始VGG的分类器部分，但这里只输出特征)
        # 计算 adaptive_pool 输出后的展平维度：
        # 最后一个卷积层的输出通道数 (cfg[vgg_name]中最后一个数字，或512) * pool_out_h * pool_out_w
        # 例如，对于VGG16，最后一个卷积层输出512通道，池化为2x2，则输入维度为 512 * 2 * 2 = 2048
        last_conv_channels = 0
        for x_cfg_item in reversed(cfg[vgg_name]): # 从后向前查找最后一个卷积层的通道数
            if isinstance(x_cfg_item, int):
                last_conv_channels = x_cfg_item
                break
        if last_conv_channels == 0: # 以防万一配置错误
            last_conv_channels = 512 
            print(f"警告: 未能在VGG配置 '{vgg_name}' 中明确找到最后一个卷积层的输出通道数，默认使用512。")
        
        # fc6: 第一个全连接层 (这里用作特征输出层)
        # 原VGG的fc6输入是 512*7*7=25088 (对于224x224输入和7x7池化后)
        # 当前设计是 最后一个卷积通道数 * adaptive_pool_H * adaptive_pool_W
        fc_in_features = last_conv_channels * self.adaptive_pool_output_size[0] * self.adaptive_pool_output_size[1] # 使用 self.adaptive_pool_output_size
        
        # 输出特征维度固定为512，如果需要其他维度，可以在实例化时修改或添加参数
        # 注意: 这个512维输出是人为设定的，如果需要与ArcFace等头部配合，应确保其与头部的in_features匹配。
        # 配置中的 `model.vgg_params.feature_dim` 应控制这里的输出维度。
        # TODO: 使这里的输出维度可配置，例如通过构造函数参数传入。 -- 已通过 feature_out_dim 参数实现
        # 暂时固定为512，与ResNetFace的常见输出维度保持一致，方便在工厂中切换。
        self.feature_out_dim = feature_out_dim # 使用传入的参数

        self.feature_out_layer = nn.Sequential(
            #将输入的特征向量 (in_features 维度) 线性地映射到输出向量 (out_features 维度)
            nn.Linear(fc_in_features, 1024), # 类似于原VGG的fc6 (通常输出4096，这里减小)
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, self.feature_out_dim), # 使用 self.feature_out_dim
            # nn.ReLU(True), # 特征输出前是否需要ReLU可以讨论，通常不需要
            # nn.Dropout(dropout_rate) # 特征输出前是否需要Dropout也可以讨论
        )
        
        # 初始化权重 (可选，但推荐)
        self._initialize_weights()

    def _make_layers(self, cfg_list: list) -> nn.Sequential:
        """
        根据VGG的配置列表构建卷积层序列。

        Args:
            cfg_list (list): VGG配置列表，例如 `[64, 'M', 128, 'M', ...]`。
                             数字代表卷积层的输出通道数，'M' 代表一个最大池化层。

        Returns:
            nn.Sequential: 包含VGG卷积和池化层的序列模块。
        """
        layers = []
        in_channels = 3 # 初始输入通道数为3 (RGB图像)
        for v_item in cfg_list:
            if v_item == 'M':
                # 添加最大池化层，kernel_size=2, stride=2
                layers.append(nn.MaxPool2D(kernel_size=2, stride=2))
            else:
               
                # 添加卷积层 (Conv2D) + 批归一化层 (BatchNorm2D) + ReLU激活函数
                # 卷积核大小为3x3，填充为1，以保持特征图尺寸（当stride=1时）
                #(nn.Conv2D 的初始化参数)说明:
                #in_channels (输入通道数): 输入特征图的通道数。第一层的输入是图片的通道数（彩色图通常是 3）。后面层的输入通道数是前面卷积层输出的通道数。
                #out_channels (输出通道数): 重要参数。 这个卷积层会使用 out_channels 个不同的探测器。
                # 每个探测器都会在输入特征图上滑动，生成一个对应的输出图。所以，这个卷积层最终会输出 out_channels 个输出图堆叠在一起，形成输出特征图。输出通道数越多，表示这个层能检测到的特征类型越多。
                #kernel_size (卷积核尺寸): 探测器的大小，通常是 3x3 或 1x1 或 5x5。
                #stride (步长): 探测器每次滑动的距离。步长大于 1 可以减小输出特征图的尺寸。
                #padding (填充): 在图片边缘填充额外的像素，通常是为了让输出特征图的尺寸更容易控制，或者避免边缘信息丢失。
                conv2d = nn.Conv2D(in_channels, v_item, kernel_size=3, padding=1)
                
                # VGG原文不含BN，但现代实践中常加入BN以改善训练（BatchNorm2D）
                #比喻: 就像在每一层神经网络的输入前进行一次"标准化考试"，确保输入数据的分布相对稳定。
                #怎么做的？ 对一个批次的数据在每个通道上计算均值和方差，然后用这些统计量对数据进行标准化，使其均值为 0，方差为 1。
                # 
                # 作用: 防止过拟合
                # 
                # 1. 加速训练收敛: 标准化输入数据可以减少梯度爆炸和消失问题，有助于模型更快地收敛。
                # 2. 提高模型稳定性: 标准化输入数据可以减少模型对初始化参数的依赖，有助于提高模型的泛化能力。
                # 3. 改善梯度传播: 标准化输入数据可以改善梯度传播，有助于提高模型的训练效果。
                layers.extend([
                    conv2d,
                    nn.BatchNorm2D(v_item), # 添加BN（批归一化，Batch Normalization）层：非常重要！ 加速训练的收敛速度，允许使用更大的学习率，并对网络参数的初始化不那么敏感，同时也有一定的正则化效果，防止过拟合。
                    nn.ReLU()   # ReLU激活，非常重要！
                    #如果没有激活函数，无论网络有多少层，它都只能学习到线性关系，无法处理复杂的非线性模式（而现实世界中的数据几乎都是非线性的）。非线性使得网络能够学习到更复杂的特征组合。
                #ReLU激活函数: 
                #比喻: 就像给神经元引入"兴奋"或"抑制"。它决定了神经元是否被"激活"以及激活的强度。
                #怎么做的？ 对卷积或池化层的输出进行一个非线性变换。最常用的是 ReLU (Rectified Linear Unit)，它的公式是 max(0, x)。如果输入是负数，输出就是 0；如果输入是正数，输出就是它本身。
                # 作用: 
                # 1. 加速训练收敛: 非线性变换可以增加模型的表达能力，有助于模型更快地收敛。
                # 2. 提高模型稳定性: 非线性变换可以减少模型对初始化参数的依赖，有助于提高模型的泛化能力。
                # 3. 改善梯度传播: 非线性变换可以改善梯度传播，有助于提高模型的训练效果。
                ])
                in_channels = v_item # 更新下一层的输入通道数
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """初始化网络权重。"""
        for m in self.sublayers(): # 使用 sublayers() 遍历所有子层
            if isinstance(m, nn.Conv2D):
                # 对卷积层使用Kaiming He正态初始化 (适用于ReLU激活)
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # PaddlePaddle 中更推荐使用内置的初始化器
                initializer = nn.initializer.KaimingNormal()
                initializer(m.weight)
                if m.bias is not None:
                    # 如果卷积层有偏置，则初始化为0
                    nn.initializer.Constant(value=0)(m.bias)
            elif isinstance(m, nn.BatchNorm2D):
                # 对批归一化层的权重初始化为1，偏置初始化为0
                nn.initializer.Constant(value=1)(m.weight)
                nn.initializer.Constant(value=0)(m.bias)
            elif isinstance(m, nn.Linear):
                # 对全连接层的权重使用正态分布初始化，偏置初始化为0
                # m.weight.data.normal_(0, 0.01)
                initializer = nn.initializer.Normal(std=0.01)
                initializer(m.weight)
                nn.initializer.Constant(value=0)(m.bias)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """前向传播过程，最终输出提取到的特征向量。"""
        # 确保输入是float32类型
        x = paddle.cast(x, 'float32')
        
        # 1. 通过VGG的卷积层序列
        x = self.features(x) # [N, C_last_conv, H_feat, W_feat]
        
        # 2. 通过自适应平均池化层
        x = self.adaptive_pool(x) # [N, C_last_conv, H_pool_out, W_pool_out]
                                  # 例如 [N, 512, 2, 2]
        
        # 3. 展平特征图，为输入到全连接层做准备
        x = paddle.flatten(x, start_axis=1) # [N, C_last_conv * H_pool_out * W_pool_out]
                                          # 例如 [N, 512 * 2 * 2] = [N, 2048]
        
        # 4. 通过特征输出层得到最终的特征向量
        feature = self.feature_out_layer(x) # [N, self.feature_out_dim]
                                            # 例如 [N, 512]
        return feature

if __name__ == '__main__':
    # 这是一个简单的单元测试/使用示例，用于验证VGGBackbone模型是否能正确实例化和运行。
    print("开始测试 VGGBackbone 特征提取器...")

    # 模拟一批输入图像 (batch_size=2, channels=3, height=64, width=64)
    # VGGBackbone由于有AdaptiveAvgPool2D，理论上可以处理不同尺寸的输入，
    # 但训练时通常使用固定尺寸。
    test_image_batch = paddle.randn([2, 3, 64, 64], dtype='float32')

    # 实例化模型参数示例
    vgg_variant = 'VGG16' # 可以尝试 'VGG11', 'VGG13', 'VGG19'
    dropout = 0.5
    
    print(f"使用VGG变体: {vgg_variant}, Dropout: {dropout}")
    vgg_model_test = VGGBackbone(vgg_name=vgg_variant, dropout_rate=dropout)
    
    # 将模型设置为评估模式
    vgg_model_test.eval()
    
    # 执行前向传播
    with paddle.no_grad():
        extracted_features = vgg_model_test(test_image_batch)
    
    print(f"输入图像批次形状: {test_image_batch.shape}")
    print(f"VGGBackbone ('{vgg_variant}') 输出特征形状: {extracted_features.shape}")
    
    # 验证输出形状是否符合预期 (当前feature_out_dim固定为512)
    expected_dim = vgg_model_test.feature_out_dim
    expected_shape = (test_image_batch.shape[0], expected_dim)
    
    if extracted_features.shape == expected_shape:
        print(f"VGGBackbone ('{vgg_variant}') 初步测试成功! 输出特征维度为 {expected_dim}。")
    else:
        print(f"VGGBackbone ('{vgg_variant}') 测试失败! 输出形状 {extracted_features.shape} 与预期 {expected_shape} 不符。")

    # 可以尝试不同输入尺寸
    print("\n测试不同输入尺寸 (128x128):")
    test_image_batch_large = paddle.randn([2, 3, 128, 128], dtype='float32')
    with paddle.no_grad():
        extracted_features_large = vgg_model_test(test_image_batch_large)
    print(f"输入图像批次形状 (大): {test_image_batch_large.shape}")
    print(f"VGGBackbone ('{vgg_variant}') 输出特征形状 (大输入): {extracted_features_large.shape}")
    if extracted_features_large.shape == expected_shape:
        print(f"VGGBackbone ('{vgg_variant}') 对大尺寸输入测试成功! 输出特征维度为 {expected_dim}。")
    else:
        print(f"VGGBackbone ('{vgg_variant}') 对大尺寸输入测试失败! 输出形状 {extracted_features_large.shape} 与预期 {expected_shape} 不符。")