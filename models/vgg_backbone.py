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

    def __init__(self, vgg_name: str = 'VGG16', dropout_rate: float = 0.5):
        """
        VGGBackbone 初始化函数。

        Args:
            vgg_name (str, optional): 指定要使用的VGG网络变体。必须是 `cfg` 字典中的一个键。
                                      默认为 'VGG16'。
            dropout_rate (float, optional): 在特征输出层之前的Dropout比率。默认为 0.5。
        """
        super(VGGBackbone, self).__init__()
        if vgg_name not in cfg:
            raise ValueError(f"不支持的VGG变体名称: {vgg_name}。支持的包括: {list(cfg.keys())}")
        
        # 1. 构建VGG的卷积层序列 (特征提取主体)
        self.features = self._make_layers(cfg[vgg_name])
        
        # 2. 自适应平均池化层
        # 将每个通道的特征图池化为一个 2x2 的小特征图 (或者其他自定义尺寸)。
        # 这有助于处理不同输入图像尺寸，并为后续的全连接层提供固定大小的输入。
        # 原VGG在进入FC层前，特征图尺寸固定为7x7 (对于224x224输入)。
        # AdaptiveAvgPool2D((H_out, W_out)) 可以池化到任意指定的输出尺寸。
        self.adaptive_pool = nn.AdaptiveAvgPool2D((2, 2)) # 输出2x2的特征图
        
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
        fc_in_features = last_conv_channels * 2 * 2 # 512 * 2 * 2 = 2048
        
        # 输出特征维度固定为512，如果需要其他维度，可以在实例化时修改或添加参数
        # 注意: 这个512维输出是人为设定的，如果需要与ArcFace等头部配合，应确保其与头部的in_features匹配。
        # 配置中的 `model.vgg_params.feature_dim` 应控制这里的输出维度。
        # TODO: 使这里的输出维度可配置，例如通过构造函数参数传入。
        # 暂时固定为512，与ResNetFace的常见输出维度保持一致，方便在工厂中切换。
        self.feature_out_dim = 512 

        self.feature_out_layer = nn.Sequential(
            nn.Linear(fc_in_features, 1024), # 类似于原VGG的fc6 (通常输出4096，这里减小)
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, self.feature_out_dim), # 类似于原VGG的fc7 (通常输出4096，这里改为目标特征维度)
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
                conv2d = nn.Conv2D(in_channels, v_item, kernel_size=3, padding=1)
                # VGG原文不含BN，但现代实践中常加入BN以改善训练
                layers.extend([
                    conv2d,
                    nn.BatchNorm2D(v_item), # 添加BN层
                    nn.ReLU()   # ReLU激活
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