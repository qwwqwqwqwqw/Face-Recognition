# heads.py
# 该模块定义了项目中使用的各种模型头部 (Model Heads)。
# 模型头部通常连接在骨干特征提取网络之后，负责将提取到的特征
# 转换为适合特定损失函数计算的形式，或者直接输出最终的预测结果。
# 例如，ArcFaceHead 实现了 ArcFace Loss 的 logits 计算逻辑，
# 而 CrossEntropyHead 则是一个标准的线性分类头。

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class ArcFaceHead(nn.Layer):
    """
    ArcFace Loss 计算头 (基于 PaddlePaddle 的 `margin_cross_entropy` 函数实现)。

    ArcFace 是一种先进的度量学习损失函数，广泛应用于人脸识别等领域。
    它通过在角度空间中对特征和类别中心（权重）施加一个加性的角度边距 (additive angular margin)，
    从而增强类内样本的紧凑性和类间样本的可分性，学习到更具判别力的人脸特征。

    该头部模块主要负责以下工作：
    1. 接收从骨干网络提取的输入特征。
    2. 内部维护一个可学习的权重矩阵，其列向量可视为各类别中心的表示。
    3. 对输入特征和权重矩阵进行L2归一化。
    4. 计算归一化后的特征与归一化后的权重之间的余弦相似度（点积），得到初步的logits。
    5. 调用 `paddle.nn.functional.margin_cross_entropy`，它会在这些logits上应用角度边距，
       并结合真实标签计算最终的 ArcFace 损失值和经过 softmax 的概率输出。

    关键参数 (m1, m2, m3, scale) 对应 `paddle.nn.functional.margin_cross_entropy` 的接口：
    标准的 ArcFace (cos(theta + margin)) 可以通过设置 m1=1.0, m2=margin, m3=0.0 来实现。
    """

    def __init__(self, in_features: int, out_features: int,
                 margin1: float = 1.0, margin2: float = 0.5,
                 margin3: float = 0.0, scale: float = 64.0):
        """
        ArcFaceHead 初始化函数。

        Args:
            in_features (int): 输入特征的维度。这应与骨干网络输出的特征维度相匹配。
                               例如，如果骨干网络输出512维特征，则 `in_features`应为512。
            out_features (int): 分类的目标类别数量 (num_classes)。例如，数据集中有多少个不同的人。
            margin1 (float, optional): ArcFace的m1参数，对应 `paddle.nn.functional.margin_cross_entropy` 中的 `margin1`。
                                     在标准的 `cos(m1*θ + m2) - m3` 公式中，它通常设为1.0。
                                     默认为 1.0。
            margin2 (float, optional): ArcFace的m2参数，对应 `paddle.nn.functional.margin_cross_entropy` 中的 `margin2`。
                                     这通常代表原始ArcFace论文中的角度边距 `m`。
                                     默认为 0.5 (rad)。
            margin3 (float, optional): ArcFace的m3参数，对应 `paddle.nn.functional.margin_cross_entropy` 中的 `margin3`。
                                     通常设为0.0。
                                     默认为 0.0。
            scale (float, optional): ArcFace的尺度因子 `s`，对应 `paddle.nn.functional.margin_cross_entropy` 中的 `scale`。
                                   它用于放大归一化后的余弦值，有助于模型在训练初期更快收敛，并形成更清晰的决策边界。
                                   默认为 64.0。
        """
        super(ArcFaceHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m1 = margin1 # 对应 paddle API 中的 margin1 (通常为1.0)
        self.m2 = margin2 # 对应 paddle API 中的 margin2 (即原始论文中的 angular margin 'm')
        self.m3 = margin3 # 对应 paddle API 中的 margin3 (通常为0.0)
        self.s = scale    # 对应 paddle API 中的 scale 's' (尺度因子)

        # 可学习的权重参数，形状为 [in_features, out_features]
        # 每一列可以看作是对应类别的中心向量（在归一化后）
        self.weight = self.create_parameter(
            shape=[in_features, out_features],
            dtype='float32',
            default_initializer=nn.initializer.XavierNormal() # 使用Xavier正态分布初始化权重
        )

    def forward(self, features: paddle.Tensor, label: paddle.Tensor) -> tuple[paddle.Tensor, paddle.Tensor]:
        """
        ArcFaceHead 的前向传播过程。

        Args:
            features (paddle.Tensor): 从骨干网络提取的输入特征。
                                      形状应为 `[batch_size, in_features]`。
            label (paddle.Tensor): 对应输入特征的真实类别标签。
                                   形状应为 `[batch_size]`，且值为整数类别ID。

        Returns:
            tuple[paddle.Tensor, paddle.Tensor]:
                - loss (paddle.Tensor): 计算得到的ArcFace损失值（标量，如果reduction='mean'）。
                - softmax_output (paddle.Tensor): 经过ArcFace处理和softmax激活后的概率输出。
                                                形状为 `[batch_size, out_features]`。
                                                可用于计算训练过程中的准确率或其他评估指标。
        """
        # 1. 对输入特征进行 L2 归一化 (沿特征维度)
        features_norm = F.normalize(features, p=2, axis=1)

        # 2. 对内部权重矩阵进行 L2 归一化 (沿特征维度，即对每一列进行归一化)
        weight_norm = F.normalize(self.weight, p=2, axis=0)

        # 3. 计算归一化特征与归一化权重的点积，得到余弦相似度 logits
        #   features_norm: [batch_size, in_features]
        #   weight_norm:   [in_features, out_features]
        #   cosine_logits: [batch_size, out_features]
        cosine_logits = paddle.matmul(features_norm, weight_norm)

        # 4. 使用 paddle.nn.functional.margin_cross_entropy 计算最终损失和softmax输出
        #    该函数内部会处理角度边距的应用和softmax计算。
        loss, softmax_output = F.margin_cross_entropy(
            logits=cosine_logits,       # 输入的余弦相似度
            label=label,                # 真实标签
            margin1=self.m1,            # ArcFace参数 m1
            margin2=self.m2,            # ArcFace参数 m2 (角度边距 m)
            margin3=self.m3,            # ArcFace参数 m3
            scale=self.s,               # ArcFace参数 s (尺度因子)
            return_softmax=True,        # 要求函数返回softmax后的概率输出
            reduction='mean'            # 对batch内的损失进行平均
        )
        return loss, softmax_output

class CrossEntropyHead(nn.Layer):
    """
    标准的交叉熵损失分类头。

    该头部模块通常包含一个全连接（线性）层，将从骨干网络提取的特征
    映射到对应各个类别的原始得分（logits）。然后，这些logits可以与
    真实标签一起用于计算标准的交叉熵损失。
    """

    def __init__(self, in_features: int, out_features: int):
        """
        CrossEntropyHead 初始化函数。

        Args:
            in_features (int): 输入特征的维度。应与骨干网络输出的特征维度匹配。
            out_features (int): 分类的目标类别数量 (num_classes)。线性层将输出此数量的logits。
        """
        super(CrossEntropyHead, self).__init__()
        # 定义一个线性层，从输入特征维度映射到类别数量的维度
        self.fc = nn.Linear(in_features, out_features)

        # PaddlePaddle的CrossEntropyLoss默认在其内部执行softmax操作。
        # 因此，在训练时，我们将直接把线性层(self.fc)的输出（即原始logits）
        # 和真实标签传递给损失函数。
        self.loss_fn = paddle.nn.CrossEntropyLoss()

    def forward(self, features: paddle.Tensor, label: paddle.Tensor = None) -> tuple[paddle.Tensor | None, paddle.Tensor]:
        """
        CrossEntropyHead 的前向传播过程。

        Args:
            features (paddle.Tensor): 从骨干网络提取的输入特征。
                                      形状应为 `[batch_size, in_features]`。
            label (paddle.Tensor, optional): 对应输入特征的真实类别标签。
                                           形状应为 `[batch_size]`。
                                           如果为 `None`（例如在推理模式下，只进行预测而不计算损失），
                                           则不计算损失。默认为 `None`。

        Returns:
            tuple[paddle.Tensor | None, paddle.Tensor]:
                - loss (paddle.Tensor or None): 计算得到的交叉熵损失值（标量，如果reduction='mean'）。
                                                如果 `label` 为 `None`，则此值为 `None`。
                - logits (paddle.Tensor): 线性层 (self.fc) 的直接输出，即原始类别得分。
                                          形状为 `[batch_size, out_features]`。
                                          该输出可用于后续的softmax、argmax（获取预测类别）等操作。
        """
        # 通过线性层计算原始logits
        logits = self.fc(features)

        # 如果没有提供标签 (通常在推理/预测模式下)，则不计算损失
        if label is None:
            # 推理模式下，我们通常只关心logits（或其softmax后的概率）。
            # 为保持返回结构与训练时一致（双输出：损失，输出），将损失设为None。
            return None, logits
            # 或者，如果调用者明确区分训练和推理，可以设计推理时仅返回 logits:
            # return logits

        # 如果提供了标签 (通常在训练/评估模式下)，则计算损失
        loss = self.loss_fn(logits, label)

        # 注意：在训练和评估准确率时，我们通常直接使用logits进行argmax来获取预测类别，
        # 而不是使用softmax后的概率（因为softmax是单调的，不改变argmax的结果）。
        # 因此，这里返回原始logits，而不是softmax_output。
        return loss, logits 