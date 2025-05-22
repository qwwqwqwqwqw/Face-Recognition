# model_factory.py
# 该模块提供了工厂函数，用于根据配置动态创建项目的核心模型组件，
# 包括骨干网络 (backbone) 和模型头部 (head)。
# 这种设计旨在降低模型构建与具体实现之间的耦合度，
# 使得在不修改训练、推理等主流程代码的前提下，能够灵活地切换和组合不同的模型架构。

import paddle.nn as nn

from heads import ArcFaceHead, CrossEntropyHead
from models.resnet_backbone import ResNetFace
from models.model_with_cbam_arcface import ResNetFaceCBAM
from models.vgg_backbone import VGGBackbone

def get_backbone(config_model_params: dict, model_type_str: str, image_size: int) -> tuple[nn.Layer, int]:
    """
    根据配置创建并返回一个骨干网络实例及其输出的特征维度。

    Args:
        config_model_params (dict): 包含特定骨干网络所需参数的字典。
                                    例如，对于VGG，可能包含 `dropout_rate`；
                                    对于ResNet，可能包含 `nf`, `n_resnet_blocks`, `feature_dim`。
                                    通常对应于配置文件中 `config.model.vgg_params` 或 `config.model.resnet_params`。
        model_type_str (str): 指定要创建的骨干网络类型。目前支持 'vgg' 和 'resnet'。
        image_size (int): 输入图像的尺寸 (主要指高度或宽度，假设为正方形)。
                          某些骨干网络（如旧版VGG，如果FC层输入维度依赖于此）的初始化可能需要此信息，
                          或者用于日志记录和校验。

    Returns:
        tuple[paddle.nn.Layer, int]:
            - paddle.nn.Layer: 实例化的骨干网络模型。
            - int: 该骨干网络输出的特征向量的维度。

    Raises:
        ValueError: 如果 `model_type_str` 不是支持的骨干网络类型。
    """
    feature_dim_out = 0  # 初始化骨干网络输出的特征维度

    if model_type_str == 'vgg':
        # 从配置参数中获取VGG特有的参数
        dropout_rate = config_model_params.get('dropout_rate', 0.5) # VGG的dropout率，默认为0.5
        # 从配置中获取 feature_dim，如果未指定，则默认为512
        feature_dim_config = config_model_params.get('feature_dim', 512) 
        
        backbone = VGGBackbone(dropout_rate=dropout_rate, feature_out_dim=feature_dim_config) # 传递 feature_out_dim
        feature_dim_out = feature_dim_config # 返回配置的特征维度
        # NOTE: 当前VGGBackbone内部的feature_out_layer输出固定为512维。
        # 如果配置中的feature_dim与此不一致，需要关注实际生效的是哪个。
        # 理想情况下，VGGBackbone也应能根据feature_dim参数调整其最终输出层。
        # 此处暂时以外部配置的feature_dim_out为准返回给调用者，但模型本身的输出层可能仍是固定的。
        # 确保配置的feature_dim与VGGBackbone的实际输出能力匹配非常重要。
        # 更新：VGGBackbone 现已修改为接受 feature_out_dim 参数

    elif model_type_str == 'resnet':
        # 从配置参数中获取ResNet特有的参数
        nf = config_model_params.get('nf', 32)  # ResNet初始卷积层的输出通道数，默认为32
        n_blocks = config_model_params.get('n_resnet_blocks', 3) # 每个ResNet stage中残差块的数量，默认为3
        # ResNetFace的输出特征维度，这个值会传递给ResNetFace的构造函数
        feature_dim_out = config_model_params.get('feature_dim', 512) # 默认为512
        
        backbone = ResNetFace(nf=nf, n=n_blocks, feature_dim=feature_dim_out)

    elif model_type_str == 'resnet_cbam':
        nf = config_model_params.get('nf', 32)
        n_blocks = config_model_params.get('n_resnet_blocks', 3)
        feature_dim_out = config_model_params.get('feature_dim', 512)
        # 使用我们在 model_with_cbam_arcface.py 里定义的 ResNetFaceCBAM
        backbone = ResNetFaceCBAM(nf=nf, n=n_blocks, feature_dim=feature_dim_out)
    else:
        # 如果请求了不支持的骨干网络类型，则抛出异常
        raise ValueError(f"不支持的骨干网络类型: {model_type_str}。支持的类型为 'vgg' 或 'resnet'。")
    
    return backbone, feature_dim_out

def get_head(config_loss_params: dict, loss_type_str: str, in_features: int, num_classes: int) -> nn.Layer:
    """
    根据配置创建并返回一个模型头部模块实例。

    模型头部通常接收来自骨干网络的特征，并根据特定的损失函数需求（如ArcFace的边距计算，
    或CrossEntropy的线性分类）进行处理，并输出用于损失计算的logits或中间结果。

    Args:
        config_loss_params (dict): 包含特定损失函数/头部所需参数的字典。
                                   例如，对于ArcFace，可能包含 `arcface_m1`, `arcface_m2`, `arcface_s` 等。
                                   通常对应于配置文件中 `config.loss.arcface_params` 或类似路径。
        loss_type_str (str): 指定要创建的头部模块类型，通常与损失函数类型相关。
                             目前支持 'arcface' 和 'cross_entropy'。
        in_features (int): 头部模块期望的输入特征维度。该值必须与对应骨干网络的输出特征维度严格匹配。
        num_classes (int): 分类任务的目标类别总数。头部模块（特别是分类头）需要此信息来确定输出维度。

    Returns:
        paddle.nn.Layer: 实例化的模型头部模块。

    Raises:
        ValueError: 如果 `loss_type_str` 不是支持的头部/损失类型。
    """
    if loss_type_str == 'arcface':
        # 从配置参数中获取ArcFace特有的参数
        # 这些参数对应PaddlePaddle `margin_cross_entropy` API中的参数
        # m1: 对应 margin_cross_entropy 的 margin1，通常为1.0，用于 cos(m1*theta + m2) - m3 中的 m1
        # m2: 对应 margin_cross_entropy 的 margin2，即原始ArcFace论文中的角度边距 'm'
        # m3: 对应 margin_cross_entropy 的 margin3，通常为0.0
        # s:  对应 margin_cross_entropy 的 scale，即原始ArcFace论文中的尺度因子 's'
        m1 = config_loss_params.get('arcface_m1', 1.0)
        m2 = config_loss_params.get('arcface_m2', 0.5) # 角度间隔 margin
        m3 = config_loss_params.get('arcface_m3', 0.0)
        s  = config_loss_params.get('arcface_s', 64.0)  # 尺度因子 scale
        
        head = ArcFaceHead(
            in_features=in_features,    # 输入特征维度，来自骨干网络
            out_features=num_classes,   # 输出类别数
            margin1=m1, 
            margin2=m2, 
            margin3=m3, 
            scale=s
        )
    elif loss_type_str == 'cross_entropy':
        # CrossEntropyHead 通常只需要输入特征维度和类别数
        # 如果有特定参数，也可以从 config_loss_params 中获取
        # cross_entropy_specific_param = config_loss_params.get('some_param', default_value)
        head = CrossEntropyHead(in_features=in_features, out_features=num_classes)
    else:
        # 如果请求了不支持的头部类型，则抛出异常
        raise ValueError(f"不支持的头部/损失类型: {loss_type_str}。支持的类型为 'arcface' 或 'cross_entropy'。")
    
    return head 