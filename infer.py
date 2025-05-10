# infer.py
# 该脚本负责人脸识别的推理 (inference) 功能。
# 主要流程包括：
# 1. 加载配置文件和命令行参数。
# 2. 根据配置设置运行设备 (CPU/GPU)。
# 3. 加载预训练好的模型文件 (.pdparams)，这包括模型权重和训练时的配置信息。
# 4. 根据模型文件中保存的配置 (model_type, loss_type, image_size等) 和当前脚本的配置，
#    使用 model_factory.py 动态实例化骨干网络 (backbone) 和必要的头部模块 (head)。
#    - 对于CrossEntropy模型，会实例化分类头 (CrossEntropyHead)。
#    - 对于ArcFace模型，推理时通常只使用骨干网络提取特征，然后与特征库比对，不直接实例化ArcFaceHead进行前向计算。
# 5. 加载模型权重到实例化的网络中。
# 6. 加载类别标签映射文件 (readme.json)，用于将预测的ID转换为可读的名称。
# 7. 对于ArcFace模型，加载预先构建的人脸特征库 (.pkl)。
# 8. 对输入的单张图像进行预处理 (缩放、归一化、标准化)。
# 9. 执行推理：
#    - ArcFace模型: 提取输入图像的特征，然后与特征库中的每个已知身份的特征计算余弦相似度，
#                   找出最相似的身份，并根据阈值判断是否为已知人物。
#    - CrossEntropy模型: 将图像输入到完整的 模型(骨干+头) 中，获取分类的logits，
#                       通过softmax得到概率，取概率最高的类别作为预测结果。
# 10. 可视化推理结果：在输入图像上标注预测的类别名称和置信度/相似度，并保存到 results/ 目录。

import os
import cv2
import argparse
import numpy as np
import paddle
# import paddle.nn as nn # 已通过 heads.py 引入或不再直接需要
# from vgg import VGGFace         # 导入VGG模型 # 已移除
# from resnet_new import ResNetFace, ArcFaceHead # 导入新版ResNet模型和ArcFaceHead # 已移除
import json # 用于加载标签映射文件 (readme.json)
import matplotlib
matplotlib.use('Agg') # 切换到非交互式后端，防止在无GUI服务器上出错
import matplotlib.pyplot as plt
import pickle # 用于加载ArcFace模型所需的人脸特征库 (.pkl文件)
from config_utils import load_config, ConfigObject # 导入配置加载工具和配置对象类型
from model_factory import get_backbone, get_head # 导入模型构建的工厂函数

def process_image_local(img_path: str, target_size: int = 64, 
                        mean_rgb: list[float] = [0.485, 0.456, 0.406], 
                        std_rgb: list[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    对单张输入图像进行预处理，为模型推理做准备。

    处理步骤包括：
    1. 使用OpenCV加载图像。
    2. 将图像缩放到指定的目标尺寸 `target_size` (正方形)。
    3. 将图像从BGR颜色空间转换为RGB颜色空间。
    4. 将像素值从 [0, 255] 归一化到 [0, 1]范围。
    5. 进行标准化处理：(pixel - mean) / std，使用给定的RGB均值和标准差。
    6. 将图像数据格式从HWC（高x宽x通道）转换为CHW（通道x高x宽）。
    7. 在最前面增加一个批次维度 (batch_size=1)，最终形状为 (1, C, H, W)。

    Args:
        img_path (str): 输入图像的文件路径。
        target_size (int, optional): 图像将被缩放到的目标正方形尺寸 (高度和宽度相同)。
                                   默认为 64。
        mean_rgb (list[float], optional): RGB三通道的均值，用于标准化。长度应为3。
                                        默认为 [0.485, 0.456, 0.406] (ImageNet常用均值)。
        std_rgb (list[float], optional): RGB三通道的标准差，用于标准化。长度应为3。
                                       默认为 [0.229, 0.224, 0.225] (ImageNet常用标准差)。

    Returns:
        np.ndarray: 预处理后的图像数据，numpy数组，数据类型为 float32，
                    形状为 (1, 3, target_size, target_size)，可以直接作为模型输入。

    Raises:
        FileNotFoundError: 如果指定的 `img_path` 文件不存在或无法读取。
        Exception: 如果图像处理过程中发生其他错误。
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"错误: 无法读取图像文件 {img_path}。请检查路径或文件是否损坏。")
    
    # 缩放图像到目标尺寸
    img_resized = cv2.resize(img, (target_size, target_size))
    # BGR -> RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    # 归一化到 [0, 1]
    img_normalized_01 = img_rgb.astype('float32') / 255.0
    
    # 标准化 (减均值，除以标准差)
    # 将均值和标准差列表转换为numpy数组，并调整形状以匹配图像的通道维度
    mean_np = np.array(mean_rgb, dtype='float32').reshape((1, 1, 3))
    std_np = np.array(std_rgb, dtype='float32').reshape((1, 1, 3))
    img_standardized_hwc = (img_normalized_01 - mean_np) / std_np
    
    # HWC -> CHW (PaddlePaddle期望的输入格式)
    img_chw = img_standardized_hwc.transpose((2, 0, 1)) # (H,W,C) -> (C,H,W)
    
    # 增加批次维度 (batch_size=1)
    img_batch = np.expand_dims(img_chw, axis=0) # (C,H,W) -> (1,C,H,W)
    
    return img_batch.astype('float32') # 确保最终是float32类型

def compute_similarity(feature_vec1: np.ndarray, feature_vec2: np.ndarray) -> float:
    """计算两个一维特征向量之间的余弦相似度。

    余弦相似度衡量两个向量在方向上的相似程度，值域为 [-1, 1]。
    值越接近1，表示两个向量越相似。

    Args:
        feature_vec1 (np.ndarray): 第一个特征向量 (一维numpy数组)。
        feature_vec2 (np.ndarray): 第二个特征向量 (一维numpy数组)。

    Returns:
        float: 计算得到的余弦相似度。如果任一向量的范数为0，则返回0.0以避免除零错误。
    """
    # 确保输入是一维向量 (如果已经是，flatten操作无影响)
    f1 = feature_vec1.flatten()
    f2 = feature_vec2.flatten()
    
    # 计算各自的L2范数 (模长)
    norm_f1 = np.linalg.norm(f1)
    norm_f2 = np.linalg.norm(f2)
    
    # 防止除以零错误：如果任一向量的范数为0，则认为它们不相似 (相似度为0)
    if norm_f1 == 0 or norm_f2 == 0:
        return 0.0
        
    # 计算点积 / (范数之积)
    similarity = np.dot(f1, f2) / (norm_f1 * norm_f2)
    return float(similarity) # 确保返回的是标准的float类型

def infer(config: ConfigObject):
    """执行人脸识别推理的核心函数。

    根据提供的配置对象 `config` 进行单张图像的人脸识别。
    支持ArcFace模型 (与特征库比对) 和CrossEntropy模型 (直接分类)。

    Args:
        config (ConfigObject): 包含所有推理所需参数的配置对象。
                               关键参数包括 `model_path`, `image_path`, `use_gpu`，
                               以及针对ArcFace的 `infer.face_library_path`, `infer.recognition_threshold`，
                               和通用的 `infer.label_file`, `infer.infer_visualize`。
    """
    # --- 1. 设置运行设备 --- 
    if config.use_gpu and paddle.is_compiled_with_cuda():
        paddle.set_device('gpu')
        print("使用 GPU 进行推理")
    else:
        paddle.set_device('cpu')
        print("使用 CPU 进行推理")
    
    # --- 2. 检查并加载模型文件 --- 
    if not config.model_path or not os.path.exists(config.model_path):
        print(f"错误: 找不到指定的模型文件路径 '{config.model_path}' 或路径未配置。请通过 --model_path 或配置文件提供。")
        return

    print(f"从模型文件 {config.model_path} 加载模型参数和配置...")
    try:
        state_dict_container = paddle.load(config.model_path)
        if not isinstance(state_dict_container, dict):
            print(f"错误: 模型文件 {config.model_path} 内容格式不正确，期望为字典。")
            return
    except Exception as e:
        print(f"错误: 加载模型文件 {config.model_path} 失败: {e}")
        return

    # --- 3. 解析模型文件中保存的训练时配置 --- 
    # 模型文件通常会保存训练时的配置，如模型类型、损失类型、图像大小等，用于正确恢复模型。
    saved_model_config_dict = state_dict_container.get('config') # 新版检查点格式
    if not saved_model_config_dict:
        saved_model_config_dict = state_dict_container.get('args') # 兼容旧版检查点格式 (argparse.Namespace)
        if saved_model_config_dict and not isinstance(saved_model_config_dict, dict):
            saved_model_config_dict = vars(saved_model_config_dict) # 将Namespace转为字典
    
    if not saved_model_config_dict or not isinstance(saved_model_config_dict, dict):
        print(f"错误: 模型文件 {config.model_path} 中缺少有效的训练配置信息 (未找到 'config' 或 'args' 键，或其值非字典)。")
        print(f"  这对于正确实例化模型至关重要。请确保模型文件是本项目训练脚本生成的。")
        return
    
    # 从保存的配置中获取模型类型、损失类型和类别数
    # 提供默认值以处理旧模型或配置不全的情况，但最好是配置完整
    model_type_from_saved_config = saved_model_config_dict.get('model_type', 'resnet') # 默认尝试resnet
    loss_type_from_saved_config = saved_model_config_dict.get('loss_type', 'arcface') # 默认尝试arcface
    num_classes_from_saved_config = saved_model_config_dict.get('num_classes')

    # 确定推理时使用的图像尺寸：优先使用模型训练时的尺寸。
    # 如果模型文件未保存，则尝试使用当前脚本配置中的，否则报错。
    image_size_from_model_file = saved_model_config_dict.get('image_size')
    effective_image_size = image_size_from_model_file # 优先使用模型自带的
    
    if hasattr(config, 'image_size') and config.image_size is not None: # 如果当前配置也指定了image_size
        if effective_image_size is not None and config.image_size != effective_image_size:
            print(f"警告: 当前配置的图像大小 ({config.image_size}) 与模型训练时的大小 ({effective_image_size}) 不一致。")
            print(f"       将优先使用模型训练时的图像大小: {effective_image_size}")
        elif effective_image_size is None:
            effective_image_size = config.image_size # 模型没存，当前配置有，则用当前的
            print(f"提示: 模型文件未记录图像大小，将使用当前配置的图像大小: {effective_image_size}")
            
    if effective_image_size is None: # 两边都没有，则无法继续
        print(f"错误: 无法确定推理时应使用的图像大小。模型配置和当前脚本配置中均未提供 'image_size'。")
        return
    print(f"推理时将使用图像大小: {effective_image_size}x{effective_image_size}")

    # --- 4. 实例化骨干网络和头部模块 (使用工厂函数) --- 
    # 获取骨干网络构建所需的参数，优先从模型文件中保存的配置里取
    backbone_params_from_model_file = saved_model_config_dict.get('backbone_params', {}) 
    # 兼容旧格式: model: { resnet_params: {...} }
    if not backbone_params_from_model_file:
        legacy_model_section = saved_model_config_dict.get('model', {})
        if model_type_from_saved_config == 'vgg' and 'vgg_params' in legacy_model_section:
            backbone_params_from_model_file = legacy_model_section['vgg_params']
        elif model_type_from_saved_config == 'resnet' and 'resnet_params' in legacy_model_section:
            backbone_params_from_model_file = legacy_model_section['resnet_params']
    
    if not backbone_params_from_model_file:
        print(f"警告: 未在模型配置 '{config.model_path}' 中找到 '{model_type_from_saved_config}_params' 或兼容的旧格式参数。")
        print(f"       骨干网络 ({model_type_from_saved_config.upper()}) 将尝试使用默认参数实例化，这可能导致错误或性能下降。")

    backbone_instance, feature_dim_from_backbone = get_backbone(
        config_model_params=backbone_params_from_model_file,
        model_type_str=model_type_from_saved_config,
        image_size=effective_image_size # 传递最终确定的图像尺寸
    )
    if not backbone_instance:
        print(f"错误: 实例化骨干网络 ({model_type_from_saved_config.upper()}) 失败。请检查模型类型和参数配置。")
        return
    print(f"骨干网络 ({model_type_from_saved_config.upper()}) 实例化成功，声明输出特征维度: {feature_dim_from_backbone}")

    head_module_instance = None # 初始化为None
    # 仅当损失类型为CrossEntropy时，才需要在推理时显式实例化和加载分类头。
    # ArcFace模型的推理依赖于骨干网络提取特征后与特征库进行比对。
    if loss_type_from_saved_config == 'cross_entropy':
        if num_classes_from_saved_config is None:
            print(f"错误: 模型配置中缺少 'num_classes'，无法为CrossEntropyLoss实例化头部模块。")
            return
        
        # 获取头部构建参数，优先从模型文件中保存的配置里取
        head_params_from_model_file = saved_model_config_dict.get('head_params', {})
        # 兼容旧格式: loss: { cross_entropy_params: {...} } (通常CE头参数为空)
        if not head_params_from_model_file:
            legacy_loss_section = saved_model_config_dict.get('loss', {})
            if 'cross_entropy_params' in legacy_loss_section:
                 head_params_from_model_file = legacy_loss_section['cross_entropy_params']
        
        head_module_instance = get_head(
            config_loss_params=head_params_from_model_file,
            loss_type_str=loss_type_from_saved_config,
            in_features=feature_dim_from_backbone, # 头部输入维度需与骨干输出匹配
            num_classes=num_classes_from_saved_config
        )
        if not head_module_instance:
            print(f"错误: 实例化头部模块 ({loss_type_from_saved_config.upper()}) 失败。")
            return
        print(f"头部模块 ({loss_type_from_saved_config.upper()}) 实例化成功。")
    elif loss_type_from_saved_config == 'arcface':
        print(f"提示: 模型损失类型为 '{loss_type_from_saved_config.upper()}'。推理时将仅使用骨干网络提取特征，并与特征库比对。")
        print(f"       不直接实例化 {loss_type_from_saved_config.upper()}Head 进行前向计算。")

    # --- 5. 加载模型权重 --- 
    weights_loaded_successfully = True
    # 加载骨干网络权重
    if 'backbone' in state_dict_container and backbone_instance:
        try:
            backbone_instance.set_state_dict(state_dict_container['backbone'])
            print(f"{model_type_from_saved_config.upper()} 骨干网络权重从 'backbone' 键加载成功。")
        except Exception as e:
            print(f"错误: 加载骨干网络权重失败: {e}")
            weights_loaded_successfully = False
    # 兼容旧VGG模型文件，其骨干权重可能存储在 'model' 键下
    elif model_type_from_saved_config == 'vgg' and 'model' in state_dict_container and backbone_instance:
        try:
            backbone_instance.set_state_dict(state_dict_container['model'])
            print(f"VGG 骨干网络权重从旧的 'model' 键加载成功。")
        except Exception as e:
            print(f"错误: 加载VGG骨干网络权重 (从'model'键) 失败: {e}")
            weights_loaded_successfully = False
    elif backbone_instance: # 骨干网络已实例化，但模型文件中无对应权重
        print(f"错误: 在模型文件 {config.model_path} 中未找到 'backbone' (或兼容的 'model' for VGG) 的骨干网络权重。")
        weights_loaded_successfully = False

    # 如果是CE模型且头部已实例化，则加载头部权重
    if head_module_instance: 
        if 'head' in state_dict_container:
            try:
                head_module_instance.set_state_dict(state_dict_container['head'])
                print(f"头部模块 ({loss_type_from_saved_config.upper()}) 权重加载成功。")
            except Exception as e:
                print(f"错误: 加载头部模块 ({loss_type_from_saved_config.upper()}) 权重失败: {e}")
                weights_loaded_successfully = False
        else:
            # 对于推理，如果CE head权重未找到，分类结果将不可靠。但特征提取可能仍有用。
            print(f"警告: 在模型文件 {config.model_path} 中未找到 'head' 的头部模块权重。")
            print(f"       CrossEntropyHead ({loss_type_from_saved_config.upper()}) 将使用其初始权重，这可能导致错误的分类结果。")
            # 视情况决定是否将 weights_loaded_successfully 设为 False。如果CE模型必须有头权重，则应设为False。
            # 此处不设为False，但用户应非常注意此警告。

    if not weights_loaded_successfully:
        print("由于部分或全部模型权重加载失败，无法继续进行可靠的推理。请检查模型文件和配置。")
        return
        
    # --- 6. 设置模型为评估模式 --- 
    # 这会关闭Dropout层，并使BatchNorm层使用学习到的均值和方差，而不是当前批次的统计数据。
    if backbone_instance: backbone_instance.eval()
    if head_module_instance: head_module_instance.eval()

    # --- 7. 加载推理辅助文件 (标签映射、特征库) ---
    # 从当前脚本的config对象中获取推理特定的配置 (通常在 infer: {...} 块下)
    infer_specific_config = config.get('infer', {})
    label_filename_from_config = infer_specific_config.get('label_file') # 现在期望 "readme.json"
    face_library_file_path = infer_specific_config.get('face_library_path') # ArcFace模型所需
    recognition_similarity_threshold = infer_specific_config.get('recognition_threshold', 0.5) # ArcFace识别阈值
    should_visualize_output = infer_specific_config.get('infer_visualize', False) # 是否可视化结果

    label_file_path = None
    if label_filename_from_config: # 检查是否提供了文件名
        # 构建完整路径
        label_file_path = os.path.join(config.data_dir, config.class_name, label_filename_from_config)

    # 加载标签ID到名称的映射 (从 CreateDataList.py 生成的 readme.json)
    label_id_to_name_map = {}
    if not label_file_path or not os.path.exists(label_file_path):
        actual_path_for_warning = label_file_path if label_file_path else f"(config.infer.label_file: '{label_filename_from_config}', config.data_dir: '{config.data_dir}', config.class_name: '{config.class_name}')"
        print(f"警告: 未找到或无法访问标签文件。预期路径: '{actual_path_for_warning}'.")
        print(f"       确保 config.infer.label_file (当前为 \"{label_filename_from_config}\") 已正确设置，并且文件存在于 {os.path.join(config.data_dir, config.class_name)} 目录下。")
        print(f"       推理结果中将只显示类别ID，而不是具体的名称。")
    else:
        try:
            with open(label_file_path, 'r', encoding='utf-8') as f_json:
                label_metadata = json.load(f_json)
                # readme.json 的结构通常是: { ..., "class_detail": [{"class_label":0, "class_name":"Alice"}, ...] }
                for class_entry in label_metadata.get('class_detail', []):
                    label_id = class_entry.get('class_label') 
                    class_name = class_entry.get('class_name')
                    if label_id is not None and class_name is not None:
                         label_id_to_name_map[int(label_id)] = str(class_name) #确保ID是整数，name是字符串
            if not label_id_to_name_map:
                print(f"警告: 标签文件 {label_file_path} 解析后未得到有效的标签ID到名称的映射。请检查文件内容和格式。")
            else:
                print(f"标签文件 {label_file_path} 加载成功，映射了 {len(label_id_to_name_map)} 个类别。")
        except Exception as e:
            print(f"错误: 读取或解析标签文件 {label_file_path} 失败: {e}")
            # 即使标签文件加载失败，仍可继续推理，只是无法显示名称。

    # 如果是ArcFace模型，则加载人脸特征库
    face_feature_library = None
    is_arcface_like_model = (loss_type_from_saved_config == 'arcface')
    
    if is_arcface_like_model:
        if not face_library_file_path or not os.path.exists(face_library_file_path):
            print(f"错误: ArcFace类模型 ({model_type_from_saved_config.upper()}+{loss_type_from_saved_config.upper()}) 推理需要人脸特征库。")
            print(f"       请通过配置文件中的 infer.face_library_path 提供有效的特征库文件路径 (当前: '{face_library_file_path}')。")
            print(f"       特征库可使用 create_face_library.py脚本生成。")
            return
        try:
            with open(face_library_file_path, 'rb') as f_lib_pickle:
                face_feature_library = pickle.load(f_lib_pickle)
            if not isinstance(face_feature_library, dict) or not face_feature_library:
                print(f"错误: 人脸特征库文件 {face_library_file_path} 加载成功，但其内容不是预期的非空字典格式。请检查特征库文件。")
                return
            print(f"人脸特征库 {face_library_file_path} 加载成功，包含 {len(face_feature_library)} 个已知身份的特征。")
        except Exception as e:
            print(f"错误: 加载人脸特征库 {face_library_file_path} 失败: {e}")
            return

    # --- 8. 预处理输入图像 ---
    if not config.image_path or not os.path.exists(config.image_path):
        print(f"错误: 未提供有效的待识别图像路径 (当前配置 image_path: '{config.image_path}') 或文件不存在。")
        return
    try:
        # 使用之前确定的 effective_image_size 进行预处理
        # 均值和标准差参数可以从config中读取，如果项目有特定值的话。此处使用ImageNet默认值。
        # mean_rgb_config = config.dataset_params.get('mean', [0.485, 0.456, 0.406])
        # std_rgb_config = config.dataset_params.get('std', [0.229, 0.224, 0.225])
        preprocessed_image_np = process_image_local(
            img_path=config.image_path, 
            target_size=effective_image_size
            # mean_rgb=mean_rgb_config, # 可选，如果需要自定义
            # std_rgb=std_rgb_config    # 可选
        )
    except FileNotFoundError as e_file:
        print(e_file); return
    except Exception as e_proc:
        print(f"处理输入图像 {config.image_path} 时发生错误: {e_proc}"); return
        
    # 将预处理后的numpy数组转换为Paddle张量
    input_image_tensor = paddle.to_tensor(preprocessed_image_np)
    
    # --- 9. 执行模型推理 --- 
    predicted_label_id = -1 # 默认为未知或错误
    prediction_score = 0.0  # 对于CE是最高置信度，对于ArcFace是最高相似度
    final_display_text_for_viz = "未知人物" # 用于可视化时显示的文本

    print(f"\n开始对图像 '{config.image_path}' 执行推理，使用模型: '{config.model_path}' ({model_type_from_saved_config.upper()}+{loss_type_from_saved_config.upper()})...")
    with paddle.no_grad(): # 推理时关闭梯度计算
        # 9a. 提取输入图像的特征 (所有模型都需要这一步)
        if not backbone_instance:
            print("错误: 骨干网络未成功实例化或加载权重，无法进行特征提取。")
            return
        input_image_features_tensor = backbone_instance(input_image_tensor)
        
        # 9b. 根据模型类型进行后续的识别处理
        if is_arcface_like_model: # ArcFace模型 (如 ResNet+ArcFace)
            if face_feature_library is None: 
                print("错误：ArcFace模型推理需要人脸特征库，但特征库未能成功加载。")
                return 
            
            # 将提取到的特征张量转换为numpy数组，并展平为一维向量
            input_image_features_np = input_image_features_tensor.numpy().flatten()
            
            best_match_label_from_lib = -1 # 初始化最佳匹配标签ID
            highest_similarity_score = -1.0 # 初始化最高相似度

            # 遍历特征库中的每一个已知身份及其平均特征向量
            for known_label_id, known_feature_vector in face_feature_library.items():
                current_similarity = compute_similarity(input_image_features_np, known_feature_vector)
                if current_similarity > highest_similarity_score:
                    highest_similarity_score = current_similarity
                    best_match_label_from_lib = known_label_id # 更新为当前最相似的已知ID
            
            predicted_label_id = best_match_label_from_lib 
            prediction_score = highest_similarity_score
            print(f"特征比对完成。输入图像与特征库中各身份的最高相似度为: {prediction_score:.4f}")

            # 结果解释 (ArcFace)
            if predicted_label_id == -1 or prediction_score < recognition_similarity_threshold: 
                predicted_label_id = -1 # 明确标记为未知
                recognized_person_name = "未知人物"
                final_display_text_for_viz = f"{recognized_person_name} (相似度 {prediction_score:.4f})"
                print(f"  识别结果: {recognized_person_name}。最高相似度 {prediction_score:.4f} 低于阈值 {recognition_similarity_threshold} 或未匹配到任何库中身份。")
            else:
                recognized_person_name = label_id_to_name_map.get(predicted_label_id, f"标签ID_{predicted_label_id}")
                final_display_text_for_viz = f"{recognized_person_name} (相似度 {prediction_score:.4f})"
                print(f"  预测身份: {recognized_person_name} (标签ID: {predicted_label_id}), 最高相似度: {prediction_score:.4f}")

        else: # CrossEntropy模型 (如 ResNet+CrossEntropy, VGG+CrossEntropy)
            if head_module_instance is None:
                print(f"错误: CrossEntropy模型 ({model_type_from_saved_config.upper()}+{loss_type_from_saved_config.upper()}) 推理需要分类头 (head_module)，但该模块未成功加载或初始化。")
                return
            
            # CrossEntropyHead的forward方法在 label=None 时应返回 (None, logits)
            _, output_logits = head_module_instance(input_image_features_tensor, label=None) 
            
            if output_logits is None:
                print(f"错误: 从 {loss_type_from_saved_config.upper()} 头部获取分类 Logits 失败。")
                return

            # 对Logits应用Softmax得到概率分布，然后取概率最高的类别及其概率值
            output_probabilities_tensor = paddle.nn.functional.softmax(output_logits, axis=1)
            output_probabilities_np = output_probabilities_tensor.numpy()
            
            predicted_label_id = np.argmax(output_probabilities_np[0])  # 取第一个样本 (batch_size=1) 的最高概率索引
            prediction_score = output_probabilities_np[0][predicted_label_id] # 最高概率值 (置信度)
            print(f"分类计算完成。")
            
            # 结果解释 (CrossEntropy)
            recognized_person_name = label_id_to_name_map.get(predicted_label_id, f"标签ID_{predicted_label_id}")
            final_display_text_for_viz = f"{recognized_person_name} (置信度 {prediction_score:.4f})"
            print(f"  预测身份: {recognized_person_name} (标签ID: {predicted_label_id}), 置信度: {prediction_score:.4f}")
    
    # 检查预测的ID是否在标签映射中，如果不在，给出提示
    if predicted_label_id != -1 and predicted_label_id not in label_id_to_name_map and label_file_path and os.path.exists(label_file_path):
        print(f"提示: 预测的类别ID '{predicted_label_id}' 在标签文件 '{label_file_path}' 中找不到对应的名称。")
    
    # --- 10. 可视化结果 (如果配置启用) --- 
    if should_visualize_output:
        print(f"正在生成可视化结果图像...")
        try:
            original_image_for_display = cv2.imread(config.image_path)
            if original_image_for_display is None: 
                print(f"警告: 无法重新读取图像 {config.image_path} 进行可视化。跳过可视化。")
                return 
            
            # 将OpenCV的BGR图像转换为RGB，以便matplotlib正确显示颜色
            image_rgb_for_display = cv2.cvtColor(original_image_for_display, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(8, 6)) # 设置图像大小
            plt.imshow(image_rgb_for_display)
            plt.title(final_display_text_for_viz, fontsize=12) # 在图像上方显示最终的识别文本
            plt.axis('off') # 关闭坐标轴显示
            
            # 创建结果保存目录 (如果不存在)
            results_output_dir = config.get('results_dir', "results") # 可从配置获取，默认为 "results"
            if not os.path.exists(results_output_dir): 
                os.makedirs(results_output_dir)
                print(f"已创建推理结果保存目录: {results_output_dir}")
            
            # 构建结果图像的文件名
            original_image_basename = os.path.basename(config.image_path)
            name_part, ext_part = os.path.splitext(original_image_basename)
            result_image_filename = f"recognition_result_for_{name_part}_{model_type_from_saved_config}_{loss_type_from_saved_config}{ext_part if ext_part else '.png'}"
            result_image_full_path = os.path.join(results_output_dir, result_image_filename)
            
            plt.savefig(result_image_full_path)
            plt.close() # 关闭图像，释放资源
            print(f"推理结果的可视化图像已保存至: {result_image_full_path}")
        except ImportError:
            print("警告: Matplotlib 未能正确导入或配置，无法显示或保存可视化结果图像。请确保已安装并配置好 Matplotlib。")
        except Exception as e_viz:
            print(f"可视化过程中发生错误: {e_viz}。结果图像可能未保存或不完整。")
    else:
        print("可视化输出未启用 (infer_visualize 未设置为 true 或配置中缺失)。")

if __name__ == '__main__':
    # --- 命令行参数解析 --- 
    parser = argparse.ArgumentParser(description='人脸识别单图推理脚本')
    
    # 核心控制参数
    parser.add_argument('--config_path', type=str, default=None, 
                        help='指定YAML配置文件的路径。如果未提供，则使用脚本内部定义的默认路径。')
    parser.add_argument('--image_path', type=str, # required=True, 但由config_utils处理，允许从YAML加载
                        help='待识别的单张输入图像路径 (必需，除非在YAML中指定)。')
    parser.add_argument('--model_path', type=str, # required=True, 但由config_utils处理
                        help='训练好的模型文件路径 (.pdparams) (必需，除非在YAML中指定)。') 
    
    # 其他可覆盖配置文件的参数
    parser.add_argument('--use_gpu', action=argparse.BooleanOptionalAction, default=None,
                        help='是否使用GPU进行推理。此命令行开关会覆盖配置文件中的 global_settings.use_gpu 设置。')
    parser.add_argument('--image_size', type=int, default=None,
                        help='输入图像预处理后的统一大小。此命令行参数会覆盖配置文件或模型自带的image_size设置。')
    
    # 推理特定参数 (覆盖配置文件中 infer: {...} 下的同名项)
    parser.add_argument('--label_file', type=str, default=None,
                        help='类别标签ID到名称映射的JSON文件路径 (覆盖 infer.label_file)。')
    parser.add_argument('--face_library_path', type=str, default=None,
                        help='ArcFace模型所需的人脸特征库文件路径 (.pkl) (覆盖 infer.face_library_path)。')
    parser.add_argument('--recognition_threshold', type=float, default=None,
                        help='ArcFace模型识别时的相似度阈值 (覆盖 infer.recognition_threshold)。')
    parser.add_argument('--infer_visualize', action=argparse.BooleanOptionalAction, default=None,
                        help='是否可视化识别结果并保存图像。此命令行开关会覆盖配置文件中的 infer.infer_visualize 设置。')

    cmd_line_args = parser.parse_args()

    # --- 配置加载与合并 --- 
    # 使用 config_utils.load_config 函数加载和合并配置。
    final_config = load_config(
        default_yaml_path='configs/default_config.yaml', 
        cmd_args_namespace=cmd_line_args
    )

    # 检查关键路径是否已配置 (model_path 和 image_path)
    if not final_config.model_path:
        parser.error("错误: 缺少模型文件路径。请通过 --model_path 命令行参数或在YAML配置文件中提供 model_path。")
    if not final_config.image_path:
        parser.error("错误: 缺少待识别图像路径。请通过 --image_path 命令行参数或在YAML配置文件中提供 image_path。")

    # 尝试设置matplotlib中文字体，以便在可视化结果中正确显示中文名称
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 使用 SimHei 字体 (黑体)
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
    except Exception as e_font:
        print(f"提示: 设置matplotlib中文字体SimHei失败: {e_font}。可视化结果中的中文可能显示为乱码。")
        print(f"       请确保系统中安装了SimHei字体，或者在代码中指定其他可用的中文字体。")

    # 执行推理
    infer(final_config)