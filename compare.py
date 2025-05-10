# coding:utf-8
# compare.py
# 该脚本负责人脸对比 (face comparison) 功能。
# 它接收两张人脸图像作为输入，使用预训练的人脸识别模型分别提取这两张图像的特征向量，
# 然后计算这两个特征向量之间的余弦相似度，以此来判断这两张图像是否属于同一个人。
#
# 主要流程包括：
# 1. 加载配置文件和命令行参数，用于获取模型路径、图像路径、运行设备等设置。
# 2. 根据配置设置运行设备 (CPU/GPU)。
# 3. 加载预训练好的模型文件 (.pdparams)。这个模型文件通常只包含骨干网络 (backbone) 的权重，
#    因为人脸对比主要依赖于高质量的特征表示，而不需要分类头。
#    模型文件也应包含训练时的配置信息 (如 model_type, image_size)，以便正确恢复骨干网络。
# 4. 根据模型文件中保存的配置和当前脚本的配置，使用 model_factory.py 动态实例化骨干网络。
# 5. 加载骨干网络权重到实例化的网络中。
# 6. 对输入的两张图像分别进行预处理 (缩放、归一化、标准化)。
# 7. 使用加载的骨干网络分别提取两张图像的人脸特征向量。
# 8. 计算两个特征向量之间的余弦相似度。
# 9. 根据预设的相似度阈值，判断两张图像是否属于同一个人。
# 10. 打印对比结果，包括相似度得分和判断结论。
# 11. （可选）如果配置了可视化，则将两张图像并排显示，并在标题中注明相似度和判断结果，
#      然后将结果图像保存到 results/ 目录。
#
# 注意: 此脚本中的 process_image_local 和 compute_similarity_fc 函数与 infer.py 中的类似函数功能相同，
#       理想情况下，这些通用工具函数可以移至一个共享的 utils.py 模块中以避免代码重复。
#       但为了脚本的独立性，暂时在此处保留了本地副本。

import os
import cv2
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，防止在无显示环境的服务器上出错
import matplotlib.pyplot as plt
import paddle
# from vgg import VGGFace         # 导入VGG模型 # 已移除
# from resnet_new import ResNetFace, ArcFaceHead # 导入新版ResNet模型和ArcFaceHead # 已移除
# from infer import process_image # Removed to avoid circular dependency, use local or utils
from config_utils import load_config, ConfigObject # 导入配置加载工具和配置对象类型
from model_factory import get_backbone # 只需要骨干网络，头部不用于特征提取

# 全局变量，用于在 compare_faces 和 show_comparison_result 之间传递模型训练时的配置信息，
# 以便在保存结果图像时，文件名能反映所用模型的类型。
# 更好的做法可能是将这些信息作为参数传递，或通过一个类来封装状态。
saved_model_config_dict_for_filename: dict = {}

def process_image_local(img_path: str, target_size: int = 64, 
                        mean_rgb: list[float] = [0.485, 0.456, 0.406], 
                        std_rgb: list[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    对单张输入图像进行预处理，为模型提取特征做准备。

    处理步骤同 infer.py 中的 process_image_local 函数：
    1. 加载图像。
    2. 缩放到 `target_size`。
    3. BGR 转 RGB。
    4. 归一化到 [0, 1]。
    5. 标准化 (减均值，除以标准差)。
    6. HWC 转 CHW。
    7. 增加批次维度 (batch_size=1)。

    Args:
        img_path (str): 输入图像的文件路径。
        target_size (int, optional): 图像将被缩放到的目标正方形尺寸。默认为 64。
        mean_rgb (list[float], optional): RGB三通道的均值。默认为 ImageNet 常用均值。
        std_rgb (list[float], optional): RGB三通道的标准差。默认为 ImageNet 常用标准差。

    Returns:
        np.ndarray: 预处理后的图像数据 (1, 3, target_size, target_size)，float32类型。

    Raises:
        FileNotFoundError: 如果图像文件无法读取。
    """
    img = cv2.imread(img_path)
    if img is None: raise FileNotFoundError(f"错误: 无法读取图像文件 {img_path}")
    img = cv2.resize(img, (target_size, target_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    mean = np.array(mean_rgb, dtype='float32').reshape((1, 1, 3))
    std = np.array(std_rgb, dtype='float32').reshape((1, 1, 3))
    img_normalized = (img - mean) / std # 修正：这里img已经是HWC，mean/std也应是HWC兼容的
    img_chw = img_normalized.transpose((2, 0, 1))
    img_expanded = np.expand_dims(img_chw, axis=0)
    return img_expanded.astype('float32')

def compute_similarity_fc(feature1: np.ndarray, feature2: np.ndarray) -> float: 
    """
    计算两个一维特征向量之间的余弦相似度。
    
    Args:
        feature1 (numpy.ndarray): 第一个特征向量 (应为一维或可展平为一维)。
        feature2 (numpy.ndarray): 第二个特征向量 (应为一维或可展平为一维)。
    
    Returns:
        float: 余弦相似度得分。范围通常在-1到1之间。
               如果任一向量为零向量 (范数为0)，则返回0.0以避免除零错误。
    """
    #展平向量以确保它们是一维的
    f1 = feature1.flatten()
    f2 = feature2.flatten()
    
    # 计算余弦相似度: dot_product / (norm(f1) * norm(f2))
    norm_f1 = np.linalg.norm(f1)
    norm_f2 = np.linalg.norm(f2)
    if norm_f1 == 0 or norm_f2 == 0:
        return 0.0 # 如果任一向量为零向量，相似度为0
    similarity = np.dot(f1, f2) / (norm_f1 * norm_f2)
    return float(similarity)

def show_comparison_result(img1_path: str, img2_path: str, similarity: float, 
                           model_type_loaded: str, loss_type_loaded_from_model_cfg: str, 
                           threshold: float, visualize_flag: bool):
    """可视化人脸对比结果并保存图像。

    如果 `visualize_flag` 为 True，则执行以下操作：
    1. 读取两张原始图像。
    2. 使用 Matplotlib 将两张图像并排显示。
    3. 在图像上方居中显示标题，包含计算得到的相似度、判断阈值以及是/否为同一人的结论。
    4. 根据判断结果（是否为同一人）设置标题颜色（绿色表示同一人，红色表示不同人）。
    5. 将结果图像保存到 `results/` 目录下。
       保存的文件名会包含模型类型 (如 resnet, vgg)、模型训练时使用的损失类型 (如 arcface, cross_entropy)、
       以及两张输入图像的文件名，以便区分和追溯。

    Args:
        img1_path (str): 第一张原始图像的文件路径。
        img2_path (str): 第二张原始图像的文件路径。
        similarity (float): 计算得到的两张图像特征的余弦相似度。
        model_type_loaded (str): 加载的模型的骨干网络类型 (如 'resnet', 'vgg')。
                                  用于生成结果图像的文件名。
        loss_type_loaded_from_model_cfg (str): 从模型配置文件中解析出的训练时使用的损失类型
                                            (如 'arcface', 'cross_entropy')。用于文件名。
        threshold (float): 判断两张图像是否为同一人的相似度阈值。
        visualize_flag (bool): 控制是否执行可视化并保存图像的标志。如果为 False，则此函数不执行任何操作。
    """
    if not visualize_flag: return

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print(f"错误: 无法读取图像进行对比可视化 (图像1: {img1_path}, 图像2: {img2_path})。跳过可视化。")
        return

    # OpenCV 读取的是 BGR 格式，Matplotlib 显示需要 RGB 格式
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 6)) # 创建一个图窗，设置大小
    plt.subplot(1, 2, 1); plt.imshow(img1_rgb); plt.title("图像1", fontproperties="SimHei"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(img2_rgb); plt.title("图像2", fontproperties="SimHei"); plt.axis('off')
    
    is_same_person = similarity >= threshold
    judgment_text = "是同一个人" if is_same_person else "不是同一个人"
    title_color = 'green' if is_same_person else 'red' # 结果为真时绿色，否则红色
    # 准备图窗的超级标题文本
    result_text = f"相似度: {similarity:.4f}\n判断: {judgment_text} (阈值: {threshold:.2f})"
    plt.suptitle(result_text, fontsize=16, color=title_color, fontproperties="SimHei")
    
    # 准备结果保存目录和文件名
    result_dir = config.get('results_dir', "results") # 尝试从主配置获取，否则用默认
    if not os.path.exists(result_dir): 
        os.makedirs(result_dir)
        print(f"已创建对比结果保存目录: {result_dir}")
    
    base1 = os.path.basename(img1_path).split('.')[0]
    base2 = os.path.basename(img2_path).split('.')[0]
    
    # 使用从全局变量 saved_model_config_dict_for_filename 获取的模型配置中的损失类型来命名
    # 这里的 loss_type_loaded_from_model_cfg 已经由 compare_faces 函数正确传递
    active_loss_type_str = loss_type_loaded_from_model_cfg 
    # 构建文件名后缀，包含模型类型和损失类型
    model_suffix = f"{model_type_loaded}_{active_loss_type_str}"

    result_filename = f"compare_{model_suffix}_{base1}_vs_{base2}.png"
    result_file_path = os.path.join(result_dir, result_filename)
    
    try:
        plt.savefig(result_file_path)
        print(f"对比结果图像已保存至: {result_file_path}")
    except Exception as e_save:
        print(f"错误: 保存对比结果图像到 {result_file_path} 失败: {e_save}")
    finally:
        plt.close() # 关闭图窗，释放资源

def compare_faces(config: ConfigObject):
    """人脸对比主函数。

    根据提供的配置对象 `config`，执行两张人脸图像的对比流程。

    Args:
        config (ConfigObject): 包含所有对比所需参数的配置对象。
                               关键参数包括 `model_path` (模型路径), `img1` (第一张图像路径),
                               `img2` (第二张图像路径), `use_gpu` (是否使用GPU)。
                               此外，`compare.compare_threshold` 和 `compare.compare_visualize` 
                               分别控制相似度阈值和是否可视化结果，这些也应在 `config` 中。
    """
    # --- 1. 检查输入图像是否存在 --- 
    if not config.img1 or not os.path.exists(config.img1): 
        print(f"错误: 第一张待对比图像路径 '{config.img1}' 未提供或文件不存在。请通过 --img1 或配置文件提供。")
        return
    if not config.img2 or not os.path.exists(config.img2): 
        print(f"错误: 第二张待对比图像路径 '{config.img2}' 未提供或文件不存在。请通过 --img2 或配置文件提供。")
        return
    
    # --- 2. 设置运行设备 --- 
    if config.use_gpu and paddle.is_compiled_with_cuda():
        paddle.set_device('gpu'); print("使用 GPU 进行人脸特征提取与对比")
    else:
        paddle.set_device('cpu'); print("使用 CPU 进行人脸特征提取与对比")

    # --- 3. 检查并加载模型文件 --- 
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

    # --- 4. 解析模型文件中保存的训练时配置 --- 
    # 全局变量 saved_model_config_dict_for_filename 用于传递给可视化函数以正确命名输出文件
    global saved_model_config_dict_for_filename 
    saved_model_config_dict_for_filename = state_dict_container.get('config') # 新版检查点格式
    if not saved_model_config_dict_for_filename:
        saved_model_config_dict_for_filename = state_dict_container.get('args') # 兼容旧版检查点格式
        if saved_model_config_dict_for_filename and not isinstance(saved_model_config_dict_for_filename, dict):
            saved_model_config_dict_for_filename = vars(saved_model_config_dict_for_filename) # Namespace转字典
    
    if not saved_model_config_dict_for_filename or not isinstance(saved_model_config_dict_for_filename, dict):
        print(f"错误: 模型文件 {config.model_path} 中缺少有效的训练配置信息。这对于正确实例化骨干网络至关重要。")
        return

    # 从保存的配置中获取模型类型和损失类型 (损失类型主要用于结果文件名中标记模型来源)
    model_type_from_saved_config = saved_model_config_dict_for_filename.get('model_type', 'resnet') # 默认resnet
    loss_type_from_saved_config = saved_model_config_dict_for_filename.get('loss_type', 'unknown_loss') # 如'arcface', 'cross_entropy'

    # 确定图像尺寸：优先使用模型训练时的尺寸
    image_size_from_model_file = saved_model_config_dict_for_filename.get('image_size')
    effective_image_size = image_size_from_model_file
    
    if hasattr(config, 'image_size') and config.image_size is not None:
        if effective_image_size is not None and config.image_size != effective_image_size:
             print(f"警告: 当前配置的图像大小 ({config.image_size}) 与模型训练时的大小 ({effective_image_size}) 不一致。将优先使用模型训练时的大小: {effective_image_size}")
        elif effective_image_size is None:
            effective_image_size = config.image_size
            print(f"提示: 模型文件未记录图像大小，将使用当前配置的图像大小: {effective_image_size}")
            
    if effective_image_size is None:
        print(f"错误: 无法确定图像处理尺寸。模型配置和当前脚本配置中均未提供 'image_size'。")
        return
    print(f"图像将处理为: {effective_image_size}x{effective_image_size}")

    # --- 5. 实例化骨干网络 (使用工厂函数) --- 
    # 获取骨干网络参数，优先从模型文件中保存的配置里取
    backbone_params_from_model_file = saved_model_config_dict_for_filename.get('backbone_params', {})
    if not backbone_params_from_model_file:
        legacy_model_section = saved_model_config_dict_for_filename.get('model', {})
        if model_type_from_saved_config == 'vgg' and 'vgg_params' in legacy_model_section:
            backbone_params_from_model_file = legacy_model_section['vgg_params']
        elif model_type_from_saved_config == 'resnet' and 'resnet_params' in legacy_model_section:
            backbone_params_from_model_file = legacy_model_section['resnet_params']
    
    if not backbone_params_from_model_file:
        print(f"警告: 未在模型配置中找到 '{model_type_from_saved_config}_params' 或兼容参数。骨干网络将尝试使用默认参数实例化。")

    backbone_instance, _ = get_backbone(
        config_model_params=backbone_params_from_model_file,
        model_type_str=model_type_from_saved_config,
        image_size=effective_image_size
    )
    if not backbone_instance:
        print(f"错误: 实例化骨干网络 ({model_type_from_saved_config.upper()}) 失败。")
        return
    print(f"骨干网络 ({model_type_from_saved_config.upper()}) 实例化成功。")

    # --- 6. 加载骨干网络权重 --- 
    # 人脸对比通常只关心骨干网络的权重。
    weights_key_to_load = 'backbone' # 新版检查点中骨干权重键名
    # 兼容旧的VGG模型文件，其骨干权重可能存储在 'model' 键下
    if model_type_from_saved_config == 'vgg' and 'model' in state_dict_container and 'backbone' not in state_dict_container:
        weights_key_to_load = 'model' 
        print("提示: 检测到可能是旧版VGG模型文件，将尝试从 'model' 键加载骨干权重。")

    if weights_key_to_load in state_dict_container and backbone_instance:
        try:
            backbone_instance.set_state_dict(state_dict_container[weights_key_to_load])
            backbone_instance.eval() # 设置为评估模式
            print(f"{model_type_from_saved_config.upper()} 骨干网络权重从 '{weights_key_to_load}' 键加载成功，并已设为评估模式。")
        except Exception as e_load_weights:
            print(f"错误: 加载骨干网络权重到实例化模型失败: {e_load_weights}")
            return
    elif backbone_instance:
        print(f"错误: 在模型文件 {config.model_path} 中未找到预期的骨干网络权重键 '{weights_key_to_load}'。")
        return
    else: # backbone_instance 本身实例化失败时，前面已返回
        pass 

    # --- 7. 预处理两张输入图像 --- 
    try:
        img1_tensor_np = process_image_local(config.img1, effective_image_size)
        img2_tensor_np = process_image_local(config.img2, effective_image_size)
    except FileNotFoundError as e_file:
        print(e_file); return
    except Exception as e_proc:
        print(f"处理输入图像时发生错误: {e_proc}"); return

    img1_tensor = paddle.to_tensor(img1_tensor_np)
    img2_tensor = paddle.to_tensor(img2_tensor_np)
    
    # --- 8. 提取特征并计算相似度 --- 
    with paddle.no_grad(): # 推理时不需要计算梯度
        feature1 = backbone_instance(img1_tensor).numpy()
        feature2 = backbone_instance(img2_tensor).numpy()

    if feature1 is None or feature2 is None or feature1.size == 0 or feature2.size == 0:
        print("错误: 特征提取失败，一个或两个图像的特征为空。无法进行比较。")
        return
        
    similarity_score = compute_similarity_fc(feature1, feature2)
    
    # --- 9. 获取对比阈值和可视化标志 --- 
    # 从当前脚本的config对象中获取对比特定的配置 (通常在 compare: {...} 块下)
    compare_specific_config = config.get('compare', {}) 
    comparison_threshold = compare_specific_config.get('compare_threshold', 0.8) # 默认阈值0.8
    visualize_comparison = compare_specific_config.get('compare_visualize', False) # 默认不可视化

    # --- 10. 判断并打印结果 --- 
    is_same_person = similarity_score >= comparison_threshold
    judgment_str = "是同一个人" if is_same_person else "不是同一个人"
    
    print(f"\n--- 人脸对比结果 ---")
    print(f"  图像1: {config.img1}")
    print(f"  图像2: {config.img2}")
    print(f"  使用模型: {config.model_path} (骨干: {model_type_from_saved_config.upper()}, 训练时损失: {loss_type_from_saved_config.upper()})")
    print(f"  计算得到的余弦相似度: {similarity_score:.4f}")
    print(f"  判断阈值: {comparison_threshold:.2f}")
    print(f"  结论: 这两张图像中的人脸 【{judgment_str}】")
    print(f"--------------------")
    
    # --- 11. 可视化结果 (如果配置启用) --- 
    show_comparison_result(
        img1_path=config.img1, 
        img2_path=config.img2, 
        similarity=similarity_score, 
        model_type_loaded=model_type_from_saved_config, 
        loss_type_loaded_from_model_cfg=loss_type_from_saved_config, 
        threshold=comparison_threshold, 
        visualize_flag=visualize_comparison
    )

if __name__ == '__main__':
    # --- 命令行参数解析 --- 
    parser = argparse.ArgumentParser(description='人脸图像对比工具脚本')
    
    # 核心控制参数
    parser.add_argument('--config_path', type=str, default=None, 
                        help='指定YAML配置文件的路径。')
    parser.add_argument('--img1', type=str, # required=True, 但由config_utils处理
                        help='第一张待对比的人脸图像路径 (必需，除非在YAML中指定)。')
    parser.add_argument('--img2', type=str, # required=True, 但由config_utils处理
                        help='第二张待对比的人脸图像路径 (必需，除非在YAML中指定)。')
    parser.add_argument('--model_path', type=str, # required=True, 但由config_utils处理
                        help='训练好的模型文件路径 (.pdparams) (必需，除非在YAML中指定)。')
    
    # 其他可覆盖配置文件的参数
    parser.add_argument('--use_gpu', action=argparse.BooleanOptionalAction, default=None,
                        help='是否使用GPU (覆盖配置文件中的 global_settings.use_gpu)。')
    parser.add_argument('--image_size', type=int, default=None,
                        help='输入图像预处理大小 (覆盖配置文件或模型自带的 image_size)。')
    
    # 对比特定参数 (覆盖配置文件中 compare: {...} 下的同名项)
    parser.add_argument('--compare_threshold', type=float, default=None,
                        help='判断为同一人的相似度阈值 (覆盖 compare.compare_threshold)。')
    parser.add_argument('--compare_visualize', action=argparse.BooleanOptionalAction, default=None,
                        help='是否可视化对比结果并保存图像 (覆盖 compare.compare_visualize)。')
    
    cmd_line_args = parser.parse_args()

    # --- 配置加载与合并 --- 
    # 使用 config_utils.load_config 函数加载和合并配置。
    final_config = load_config(
        default_yaml_path='configs/default_config.yaml',
        cmd_args_namespace=cmd_line_args
    )

    # 检查关键路径是否已配置 (model_path, img1, img2)
    # config_utils.load_config 会在必要参数缺失时尝试从args中找，如果还找不到且argparse中定义为required，会报错。
    # 此处可再加一层检查以确保用户友好性，特别是对于非required但在逻辑上必需的参数。
    if not final_config.model_path:
        parser.error("错误: 缺少模型文件路径。请通过 --model_path 命令行参数或在YAML配置文件中提供 model_path。")
    if not final_config.img1:
        parser.error("错误: 缺少第一张对比图像路径。请通过 --img1 命令行参数或在YAML配置文件中提供 img1。")
    if not final_config.img2:
        parser.error("错误: 缺少第二张对比图像路径。请通过 --img2 命令行参数或在YAML配置文件中提供 img2。")

    # 尝试设置matplotlib中文字体，以便在可视化结果中正确显示中文
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 使用 SimHei 字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    except Exception as e_font:
        print(f"提示: 设置matplotlib中文字体SimHei失败: {e_font}。可视化结果中的中文可能显示为乱码。")
        print(f"       请确保系统中安装了SimHei字体，或在代码中指定其他可用的中文字体。")

    print(f"开始对比图像: \n  图像1: {final_config.img1}\n  图像2: {final_config.img2}")
    compare_faces(final_config) 