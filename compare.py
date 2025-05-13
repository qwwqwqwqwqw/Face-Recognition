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

from distutils import config
import json
import os
import cv2
import argparse
import numpy as np
import matplotlib

import heads
import model_factory
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

def compare_faces(config: ConfigObject, cmd_args: argparse.Namespace):
    """
    主比较函数，加载模型、人脸库和目标图片，进行比较。
    """
    # --- 设置设备 ---
    use_gpu_flag = config.use_gpu and paddle.is_compiled_with_cuda()
    paddle.set_device('gpu' if use_gpu_flag else 'cpu')
    print(f"使用 {'GPU' if use_gpu_flag else 'CPU'} 进行比较")

    # --- 加载模型 (复用 infer.py 的逻辑或简化) ---
    # 需要确定类别数来初始化模型
    num_classes = config.num_classes
    if num_classes is None:
        raise ValueError("错误: 比较函数的配置中缺少 'num_classes'。")
    
    backbone_instance = model_factory.create_backbone(config.model_type)
    head_module_instance = heads.create_head(config.loss_type, num_classes=num_classes, config=config)

    model_path = config.get('trained_model_path') # 优先使用指定路径
    if not model_path:
        # 自动构建路径 (逻辑同 infer.py)
        hardware_id = "gpu" if use_gpu_flag else "cpu"
        train_source = 'auto' # 假设比较时使用自动训练的模型，或可配置
        filename_base = f"best_model_{config.model_type}_{config.loss_type}_{hardware_id}_{train_source}"
        model_filename = f"{filename_base}.pdparams"
        model_path = os.path.join(config.get('model_save_dir', 'model'), model_filename)
        print(f"未指定 trained_model_path，尝试加载自动构建的最佳模型路径: {model_path}")
    else:
        print(f"将从指定路径加载模型: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"错误: 模型文件未找到: {model_path}")

    try:
        model_data = paddle.load(model_path)
        if 'backbone' in model_data: backbone_instance.set_state_dict(model_data['backbone'])
        else: print("警告: 模型文件中无 'backbone' 权重。")
        if 'head' in model_data: head_module_instance.set_state_dict(model_data['head'])
        else: print("警告: 模型文件中无 'head' 权重。")
        print(f"模型权重从 {model_path} 加载成功。")
        # 可以在此添加元数据检查逻辑，同 infer.py
    except Exception as e:
        raise RuntimeError(f"加载模型 {model_path} 失败: {e}")

    backbone_instance.eval()
    head_module_instance.eval()

    # --- 加载人脸库特征 ---
    library_path = cmd_args.library_path or config.get('face_library_path', 'face_library.npy')
    label_map_path = os.path.splitext(library_path)[0] + '_label_map.json'
    
    if not os.path.exists(library_path) or not os.path.exists(label_map_path):
        raise FileNotFoundError(f"错误: 人脸库文件 ({library_path}) 或标签映射文件 ({label_map_path}) 未找到。请先运行 create_face_library.py。")

    try:
        face_library_features = np.load(library_path)
        with open(label_map_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f) # label_map 是 {index: class_name} 的形式
            # 将其反转为 {class_name: index} 可能更方便，或者保留原样根据索引查找名称
        print(f"人脸库加载成功: {face_library_features.shape[0]} 个特征，来自 {len(label_map)} 个类别。")
        print(f"标签映射文件加载成功: {label_map_path}")
    except Exception as e:
        raise RuntimeError(f"加载人脸库或标签映射失败: {e}")

    # --- 处理目标图片 ---
    target_image_path = cmd_args.target_image_path
    if not os.path.exists(target_image_path):
        raise FileNotFoundError(f"错误: 目标图片文件未找到: {target_image_path}")

    try:
        # 复用或定义图像预处理逻辑
        # 注意：这里的预处理必须与训练和创建人脸库时完全一致！
        image_size = config.get('image_size')
        if not image_size:
            raise ValueError("错误: 配置中未找到 'image_size'。")
        
        img = cv2.imread(target_image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {target_image_path}")
        img = preprocess_image(img, image_size) # 使用与训练/建库一致的预处理函数
        img_tensor = paddle.to_tensor(img, dtype='float32')
        img_tensor = paddle.unsqueeze(img_tensor, axis=0) # 增加 batch 维度
    except Exception as e:
        raise RuntimeError(f"处理目标图片 {target_image_path} 失败: {e}")

    # --- 提取目标图片特征 ---
    with paddle.no_grad():
        target_features = backbone_instance(img_tensor)
        # 根据模型类型，决定是否需要头部输出，通常比较用骨干特征
        # target_features = head_module_instance(target_features) # 如果损失函数改变了特征空间，可能需要
        target_feature_vector = target_features.numpy().flatten() # 转换为一维 NumPy 数组

    # --- 计算相似度并进行比较 ---
    similarities = calculate_cosine_similarity(target_feature_vector, face_library_features)
    
    # 找到最相似的人脸索引和得分
    best_match_index = np.argmax(similarities)
    best_score = similarities[best_match_index]
    
    # 从 label_map 获取对应的类别名称
    # 注意 label_map 的结构，假设是 {index_str: class_name}
    predicted_class_name = label_map.get(str(best_match_index), "未知类别")

    print(f"\n--- 比较结果 ---")
    print(f"目标图片: {target_image_path}")
    print(f"最匹配的人脸库索引: {best_match_index}")
    print(f"预测类别: {predicted_class_name}")
    print(f"相似度得分 (余弦相似度): {best_score:.4f}")

    # 可以根据阈值判断是否匹配
    similarity_threshold = config.get('similarity_threshold', 0.5) # 从配置获取阈值，默认为 0.5
    if best_score >= similarity_threshold:
        print(f"判定结果: 匹配成功 (得分 >= {similarity_threshold})")
    else:
        print(f"判定结果: 匹配失败 (得分 < {similarity_threshold})")
    
    # 如果需要显示最相似的前 N 个结果
    top_n = 3 # 显示前 3 个
    if face_library_features.shape[0] >= top_n:
        top_indices = np.argsort(similarities)[-top_n:][::-1] # 获取最高分的N个索引
        print(f"\n前 {top_n} 个最相似结果:")
        for i, idx in enumerate(top_indices):
            score = similarities[idx]
            name = label_map.get(str(idx), "未知类别")
            print(f"  {i+1}. 类别: {name}, 索引: {idx}, 得分: {score:.4f}")

# --- Helper 函数 (如果还没定义) ---
def preprocess_image(image, target_size):
    """简单的图像预处理，需要与训练时保持一致。"""
    # 示例: 调整大小, 归一化
    image = cv2.resize(image, (target_size, target_size))
    image = image.astype('float32') / 255.0
    # 可能需要调整通道顺序 BGR -> RGB
    # image = image[:, :, ::-1]
    # 转换为 CHW 格式 (PaddlePaddle 需要)
    image = image.transpose((2, 0, 1))
    return image

def calculate_cosine_similarity(vector, matrix):
    """计算一个向量与一个矩阵中所有行向量的余弦相似度。"""
    # 标准化向量和矩阵的行
    vector_norm = vector / np.linalg.norm(vector)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    # 计算点积 (即余弦相似度)
    similarity = np.dot(matrix_norm, vector_norm)
    return similarity

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='人脸比较脚本')

    # 核心控制参数
    parser.add_argument('--config_path', type=str, default=None, help='YAML配置文件路径。')
    parser.add_argument('--use_gpu', action=argparse.BooleanOptionalAction, default=None, help='是否使用GPU。')
    parser.add_argument('--active_config', type=str, default=None, help='覆盖YAML中的active_config。')

    # 输入参数
    parser.add_argument('--target_image_path', type=str, required=True, help='需要识别的目标人脸图片路径。')
    parser.add_argument('--library_path', type=str, default=None, help='人脸库特征文件 (.npy) 路径。默认从配置或 face_library.npy 获取。')
    
    # 可覆盖配置文件的参数
    parser.add_argument('--num_classes', type=int, default=None, help='覆盖配置文件中的类别数 (必须与训练和建库时一致)。')
    parser.add_argument('--model_save_dir', type=str, help='模型保存目录 (用于自动构建模型路径)。')
    parser.add_argument('--model_type', type=str, choices=['vgg', 'resnet'], help='骨干网络类型')
    parser.add_argument('--loss_type', type=str, choices=['cross_entropy', 'arcface'], help='损失/头部类型')
    parser.add_argument('--image_size', type=int, help='图像尺寸')
    parser.add_argument('--trained_model_path', type=str, default=None, help='指定要加载的训练好的模型 (.pdparams) 文件路径。如果提供，将优先于自动构建的路径。')
    parser.add_argument('--similarity_threshold', type=float, help='覆盖配置文件中的相似度阈值。')
    parser.add_argument('--face_library_path', type=str, help='覆盖配置文件中的人脸库特征文件路径。')

    cmd_line_args = parser.parse_args()

    # --- 配置加载与合并 ---
    final_config = load_config(
        default_yaml_path='configs/default_config.yaml',
        cmd_args_namespace=cmd_line_args
    )

    # 打印最终生效的关键配置信息
    print("\n--- 最终生效的比较配置 (YAML与命令行合并后) ---")
    # ... (打印关键配置) ...
    print(f"  类别数 (num_classes): {final_config.num_classes}")
    print(f"  目标图片: {cmd_line_args.target_image_path}") # 使用命令行传入的路径
    print(f"  人脸库路径: {cmd_line_args.library_path or final_config.get('face_library_path', 'face_library.npy')}")
    print(f"  加载的模型路径: {final_config.get('trained_model_path') or '自动构建'}")
    print(f"  相似度阈值: {final_config.similarity_threshold}")
    print("---------------------------------------------------")

    # 检查 num_classes 是否最终确定
    if final_config.get('num_classes') is None:
         parser.error("错误: 最终配置中未能确定 'num_classes'。请检查YAML文件和命令行参数 --num_classes。")

    try:
        compare_faces(final_config, cmd_line_args)
    except FileNotFoundError as e:
        print(f"比较失败: {e}")
    except RuntimeError as e:
        print(f"比较时发生运行时错误: {e}")
    except ValueError as e:
        print(f"配置或输入错误: {e}")
    except Exception as e:
        print(f"发生意外错误: {e}") 