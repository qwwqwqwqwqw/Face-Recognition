# coding:utf-8
import os
import cv2
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，防止在无显示环境的服务器上出错
import matplotlib.pyplot as plt
import paddle
from vgg import VGGFace         # 导入VGG模型
from resnet_new import ResNetFace, ArcFaceHead # 导入新版ResNet模型和ArcFaceHead
# from infer import process_image # Removed to avoid circular dependency, use local or utils
from config_utils import load_config # 导入配置加载工具

# Copied process_image here for self-containment. Ideally, move to a utils.py
def process_image_local(img_path, size=64):
    img = cv2.imread(img_path)
    if img is None: raise FileNotFoundError(f"错误: 无法读取图像文件 {img_path}")
    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img_chw = img.transpose((2, 0, 1))
    img_standardized = (img_chw - mean) / std
    img_expanded = np.expand_dims(img_standardized, axis=0)
    return img_expanded.astype('float32')

def extract_feature(model_pack, img_tensor):
    """
    使用给定模型提取图像张量的特征
    Args:
        model_pack (tuple): 对于VGG是(vgg_model_instance, None)，对于ResNet是(resnet_backbone_instance, None)
        img_tensor (paddle.Tensor): 预处理后的图像张量
    Returns:
        numpy.ndarray: 提取到的特征向量, 或 None 如果失败
    """
    vgg_model, resnet_backbone = model_pack
    feature = None
    with paddle.no_grad():
        if vgg_model:
            vgg_model.eval()
            feature, _ = vgg_model(img_tensor) 
        elif resnet_backbone:
            resnet_backbone.eval()
            feature = resnet_backbone(img_tensor)
        else:
            print("错误: extract_feature 中模型未正确初始化。") # 明确错误信息
            return None # 返回None表示失败
    return feature.numpy() if feature is not None else None

def compute_similarity_fc(feature1, feature2): # Renamed to avoid conflict if infer.py is imported elsewhere
    """
    计算两个特征向量之间的余弦相似度
    Args:
        feature1 (numpy.ndarray): 第一个特征向量
        feature2 (numpy.ndarray): 第二个特征向量
    Returns:
        float: 余弦相似度得分 (范围通常在-1到1之间，但经L2归一化后可能更接近0到1)
    """
    #展平向量以确保它们是一维的
    f1 = feature1.flatten()
    f2 = feature2.flatten()
    
    # 计算余弦相似度
    # dot_product / (norm(f1) * norm(f2))
    # 避免除以零错误
    norm_f1 = np.linalg.norm(f1)
    norm_f2 = np.linalg.norm(f2)
    if norm_f1 == 0 or norm_f2 == 0:
        return 0.0 # 如果任一向量为零向量，相似度为0
    similarity = np.dot(f1, f2) / (norm_f1 * norm_f2)
    return similarity

def show_comparison_result(img1_path, img2_path, similarity, model_type_loaded, loss_type_loaded, threshold, visualize_flag):
    """可视化人脸对比结果并保存图像，使用config_obj控制是否可视化"""
    if not visualize_flag: return

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print(f"错误: 无法读取图像进行对比可视化 (img1: {img1_path}, img2: {img2_path})"); return

    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1); plt.imshow(img1_rgb); plt.title("图像1", fontproperties="SimHei"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(img2_rgb); plt.title("图像2", fontproperties="SimHei"); plt.axis('off')
    
    is_same_person = similarity >= threshold
    judgment_text = "同一个人" if is_same_person else "不同人"
    title_color = 'green' if is_same_person else 'red'
    result_text = f"相似度: {similarity:.4f}\n判断: {judgment_text} (阈值: {threshold})"
    plt.suptitle(result_text, fontsize=16, color=title_color, fontproperties="SimHei")
    
    result_dir = "results"
    if not os.path.exists(result_dir): os.makedirs(result_dir); print(f"创建结果保存目录: {result_dir}")
    
    base1 = os.path.basename(img1_path).split('.')[0]
    base2 = os.path.basename(img2_path).split('.')[0]
    model_suffix = f"{model_type_loaded}_{loss_type_loaded if model_type_loaded == 'resnet' else 'ce'}"
    result_filename = f"compare_{model_suffix}_{base1}_vs_{base2}.png"
    result_file_path = os.path.join(result_dir, result_filename)
    
    plt.savefig(result_file_path)
    print(f"对比结果图像已保存至: {result_file_path}")
    # try: plt.show() # Usually not wanted in scripts
    # except Exception as e: print(f"无法显示对比图像，但结果已保存: {e}")

def compare_faces(config):
    """人脸对比主函数，使用config对象获取参数"""
    if not os.path.exists(config.img1): print(f"错误: 图像1不存在: {config.img1}"); return
    if not os.path.exists(config.img2): print(f"错误: 图像2不存在: {config.img2}"); return
    
    if config.use_gpu and paddle.is_compiled_with_cuda():
        paddle.set_device('gpu'); print("使用 GPU 进行人脸对比")
    else:
        paddle.set_device('cpu'); print("使用 CPU 进行人脸对比")

    if not os.path.exists(config.model_path):
        print(f"错误: 找不到指定的模型文件 {config.model_path}"); return

    print(f"从 {config.model_path} 加载模型...")
    state_dict_container = paddle.load(config.model_path)

    saved_model_config_dict = state_dict_container.get('config', {}) # Config saved with the model
    if not saved_model_config_dict: 
        saved_model_config_dict = state_dict_container.get('args', {})
        if saved_model_config_dict and not isinstance(saved_model_config_dict, dict):
            saved_model_config_dict = vars(saved_model_config_dict)
    if not saved_model_config_dict: 
        print(f"错误: 模型文件 {config.model_path} 中缺少训练配置信息。"); return

    # Params from SAVED model config
    model_type_loaded = saved_model_config_dict.get('model_type', 'vgg')
    num_classes_loaded = saved_model_config_dict.get('num_classes') # For VGG structure
    loss_type_loaded = saved_model_config_dict.get('loss_type', 'cross_entropy') # For ResNet suffix

    image_size_from_model = saved_model_config_dict.get('image_size')
    current_image_size = image_size_from_model if image_size_from_model is not None else config.image_size

    if config.image_size != current_image_size and image_size_from_model is not None:
        print(f"警告: 命令行图像大小 ({config.image_size})与模型训练时({current_image_size})不一致。将使用: {current_image_size}")

    vgg_model_instance = None
    resnet_backbone_instance = None

    if model_type_loaded == 'vgg':
        saved_vgg_params = saved_model_config_dict.get('model', {}).get('vgg_params', {})
        dropout_rate_loaded = saved_vgg_params.get('dropout_rate', 0.5)
        vgg_model_instance = VGGFace(num_classes=num_classes_loaded, dropout_rate=dropout_rate_loaded)
        if 'model' in state_dict_container: vgg_model_instance.set_state_dict(state_dict_container['model'])
        else: print(f"错误: VGG模型权重 'model' 不在 {config.model_path} 中。"); return
        print(f"使用 VGG 模型进行对比 (定义分类数={num_classes_loaded})")
    elif model_type_loaded == 'resnet':
        saved_resnet_params = saved_model_config_dict.get('model', {}).get('resnet_params', {})
        feature_dim_loaded = saved_resnet_params.get('feature_dim', 512)
        nf_loaded = saved_resnet_params.get('nf', 32)
        n_resnet_blocks_loaded = saved_resnet_params.get('n_resnet_blocks', 3)
        resnet_backbone_instance = ResNetFace(nf=nf_loaded, n=n_resnet_blocks_loaded, feature_dim=feature_dim_loaded)
        if 'backbone' in state_dict_container: resnet_backbone_instance.set_state_dict(state_dict_container['backbone'])
        else: print(f"错误: ResNet骨干权重 'backbone' 不在 {config.model_path} 中。"); return
        print(f"使用 ResNet 模型骨干进行对比 (特征维度={feature_dim_loaded})")
    else: print(f"错误: 不支持的模型类型 '{model_type_loaded}'。"); return
    
    model_pack_for_extraction = (vgg_model_instance, resnet_backbone_instance)

    try:
        img1_tensor_np = process_image_local(config.img1, current_image_size)
        img2_tensor_np = process_image_local(config.img2, current_image_size)
    except FileNotFoundError as e: print(e); return
    except Exception as e: print(f"处理图像时发生错误: {e}"); return

    img1_tensor = paddle.to_tensor(img1_tensor_np)
    img2_tensor = paddle.to_tensor(img2_tensor_np)
    
    feature1 = extract_feature(model_pack_for_extraction, img1_tensor)
    feature2 = extract_feature(model_pack_for_extraction, img2_tensor)

    if feature1 is None or feature2 is None: # 检查特征提取是否成功
        print("错误: 特征提取失败，无法进行比较。"); return
        
    similarity = compute_similarity_fc(feature1, feature2)
    
    # Get params from current script's config (compare sub-block)
    compare_cfg = config.get('compare', {})
    threshold_val = compare_cfg.get('compare_threshold', 0.8)
    visualize_flag = compare_cfg.get('compare_visualize', False)

    is_same_person = similarity >= threshold_val
    judgment = "是同一个人" if is_same_person else "不是同一个人"
    
    print(f"图像1: {config.img1}")
    print(f"图像2: {config.img2}")
    print(f"使用模型类型: {model_type_loaded}{(f' ({loss_type_loaded})' if model_type_loaded == 'resnet' else '')}")
    print(f"计算得到的相似度: {similarity:.4f}")
    print(f"判断结果 (阈值 {threshold_val}): {judgment}")
    
    show_comparison_result(config.img1, config.img2, similarity, model_type_loaded, loss_type_loaded, threshold_val, visualize_flag)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='人脸对比工具脚本')
    
    # --- 关键命令行参数 ---
    parser.add_argument('--config_path', type=str, default=None,
                        help='YAML 配置文件路径。')
    parser.add_argument('--img1', type=str, required=True, help='第一张待对比的人脸图像路径 (必需)。')
    parser.add_argument('--img2', type=str, required=True, help='第二张待对比的人脸图像路径 (必需)。')
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型文件路径 (必需)。')
    parser.add_argument('--use_gpu', action=argparse.BooleanOptionalAction, help='是否使用GPU (覆盖配置文件)。')

    # --- 其他参数 (将从配置文件读取，也可通过命令行覆盖) ---
    parser.add_argument('--image_size', type=int, help='输入图像预处理大小 (覆盖全局 image_size)。')
    parser.add_argument('--compare_threshold', type=float, help='判断为同一人的相似度阈值 (覆盖 compare.compare_threshold)。')
    parser.add_argument('--compare_visualize', action=argparse.BooleanOptionalAction, # Python 3.9+
                        help='是否可视化对比结果 (覆盖 compare.compare_visualize)。')
    
    args = parser.parse_args()

    # 加载配置
    config = load_config(default_yaml_path='configs/default_config.yaml', cmd_args_namespace=args)

    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"设置matplotlib中文字体失败: {e}。如果标题中文显示乱码，请手动配置matplotlib。")

    print(f"开始对比图像: \n1. {config.img1}\n2. {config.img2}")
    compare_faces(config) # 传递合并后的config对象