# coding:utf-8
import os
import cv2
import argparse
import numpy as np
import paddle
import paddle.nn as nn # 确保导入nn，因为Linear头会用到
from vgg import VGGFace         # 导入VGG模型
from resnet_new import ResNetFace, ArcFaceHead # 导入新版ResNet模型和ArcFaceHead
import json
import matplotlib.pyplot as plt
import pickle # 用于加载特征库
from config_utils import load_config # 导入配置加载工具

def process_image_local(img_path, size=64):
    """
    处理图像，调整大小并进行归一化和标准化预处理
    Args:
        img_path (str): 输入图像的路径
        size (int): 目标图像大小 (高度和宽度相同)
    Returns:
        numpy.ndarray: 预处理后的图像数据，形状为 (1, 3, size, size)
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"错误: 无法读取图像文件 {img_path}")
    
    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img_chw = img.transpose((2, 0, 1))
    img_standardized = (img_chw - mean) / std
    
    img_expanded = np.expand_dims(img_standardized, axis=0)
    return img_expanded.astype('float32')

def compute_similarity(feature1, feature2):
    """计算两个特征向量之间的余弦相似度"""
    f1 = feature1.flatten()
    f2 = feature2.flatten()
    norm_f1 = np.linalg.norm(f1)
    norm_f2 = np.linalg.norm(f2)
    if norm_f1 == 0 or norm_f2 == 0:
        return 0.0
    return np.dot(f1, f2) / (norm_f1 * norm_f2)

def infer(config):
    """推理函数，使用config对象获取参数"""
    if config.use_gpu and paddle.is_compiled_with_cuda():
        paddle.set_device('gpu')
        print("使用 GPU 进行推理")
    else:
        paddle.set_device('cpu')
        print("使用 CPU 进行推理")
    
    if not os.path.exists(config.model_path): # model_path 现在是推理时指定的具体模型文件
        print(f"错误: 找不到指定的模型文件 {config.model_path}")
        return

    print(f"从 {config.model_path} 加载模型...")
    state_dict_container = paddle.load(config.model_path)

    # 从模型文件中恢复训练时的配置 (之前是 args，现在应该是 config 的字典形式)
    # 键名在 train.py 中保存为 'config'
    saved_model_config_dict = state_dict_container.get('config')
    if not saved_model_config_dict:
        # 尝试兼容旧的保存方式 (键为 'args')
        saved_model_config_dict = state_dict_container.get('args')
        if saved_model_config_dict and not isinstance(saved_model_config_dict, dict):
            saved_model_config_dict = vars(saved_model_config_dict) # 转为dict
    
    if not saved_model_config_dict:
        print(f"错误: 模型文件 {config.model_path} 中缺少训练配置信息 ('config' 或 'args' 键)。")
        print("请确保模型是使用更新后的 train.py 保存的。")
        return
    
    # 从保存的配置中获取关键信息
    # 使用 .get() 方法避免因缺少键而引发KeyError，并提供默认值
    model_type_loaded = saved_model_config_dict.get('model_type', 'vgg')
    num_classes_loaded = saved_model_config_dict.get('num_classes')
    loss_type_loaded = saved_model_config_dict.get('loss_type', 'cross_entropy')
    # 图像大小: 优先使用模型训练时的，然后是当前infer脚本config的，最后是默认64
    image_size_from_model = saved_model_config_dict.get('image_size')
    current_image_size = image_size_from_model if image_size_from_model is not None else config.image_size

    if config.image_size != current_image_size and image_size_from_model is not None:
        print(f"警告: 推理脚本图像大小 ({config.image_size}) 与模型训练时 ({current_image_size}) 不一致。将使用模型训练时的大小: {current_image_size}")

    backbone_loaded = None
    head_module_loaded = None
    vgg_model_instance_loaded = None

    if model_type_loaded == 'vgg':
        saved_vgg_params = saved_model_config_dict.get('model', {}).get('vgg_params', {})
        dropout_rate_loaded = saved_vgg_params.get('dropout_rate', 0.5)
        vgg_model_instance_loaded = VGGFace(num_classes=num_classes_loaded, dropout_rate=dropout_rate_loaded)
        if 'model' in state_dict_container:
            vgg_model_instance_loaded.set_state_dict(state_dict_container['model'])
            print(f"使用 VGG 模型进行推理，分类数量: {num_classes_loaded}")
        else:
            print(f"错误: VGG 模型权重 'model' 不在 {config.model_path} 中。")
            return
    elif model_type_loaded == 'resnet':
        saved_resnet_params = saved_model_config_dict.get('model', {}).get('resnet_params', {})
        feature_dim_loaded = saved_resnet_params.get('feature_dim', 512)
        nf_loaded = saved_resnet_params.get('nf', 32)
        n_resnet_blocks_loaded = saved_resnet_params.get('n_resnet_blocks', 3)

        backbone_loaded = ResNetFace(nf=nf_loaded, n=n_resnet_blocks_loaded, feature_dim=feature_dim_loaded)
        if 'backbone' in state_dict_container:
            backbone_loaded.set_state_dict(state_dict_container['backbone'])
            print(f"使用 ResNet 模型骨干进行推理，特征维度: {feature_dim_loaded}")
        else:
            print(f"错误: ResNet 模型骨干权重 'backbone' 不在 {config.model_path} 中。")
            return

        # 对于推理，如果不是ArcFace且不需要进行闭集分类，head_module可能不是必需的
        # 但如果模型保存了head权重，且需要用它（比如闭集识别），则加载
        if 'head' in state_dict_container:
            if loss_type_loaded == 'arcface':
                saved_arcface_params = saved_model_config_dict.get('loss', {}).get('arcface_params', {})
                arcface_m1_loaded = saved_arcface_params.get('arcface_m1', 1.0)
                arcface_m2_loaded = saved_arcface_params.get('arcface_m2', 0.5)
                arcface_m3_loaded = saved_arcface_params.get('arcface_m3', 0.0)
                arcface_s_loaded = saved_arcface_params.get('arcface_s', 64.0)
                head_module_loaded = ArcFaceHead(in_features=feature_dim_loaded, out_features=num_classes_loaded,
                                              margin1=arcface_m1_loaded, margin2=arcface_m2_loaded, 
                                              margin3=arcface_m3_loaded, scale=arcface_s_loaded)
                print(f"  L--> 加载 ArcFaceHead，分类数量: {num_classes_loaded}")
            elif loss_type_loaded == 'cross_entropy':
                head_module_loaded = nn.Linear(feature_dim_loaded, num_classes_loaded)
                print(f"  L--> 加载普通分类头 (Linear)，分类数量: {num_classes_loaded}")
            if head_module_loaded: head_module_loaded.set_state_dict(state_dict_container['head'])
        elif loss_type_loaded == 'arcface': # 如果是ArcFace但模型没存head，这通常不应该发生
            print(f"警告: 模型为ArcFace但未找到 'head' 权重。特征提取可能仍可用，但依赖head的分类会失败。")
        # 如果是非ArcFace且没有head权重，对于某些推理任务（如仅特征提取）也是可接受的

    else:
        print(f"错误: 不支持的模型类型 '{model_type_loaded}' (从模型文件中读取得到)。")
        return
    
    if vgg_model_instance_loaded: vgg_model_instance_loaded.eval()
    if backbone_loaded: backbone_loaded.eval()
    if head_module_loaded: head_module_loaded.eval()

    # Parameters from the CURRENT script's config (config.infer sub-block)
    infer_cfg = config.get('infer', {})
    label_file_path = infer_cfg.get('label_file', 'data/face/readme.json')
    face_lib_path = infer_cfg.get('face_library_path')
    recognition_thresh = infer_cfg.get('recognition_threshold', 0.5)
    visualize_output = infer_cfg.get('infer_visualize', False)

    label_id_to_name_map = {}
    # config.label_file 来自 infer 脚本的配置或命令行
    if not label_file_path or not os.path.exists(label_file_path):
        print(f"警告: 找不到标签文件 {label_file_path}。将只输出类别ID。")
    else:
        try:
            with open(label_file_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
                for class_info in label_data.get('class_detail', []):
                    label_id = class_info.get('class_label') 
                    class_name = class_info.get('class_name')
                    if label_id is not None and class_name is not None:
                         label_id_to_name_map[label_id] = class_name 
            if not label_id_to_name_map:
                print(f"警告: 标签文件 {label_file_path} 未能成功解析出标签映射。")
        except Exception as e:
            print(f"错误: 读取或解析标签文件 {label_file_path} 失败: {e}")

    face_library = None
    # is_arcface_model 现在基于加载的模型配置
    is_arcface_model_runtime = (model_type_loaded == 'resnet' and loss_type_loaded == 'arcface')
    
    # config.face_library_path 和 config.recognition_threshold 来自 infer 脚本的配置
    if is_arcface_model_runtime:
        if not face_lib_path or not os.path.exists(face_lib_path):
            print(f"错误: ArcFace模型需要人脸特征库。请提供有效的 infer.face_library_path (当前: {face_lib_path})。")
            return
        try:
            with open(face_lib_path, 'rb') as f_lib:
                face_library = pickle.load(f_lib)
            print(f"人脸特征库 {face_lib_path} 加载成功，包含 {len(face_library)} 个已知身份。")
            if not isinstance(face_library, dict):
                print("错误: 特征库格式不正确，应为字典类型。"); return
        except Exception as e:
            print(f"加载人脸特征库 {face_lib_path} 失败: {e}"); return

    try:
        img_tensor_np = process_image_local(config.image_path, current_image_size)
    except FileNotFoundError as e:
        print(e); return
    except Exception as e:
        print(f"处理图像 {config.image_path} 时发生错误: {e}"); return
        
    img_tensor = paddle.to_tensor(img_tensor_np)
    
    pred_label_id = -1
    pred_score = 0.0

    with paddle.no_grad(): 
        if is_arcface_model_runtime and face_library:
            if not backbone_loaded:
                print("错误: ResNet backbone 未初始化，无法提取特征用于ArcFace比对。"); return
            input_feature = backbone_loaded(img_tensor).numpy().flatten()
            
            best_match_label = -1
            highest_similarity = -1.0
            if not face_library: print("错误：ArcFace模型推理需要人脸库，但人脸库为空或未加载。"); return

            for lib_label, lib_feature_vector in face_library.items():
                similarity = compute_similarity(input_feature, lib_feature_vector)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match_label = lib_label
            pred_label_id = best_match_label
            pred_score = highest_similarity
            print(f"ArcFace 特征比对完成。")
        else: # VGG 或 ResNet+CrossEntropy (闭集分类逻辑)
            final_scores_for_pred = None
            if model_type_loaded == 'vgg' and vgg_model_instance_loaded:
                _, logits = vgg_model_instance_loaded(img_tensor) 
                final_scores_for_pred = logits
            elif model_type_loaded == 'resnet' and loss_type_loaded == 'cross_entropy' and backbone_loaded and head_module_loaded:
                features_bb = backbone_loaded(img_tensor)
                logits = head_module_loaded(features_bb)
                final_scores_for_pred = logits
            elif model_type_loaded == 'resnet' and loss_type_loaded == 'cross_entropy' and not head_module_loaded:
                 print("错误: ResNet+CrossEntropy 模型推理需要分类头 (head_module)，但未加载或初始化。"); return
            
            if final_scores_for_pred is None:
                print("错误: 未能计算出最终的预测分数 (非ArcFace模式)。可能是模型或head未正确加载。"); return
            
            probs_np = paddle.nn.functional.softmax(final_scores_for_pred, axis=1).numpy()
            pred_label_id = np.argmax(probs_np[0])  
            pred_score = probs_np[0][pred_label_id]
    
    predicted_class_name = label_id_to_name_map.get(pred_label_id, f"标签ID:{pred_label_id}")
    
    if is_arcface_model_runtime:
        if pred_score < recognition_thresh:
            print(f"输入人脸与库中所有已知身份的最高相似度为 {pred_score:.4f} (低于阈值 {recognition_thresh})。")
            predicted_class_name = f"未知人物 (相似度 {pred_score:.4f})"
        else:
            print(f"预测的人脸类别 (基于特征库): {predicted_class_name}, 最高相似度: {pred_score:.4f}")
    else:
        print(f"预测的人脸类别 (基于分类): {predicted_class_name}, 置信度: {pred_score:.4f}")

    if pred_label_id not in label_id_to_name_map and pred_label_id != -1 and label_file_path and os.path.exists(label_file_path):
        print(f"注意: 预测的类别ID {pred_label_id} 在标签文件 {label_file_path} 中找不到对应的名称。")
    
    if visualize_output: # 使用配置中的 infer_visualize
        try:
            img_display = cv2.imread(config.image_path)
            if img_display is None: print(f"警告: 无法读取图像 {config.image_path} 进行可视化。"); return
            img_rgb_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(8, 6))
            plt.imshow(img_rgb_display)
            title_text = f"{predicted_class_name}"
            if is_arcface_model_runtime: title_text += f" (Sim: {pred_score:.4f})"
            else: title_text += f" (Conf: {pred_score:.4f})"
            plt.title(title_text, fontproperties="SimHei") 
            plt.axis('off') 
            result_dir = "results"
            if not os.path.exists(result_dir): os.makedirs(result_dir); print(f"创建结果保存目录: {result_dir}")
            base_filename = os.path.basename(config.image_path)
            result_image_filename = f"recognition_{model_type_loaded}_{loss_type_loaded if model_type_loaded=='resnet' else 'ce'}_{base_filename}"
            result_file_path = os.path.join(result_dir, result_image_filename)
            plt.savefig(result_file_path)
            print(f"结果图像已保存至: {result_file_path}")
            plt.show()
        except ImportError:
            print("警告: matplotlib 未能正确导入或缺少中文字体支持，无法显示图像，但结果已保存。")
        except Exception as e:
            print(f"可视化过程中发生错误: {e}。结果图像可能已保存。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='人脸识别推理脚本')
    
    # --- 关键命令行参数 ---
    parser.add_argument('--config_path', type=str, default='configs/default_config.yaml',
                        help='YAML 配置文件路径。')
    parser.add_argument('--image_path', type=str, required=True, help='待识别的输入图像路径 (必需)。')
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型文件路径 (必需)。') 
    parser.add_argument('--use_gpu', action=argparse.BooleanOptionalAction, 
                        help='是否使用GPU进行推理 (覆盖配置文件)。')

    # --- 其他参数 (将从配置文件读取，也可通过命令行覆盖) ---
    # 这些参数的默认值建议在YAML中维护
    parser.add_argument('--image_size', type=int, help='输入图像预处理后的统一大小 (覆盖全局 image_size)。')
    parser.add_argument('--label_file', type=str, help='类别标签与名称映射的json文件路径 (覆盖 infer.label_file)。')
    parser.add_argument('--face_library_path', type=str, 
                        help='人脸特征库文件路径 (当使用ArcFace等特征比对模型时必需, 覆盖 infer.face_library_path)。')
    parser.add_argument('--recognition_threshold', type=float, 
                        help='对于ArcFace/特征比对模型, 判断为已知身份的最低相似度阈值 (覆盖 infer.recognition_threshold)。')
    parser.add_argument('--infer_visualize', action=argparse.BooleanOptionalAction, # Python 3.9+
                        help='是否可视化识别结果并保存图像 (覆盖 infer.infer_visualize)。')

    args = parser.parse_args()
    
    # 加载配置
    config = load_config(default_yaml_path='configs/default_config.yaml', cmd_args_namespace=args)

    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"设置matplotlib中文字体失败: {e}。如果标题中文显示乱码，请手动配置matplotlib。")

    print(f"开始推理图像: {config.image_path}")
    infer(config) # 传递合并后的config对象 