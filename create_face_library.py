import os
import argparse
import numpy as np
import paddle
import pickle # 用于保存特征库
import json # 也可用于保存，或保存元数据
from tqdm import tqdm # 用于显示进度条
import cv2 # 用于本地图像处理

from vgg import VGGFace
from resnet_new import ResNetFace # ArcFaceHead在这里不需要，只用backbone提特征
from config_utils import load_config # 导入配置加载工具

# Copied process_image here for self-containment. Ideally, move to a utils.py
def process_image_local(img_path, size=64):
    img = cv2.imread(img_path) # cv2 needs to be imported if not already
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

def create_face_library(config):
    """创建人脸特征库，使用config对象获取参数"""
    if config.use_gpu and paddle.is_compiled_with_cuda():
        paddle.set_device('gpu')
        print("使用 GPU 进行特征提取")
    else:
        paddle.set_device('cpu')
        print("使用 CPU 进行特征提取")

    # config.model_path 是创建特征库时指定的具体模型文件路径
    if not os.path.exists(config.model_path):
        print(f"错误: 找不到指定的模型文件 {config.model_path}")
        return

    print(f"从 {config.model_path} 加载模型...")
    state_dict_container = paddle.load(config.model_path)

    # 从模型文件中恢复训练时的配置 (之前是 args，现在应该是 config 的字典形式)
    saved_model_config_dict = state_dict_container.get('config', {})
    if not saved_model_config_dict:
        saved_model_config_dict = state_dict_container.get('args', {})
        if saved_model_config_dict and not isinstance(saved_model_config_dict, dict):
            saved_model_config_dict = vars(saved_model_config_dict)
    
    if not saved_model_config_dict:
        print(f"错误: 模型文件 {config.model_path} 中缺少训练配置信息 ('config' 或 'args' 键)。")
        return
    
    model_type_loaded = saved_model_config_dict.get('model_type', 'vgg')
    num_classes_loaded = saved_model_config_dict.get('num_classes') # VGG结构定义可能需要
    # 图像大小: 优先用模型保存的，然后是当前脚本config的，最后默认64
    image_size_from_model = saved_model_config_dict.get('image_size')
    # config.image_size 来自 create_face_library 脚本的配置或命令行
    current_image_size = image_size_from_model if image_size_from_model is not None else config.image_size

    if config.image_size != current_image_size and image_size_from_model is not None:
        print(f"警告: 命令行图像大小 ({config.image_size})与模型训练时({current_image_size})不一致。将使用: {current_image_size}")

    vgg_model_instance = None
    resnet_backbone_instance = None

    if model_type_loaded == 'vgg':
        saved_vgg_params = saved_model_config_dict.get('model', {}).get('vgg_params', {})
        dropout_rate_loaded = saved_vgg_params.get('dropout_rate', 0.5)
        vgg_model_instance = VGGFace(num_classes=num_classes_loaded, dropout_rate=dropout_rate_loaded)
        if 'model' in state_dict_container:
            vgg_model_instance.set_state_dict(state_dict_container['model'])
            vgg_model_instance.eval()
            print(f"VGG 模型加载成功。")
        else:
            print(f"错误: VGG模型权重 'model' 不在 {config.model_path} 中。"); return
    elif model_type_loaded == 'resnet':
        saved_resnet_params = saved_model_config_dict.get('model', {}).get('resnet_params', {})
        feature_dim_loaded = saved_resnet_params.get('feature_dim', 512)
        nf_loaded = saved_resnet_params.get('nf', 32)
        n_resnet_blocks_loaded = saved_resnet_params.get('n_resnet_blocks', 3)
        # ArcFace head 不是必须的，因为只用backbone提特征
        resnet_backbone_instance = ResNetFace(nf=nf_loaded, n=n_resnet_blocks_loaded, feature_dim=feature_dim_loaded)
        if 'backbone' in state_dict_container:
            resnet_backbone_instance.set_state_dict(state_dict_container['backbone'])
            resnet_backbone_instance.eval()
            print(f"ResNet 模型骨干加载成功。")
        else:
            print(f"错误: ResNet骨干权重 'backbone' 不在 {config.model_path} 中。"); return
    else:
        print(f"错误: 不支持的模型类型 '{model_type_loaded}'。"); return

    # config.data_list_file 来自 create_face_library 脚本的配置或命令行
    if not config.data_list_file or not os.path.exists(config.data_list_file):
        print(f"错误: 找不到数据列表文件 {config.data_list_file}"); return

    image_paths_and_labels = []
    with open(config.data_list_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                image_paths_and_labels.append((parts[0], int(parts[1])))
            else:
                print(f"警告: 跳过格式不正确的行: {line.strip()}")
    
    if not image_paths_and_labels: print("错误: 数据列表文件为空或未能解析出任何图像和标签。"); return

    features_by_label = {}
    print(f"开始从 {len(image_paths_and_labels)} 张图像中提取特征...")

    for img_path, label in tqdm(image_paths_and_labels, desc="提取特征"):
        try:
            img_tensor_np = process_image_local(img_path, current_image_size)
            img_tensor = paddle.to_tensor(img_tensor_np)
            current_feature = None
            with paddle.no_grad():
                if vgg_model_instance:
                    current_feature, _ = vgg_model_instance(img_tensor)
                elif resnet_backbone_instance:
                    current_feature = resnet_backbone_instance(img_tensor)
            if current_feature is not None:
                if label not in features_by_label: features_by_label[label] = []
                features_by_label[label].append(current_feature.numpy().flatten())
        except FileNotFoundError:
            print(f"警告: 图像文件未找到 {img_path}，已跳过。")
        except Exception as e:
            print(f"警告: 处理图像 {img_path} 时发生错误: {e}，已跳过。")

    if not features_by_label: print("错误: 未能成功提取任何特征。请检查模型和数据。"); return

    average_features_library = {}
    print("\n计算每个类别的平均特征向量...")
    for label, feat_list in tqdm(features_by_label.items(), desc="计算平均特征"):
        if feat_list:
            avg_feat = np.mean(np.array(feat_list), axis=0)
            average_features_library[label] = avg_feat
        else:
            print(f"警告: 类别 {label} 没有成功提取到任何特征，将从库中排除。")
    
    if not average_features_library: print("错误: 未能计算出任何类别的平均特征。特征库为空。"); return

    # config.output_library_path 来自 create_face_library 脚本配置或命令行
    lib_cfg = config.get('create_library', {})
    output_lib_path = lib_cfg.get('output_library_path', f'model/face_library_{model_type_loaded}.pkl')
    output_dir = os.path.dirname(output_lib_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir); print(f"创建输出目录: {output_dir}")

    try:
        with open(output_lib_path, 'wb') as f_lib:
            pickle.dump(average_features_library, f_lib)
        print(f"\n人脸特征库已成功保存到: {output_lib_path}")
        print(f"库中包含 {len(average_features_library)} 个身份的特征。")
        
        # 可选的JSON元数据保存，可以使用 config.label_file 如果在config中定义并提供了路径
        # 例如：readme_json_path = config.get('label_file', os.path.join(os.path.dirname(config.data_list_file), "readme.json"))
        # 但当前 create_face_library 的参数中没有直接的 label_file，所以保持原有逻辑
        readme_json_path = os.path.join(os.path.dirname(config.data_list_file), "readme.json")
        if os.path.exists(readme_json_path):
            label_to_name = {}
            try:
                with open(readme_json_path, 'r', encoding='utf-8') as f_readme:
                    meta_data = json.load(f_readme)
                    for class_info in meta_data.get('class_detail', []):
                        label_to_name[class_info.get('class_label')] = class_info.get('class_name')
                # (此处可以添加保存带名称的JSON库的逻辑，如果需要的话)
            except Exception as e:
                print(f"处理readme.json时出错: {e}")
    except Exception as e:
        print(f"保存特征库到 {output_lib_path} 失败: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='创建人脸特征库脚本')
    
    # --- 关键命令行参数 ---
    parser.add_argument('--config_path', type=str, default='configs/default_config.yaml',
                        help='YAML 配置文件路径。')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='训练好的模型文件路径 (例如: model/face_model_resnet.pdparams) (必需)。')
    parser.add_argument('--data_list_file', type=str, required=True, 
                        help='包含图像路径和标签的数据列表文件 (例如: data/face/trainer.list) (必需)。')
    parser.add_argument('--use_gpu', action=argparse.BooleanOptionalAction, 
                        help='是否使用GPU进行特征提取 (覆盖配置文件)。')

    # --- 其他参数 (将从配置文件读取，也可通过命令行覆盖) ---
    parser.add_argument('--output_library_path', type=str, 
                        help='生成的人脸特征库文件保存路径 (覆盖配置文件)。')
    parser.add_argument('--image_size', type=int, 
                        help='图像预处理尺寸 (覆盖配置文件, 但模型自身保存的image_size优先)。')
    
    args = parser.parse_args()

    # 加载配置
    # default_yaml_path 是此脚本认为的默认配置文件位置 (configs/default_config.yaml)
    # cmd_args_namespace 是解析后的命令行参数
    config = load_config(default_yaml_path='configs/default_config.yaml', cmd_args_namespace=args)
    
    create_face_library(config) # 传递合并后的config对象 