# create_face_library.py
# 该脚本的目的是为基于特征比对的人脸识别模型 (如 ArcFace) 创建一个人脸特征库。
# 特征库是一个字典，其中键是人物的身份标签 (通常是整数ID)，值是该人物的多张图像的平均特征向量。
#
# 主要流程：
# 1. 加载配置文件和命令行参数，获取模型路径、数据集路径、运行设备等信息。
# 2. 设置运行设备 (CPU/GPU)。
# 3. 加载预训练好的人脸识别模型（通常只加载骨干网络部分，因为特征提取是主要任务）。
#    模型文件应包含训练时的配置信息 (如 model_type, image_size) 以便正确恢复骨干网络。
# 4. 使用 model_factory.py 动态实例化骨干网络。
# 5. 加载骨干网络权重到实例化的网络中，并设置为评估模式。
# 6. 遍历指定的数据集目录 (由 config.data_dir 和 config.class_name 定义)。
#    数据集目录结构应为：
#      data_dir/
#      └── class_name/  (例如: "face_dataset")
#          ├── person1_label_X/ (X是分配给person1的整数ID)
#          │   ├── image1.jpg
#          │   ├── image2.png
#          │   └── ...
#          ├── person2_label_Y/
#          │   ├── imageA.jpg
#          │   └── ...
#          └── readme.json (由 CreateDataList.py 生成，包含类别ID到真实姓名的映射)
#    脚本会解析子目录名中的标签ID (例如 "person1_label_X" 中的 X)。
# 7. 对每个身份的每张图像进行以下操作：
#    a. 图像预处理 (缩放、归一化、标准化)。
#    b. 使用加载的骨干网络提取图像的人脸特征向量。
# 8. 对每个身份，将其所有图像的特征向量进行平均，得到该身份的平均特征向量。
# 9. 将所有身份的平均特征向量存储在一个字典中 {label_id: avg_feature_vector}。
# 10. 使用 pickle 将这个特征库字典序列化并保存到磁盘文件 (通常是 .pkl 文件)。
#     保存路径由配置中的 `infer.face_library_path` 指定。
# 11. 同时，确保 `readme.json` (包含标签ID到真实名称的映射) 与特征库一起提供，
#     以便在后续推理时可以将识别出的标签ID转换为可读的人名。
#
# 注意: 此脚本中的 process_image_local 函数与 infer.py 和 compare.py 中的类似函数功能相同，
#       理想情况下应移至共享的 utils.py 模块。

import os
import cv2
import argparse
import numpy as np
import paddle
import pickle # 用于序列化和保存特征库字典
import json   # 用于读取 readme.json 以获取标签ID到真实姓名的映射
from tqdm import tqdm # 用于显示处理进度条
from config_utils import load_config, ConfigObject # 导入配置加载工具和配置对象类型
from model_factory import get_backbone # 仅需骨干网络进行特征提取

# 假设 process_image_local 与 infer.py, compare.py 中的版本功能一致
# 为避免重复，理想情况下应将其移至 utils.py 并从那里导入
def process_image_local(img_path: str, target_size: int = 64, 
                        mean_rgb: list[float] = [0.485, 0.456, 0.406], 
                        std_rgb: list[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    对单张输入图像进行预处理，为模型提取特征做准备。
    （详细文档参考 infer.py 或 compare.py 中的同名函数）
    """
    img = cv2.imread(img_path)
    if img is None: raise FileNotFoundError(f"错误: 无法读取图像文件 {img_path}")
    img = cv2.resize(img, (target_size, target_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    mean = np.array(mean_rgb, dtype='float32').reshape((1, 1, 3))
    std = np.array(std_rgb, dtype='float32').reshape((1, 1, 3))
    img_normalized = (img - mean) / std
    img_chw = img_normalized.transpose((2, 0, 1))
    img_expanded = np.expand_dims(img_chw, axis=0)
    return img_expanded.astype('float32')

def create_face_library(config: ConfigObject):
    """创建人脸特征库的主函数。

    根据提供的配置对象 `config`，遍历数据集，提取每个身份的平均特征向量，
    并将这些特征向量保存到指定的特征库文件中。

    Args:
        config (ConfigObject): 包含所有所需参数的配置对象。
                               关键参数包括 `model_path` (模型路径), `data_dir` (数据集根目录),
                               `class_name` (数据集子目录名), `use_gpu` (是否使用GPU), 
                               以及 `infer.face_library_path` (特征库保存路径)。
    """
    # --- 1. 设置运行设备 --- 
    if config.use_gpu and paddle.is_compiled_with_cuda():
        paddle.set_device('gpu')
        print("使用 GPU 进行特征提取")
    else:
        paddle.set_device('cpu')
        print("使用 CPU 进行特征提取")
    
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
    saved_model_config_dict = state_dict_container.get('config')
    if not saved_model_config_dict:
        saved_model_config_dict = state_dict_container.get('args')
        if saved_model_config_dict and not isinstance(saved_model_config_dict, dict):
            saved_model_config_dict = vars(saved_model_config_dict)
    
    if not saved_model_config_dict or not isinstance(saved_model_config_dict, dict):
        print(f"错误: 模型文件 {config.model_path} 中缺少有效的训练配置信息。")
        return

    model_type_from_saved_config = saved_model_config_dict.get('model_type', 'resnet')
    # 确定图像尺寸：优先使用模型训练时的尺寸
    image_size_from_model_file = saved_model_config_dict.get('image_size')
    effective_image_size = image_size_from_model_file
    if hasattr(config, 'image_size') and config.image_size is not None:
        if effective_image_size is not None and config.image_size != effective_image_size:
             print(f"警告: 当前配置的图像大小 ({config.image_size}) 与模型训练时 ({effective_image_size}) 不一致。将优先使用模型训练时的大小: {effective_image_size}")
        elif effective_image_size is None:
            effective_image_size = config.image_size
    if effective_image_size is None:
        print(f"错误: 无法确定图像处理尺寸。模型或当前配置均未提供 'image_size'。")
        return
    print(f"特征提取时将使用图像大小: {effective_image_size}x{effective_image_size}")

    # --- 4. 实例化骨干网络 --- 
    backbone_params_from_model_file = saved_model_config_dict.get('backbone_params', {})
    if not backbone_params_from_model_file:
        legacy_model_section = saved_model_config_dict.get('model', {})
        if model_type_from_saved_config == 'vgg' and 'vgg_params' in legacy_model_section:
            backbone_params_from_model_file = legacy_model_section['vgg_params']
        elif model_type_from_saved_config == 'resnet' and 'resnet_params' in legacy_model_section:
            backbone_params_from_model_file = legacy_model_section['resnet_params']
    
    backbone_instance, _ = get_backbone(
        config_model_params=backbone_params_from_model_file,
        model_type_str=model_type_from_saved_config,
        image_size=effective_image_size
    )
    if not backbone_instance:
        print(f"错误: 实例化骨干网络 ({model_type_from_saved_config.upper()}) 失败。")
        return
    print(f"骨干网络 ({model_type_from_saved_config.upper()}) 实例化成功。")

    # --- 5. 加载骨干网络权重并设为评估模式 --- 
    weights_key_to_load = 'backbone'
    if model_type_from_saved_config == 'vgg' and 'model' in state_dict_container and 'backbone' not in state_dict_container:
        weights_key_to_load = 'model' 

    if weights_key_to_load in state_dict_container and backbone_instance:
        try:
            backbone_instance.set_state_dict(state_dict_container[weights_key_to_load])
            backbone_instance.eval() 
            print(f"{model_type_from_saved_config.upper()} 骨干网络权重从 '{weights_key_to_load}' 键加载成功，并设为评估模式。")
        except Exception as e_load_weights:
            print(f"错误: 加载骨干网络权重失败: {e_load_weights}")
            return
    elif backbone_instance:
        print(f"错误: 模型文件 {config.model_path} 中未找到预期的骨干网络权重键 '{weights_key_to_load}'。")
        return

    # --- 6. 确定数据集路径和特征库保存路径 --- 
    dataset_base_path = os.path.join(config.data_dir, config.class_name)
    if not os.path.isdir(dataset_base_path):
        print(f"错误: 数据集路径 '{dataset_base_path}' 不是一个有效的目录。请检查 config.data_dir 和 config.class_name。")
        return
    
    infer_specific_config = config.get('infer', {}) # 推理相关配置通常在 infer 键下
    face_library_save_path = infer_specific_config.get('face_library_path')
    if not face_library_save_path:
        print(f"错误: 未在配置中指定人脸特征库的保存路径 (期望在 infer.face_library_path)。")
        # 可以选择一个默认路径，或者直接报错返回
        # default_lib_name = f"face_features_{model_type_from_saved_config}.pkl"
        # face_library_save_path = os.path.join(config.get('results_dir', 'results'), default_lib_name)
        # print(f"       将尝试使用默认路径: {face_library_save_path}")
        return # 修复：原先此处没有return，导致后续代码可能在路径无效时执行
        
    # 确保保存特征库的目录存在
    lib_save_dir = os.path.dirname(face_library_save_path)
    if lib_save_dir and not os.path.exists(lib_save_dir):
        os.makedirs(lib_save_dir)
        print(f"已创建特征库保存目录: {lib_save_dir}")

    # --- 7. 遍历数据集，提取并平均每个身份的特征 --- 
    face_feature_library = {} # 初始化空的特征库字典 {label_id: avg_feature_vector}
    identities_processed_count = 0
    total_images_processed_count = 0

    print(f"\n开始从数据集 '{dataset_base_path}' 创建人脸特征库...")
    # 遍历数据集基路径下的每个子目录 (每个子目录代表一个身份)
    # 使用 try-except 块来处理 os.listdir 可能遇到的权限等问题
    try:
        person_directories = os.listdir(dataset_base_path)
    except OSError as e_listdir:
        print(f"错误: 无法列出数据集目录 '{dataset_base_path}' 中的内容: {e_listdir}")
        return

    for person_dir_name in tqdm(person_directories, desc="处理身份进度"):
        person_full_path = os.path.join(dataset_base_path, person_dir_name)
        if not os.path.isdir(person_full_path): continue # 跳过非目录文件，如 readme.json

        # 解析身份标签ID。目录名格式通常为 "真实姓名_label_ID" 或 "ID"。
        # 优先尝试从目录名中提取 "_label_" 后面的数字作为ID。
        label_id_str = None
        person_label_id_resolved = None # 用于存储解析成功的ID
        if "_label_" in person_dir_name:
            try:
                label_id_str = person_dir_name.split("_label_")[-1]
                person_label_id_resolved = int(label_id_str)
            except (ValueError, IndexError):
                print(f"警告: 无法从目录名 '{person_dir_name}' 中解析出有效的标签ID (格式应为 ..._label_ID)。将尝试使用目录名作为ID。")
                person_label_id_resolved = person_dir_name # 使用整个目录名作为ID (可能需要后续处理)
        else:
            try:
                person_label_id_resolved = int(person_dir_name)
            except ValueError:
                print(f"警告: 目录名 '{person_dir_name}' 不符合预期的标签ID格式。将使用目录名字符串作为临时ID。")
                person_label_id_resolved = person_dir_name

        person_features_list = [] # 存储当前身份所有图像的特征向量列表
        num_images_for_person = 0

        try:
            images_in_person_dir = os.listdir(person_full_path)
        except OSError as e_listdir_person:
            print(f"警告: 无法列出身份目录 '{person_full_path}' 中的图像: {e_listdir_person}。跳过此身份。")
            continue

        for image_filename in images_in_person_dir:
            image_full_path = os.path.join(person_full_path, image_filename)
            if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue 
            
            try:
                preprocessed_image_np = process_image_local(image_full_path, effective_image_size)
                image_tensor = paddle.to_tensor(preprocessed_image_np)
                
                with paddle.no_grad(): 
                    feature_vector_tensor = backbone_instance(image_tensor)
                    feature_vector_np = feature_vector_tensor.numpy().flatten() 
                
                if feature_vector_np.size > 0:
                    person_features_list.append(feature_vector_np)
                    num_images_for_person += 1
                else:
                    print(f"警告: 从图像 {image_full_path} 提取的特征为空，将跳过此图像。")

            except FileNotFoundError:
                print(f"警告: 图像文件 {image_full_path} 未找到或无法读取，跳过。")
            except Exception as e_img_proc:
                print(f"警告: 处理图像 {image_full_path} 或提取特征时发生错误: {e_img_proc}，跳过此图像。")
        
        if person_features_list:
            avg_feature_vector = np.mean(person_features_list, axis=0)
            face_feature_library[person_label_id_resolved] = avg_feature_vector # 使用解析后的ID
            identities_processed_count += 1
            total_images_processed_count += num_images_for_person
        else:
            print(f"警告: 未能为身份 '{person_dir_name}' (解析ID: {person_label_id_resolved}) 提取任何有效的人脸特征。该身份将不会包含在特征库中。")

    # --- 8. 保存特征库到文件 --- 
    if not face_feature_library:
        print("错误: 未能从数据集中提取任何有效的人脸特征。特征库为空，将不会保存。")
        print(f"       请检查数据集路径 '{dataset_base_path}' 的结构、图像内容以及模型是否能正确提取特征。")
        return

    try:
        with open(face_library_save_path, 'wb') as f_pickle_out:
            pickle.dump(face_feature_library, f_pickle_out)
        print(f"\n人脸特征库创建完成！")
        print(f"  处理了 {identities_processed_count} 个不同身份，总共 {total_images_processed_count} 张有效图像。")
        print(f"  特征库已保存到: {face_library_save_path}")
        print(f"  特征库中包含 {len(face_feature_library)} 个身份的平均特征向量。")
        # 修复 f-string: \" 应为 \\" 或直接移除不必要的转义
        readme_json_expected_path = os.path.join(dataset_base_path, "readme.json")
        print(f"提示: 请确保与此特征库配套使用的标签映射文件 (例如 '{readme_json_expected_path}')是最新的，")
        print(f"       并且在推理时正确加载，以便将识别出的标签ID转换为可读的人名。")

    except Exception as e_save_lib:
        print(f"错误: 保存人脸特征库到文件 {face_library_save_path} 失败: {e_save_lib}")

if __name__ == '__main__':
    # --- 命令行参数解析 --- 
    parser = argparse.ArgumentParser(description='创建人脸特征库脚本')
    
    # 核心控制参数
    parser.add_argument('--config_path', type=str, default=None, 
                        help='指定YAML配置文件的路径。')
    parser.add_argument('--model_path', type=str, # required=True, 但由config_utils处理
                        help='训练好的人脸识别模型文件路径 (.pdparams) (必需，除非在YAML中指定)。') 
    
    # 数据集和输出路径相关 (通常由YAML配置，但允许命令行覆盖)
    parser.add_argument('--data_dir', type=str, default=None,
                        help='包含各个人物图像子目录的数据集根目录 (覆盖 global_settings.data_dir)。')
    parser.add_argument('--class_name', type=str, default=None,
                        help='数据集根目录下的特定子目录名 (覆盖 global_settings.class_name)。')
    parser.add_argument('--face_library_path', type=str, default=None,
                        help='生成的特征库文件保存路径 (.pkl) (覆盖 infer.face_library_path)。')
    
    # 其他可覆盖配置文件的参数
    parser.add_argument('--use_gpu', action=argparse.BooleanOptionalAction, default=None,
                        help='是否使用GPU进行特征提取 (覆盖 global_settings.use_gpu)。')
    parser.add_argument('--image_size', type=int, default=None,
                        help='输入图像预处理后的统一大小 (覆盖配置文件或模型自带的 image_size)。')

    cmd_line_args = parser.parse_args()

    # --- 配置加载与合并 --- 
    final_config = load_config(
        default_yaml_path='configs/default_config.yaml',
        cmd_args_namespace=cmd_line_args
    )

    # 检查关键参数是否已配置 (model_path, data_dir, class_name, face_library_path)
    if not final_config.model_path:
        parser.error("错误: 缺少模型文件路径。请通过 --model_path 或YAML配置提供。")
    if not final_config.data_dir:
        parser.error("错误: 缺少数据集根目录。请通过 --data_dir 或YAML配置提供。")
    if not final_config.class_name:
        parser.error("错误: 缺少数据集子目录名。请通过 --class_name 或YAML配置提供。")
    # 检查 infer.face_library_path 是否在配置中
    infer_cfg_check = final_config.get('infer')
    if not infer_cfg_check or not infer_cfg_check.get('face_library_path'): 
        parser.error("错误: 缺少特征库保存路径。请在YAML配置文件的 infer.face_library_path 中指定，或通过 --face_library_path 提供。")

    # 执行特征库创建
    create_face_library(final_config) 