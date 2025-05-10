# coding:utf-8
import os
import cv2
import argparse
import numpy as np
import paddle
from vgg import VGGFace         # 导入VGG模型
from resnet_new import ResNetFace # 导入新版ResNet模型
import json
import matplotlib.pyplot as plt

def process_image(img_path, size=64):
    """
    处理图像，调整大小并进行归一化和标准化预处理
    Args:
        img_path (str): 输入图像的路径
        size (int): 目标图像大小 (高度和宽度相同)
    Returns:
        numpy.ndarray: 预处理后的图像数据，形状为 (1, 3, size, size)
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"错误: 无法读取图像文件 {img_path}")
    
    # 调整大小
    img = cv2.resize(img, (size, size))
    # BGR 转换为 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 转换为float32类型，并归一化到[0, 1]范围
    img = img.astype('float32') / 255.0
    
    # 标准化 (使用ImageNet的均值和标准差)
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    # HWC -> CHW 并进行标准化
    img_chw = img.transpose((2, 0, 1))
    img_standardized = (img_chw - mean) / std
    
    # 添加批处理维度 (batch_size=1)
    img_expanded = np.expand_dims(img_standardized, axis=0)
    return img_expanded.astype('float32') # 确保最终是float32

def infer(args):
    """推理函数"""
    # 设置运行设备 (GPU或CPU)
    if args.use_gpu and paddle.is_compiled_with_cuda():
        paddle.set_device('gpu')
        print("使用 GPU 进行推理")
    else:
        paddle.set_device('cpu')
        print("使用 CPU 进行推理")
    
    # -------------------- 模型选择与实例化 --------------------
    if args.model_type == 'vgg':
        model = VGGFace(num_classes=args.num_classes, dropout_rate=0.5) # VGG特有的dropout_rate
        print(f"使用 VGG 模型进行推理，分类数量: {args.num_classes}")
    elif args.model_type == 'resnet':
        # ResNetFace实例化，nf和n可以使用默认值或从命令行参数传入
        # 注意：推理时通常num_classes参数来自训练好的模型，但这里为保持与train.py一致性而传入
        # 实际分类头的权重是在加载模型时确定的。
        model = ResNetFace(num_classes=args.num_classes, nf=args.nf, n=args.n_resnet_blocks)
        print(f"使用 ResNet 模型进行推理，分类数量: {args.num_classes}, nf: {args.nf}, n: {args.n_resnet_blocks}")
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    # ---------------------------------------------------------
    
    # ----------- 构建模型参数文件路径 (根据模型类型) -----------
    # args.model_path 现在是模型文件所在的目录
    model_filename = f"face_model_{args.model_type}.pdparams"
    actual_model_path = os.path.join(args.model_path, model_filename)
    # ---------------------------------------------------------

    # 加载模型参数
    if os.path.exists(actual_model_path):
        state_dict = paddle.load(actual_model_path)
        model.set_state_dict(state_dict)
        print(f"模型加载成功: {actual_model_path}")
    else:
        print(f"错误: 找不到模型文件 {actual_model_path}")
        print(f"请确保已使用 'python train.py --model_type {args.model_type} ...' 完成训练，并且模型保存在指定目录。")
        return
    
    model.eval() # 设置为评估模式
    
    # 加载标签映射文件 (readme.json)
    label_dict = {}
    if not os.path.exists(args.label_file):
        print(f"警告: 找不到标签文件 {args.label_file}。将只输出类别ID。")
    else:
        try:
            with open(args.label_file, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
                # 从readme.json的class_detail中构建 {label_id: class_name} 的映射
                # 注意：readme.json中的class_label是从1开始的，而模型输出的label通常从0开始
                # CreateDataList.py 写入的label是从0开始的，这里假设label_file里的label是0-indexed
                for class_info in label_data.get('class_detail', []):
                    # readme.json中的class_label可能是1-indexed, 需要确认CreateDataList.py的实现
                    # 假设CreateDataList.py生成的label是0-indexed
                    label_id = class_info.get('class_label') # CreateDataList.py中class_label从0开始递增
                    class_name = class_info.get('class_name')
                    if label_id is not None and class_name is not None:
                         label_dict[label_id] = class_name 
            if not label_dict:
                print(f"警告: 标签文件 {args.label_file} 未能成功解析出标签映射。")
        except Exception as e:
            print(f"错误: 读取或解析标签文件 {args.label_file} 失败: {e}")

    # 处理输入图像
    try:
        img_tensor_np = process_image(args.image_path, args.image_size)
    except FileNotFoundError as e:
        print(e)
        return
    except Exception as e:
        print(f"处理图像 {args.image_path} 时发生错误: {e}")
        return
        
    img_tensor = paddle.to_tensor(img_tensor_np)
    
    # 执行推理
    with paddle.no_grad(): # 推理时不需要计算梯度
        features, logits = model(img_tensor)
        # 对logits应用softmax获取概率分布
        probs = paddle.nn.functional.softmax(logits, axis=1).numpy()
    
    # 获取预测结果
    pred_label_id = np.argmax(probs[0])  # 获取概率最高的类别ID (0-indexed)
    pred_score = probs[0][pred_label_id] # 获取对应的置信度
    
    # 输出预测结果
    predicted_class_name = label_dict.get(pred_label_id, f"未知ID:{pred_label_id}")
    print(f"预测的人脸类别: {predicted_class_name}, 置信度: {pred_score:.4f}")
    if pred_label_id not in label_dict and os.path.exists(args.label_file):
        print(f"注意: 预测的类别ID {pred_label_id} 在标签文件 {args.label_file} 中找不到对应的名称。")
    
    # 如果需要可视化结果
    if args.visualize:
        try:
            # 读取原始图像用于显示
            img_display = cv2.imread(args.image_path)
            if img_display is None:
                print(f"警告: 无法读取图像 {args.image_path} 进行可视化。")
                return

            # BGR转RGB以便matplotlib正确显示
            img_rgb_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(8, 6))
            plt.imshow(img_rgb_display)
            
            # 设置标题为预测结果
            result_text = f"{predicted_class_name} ({pred_score:.4f})"
            plt.title(result_text, fontproperties="SimHei") # 指定中文字体，确保能显示中文
            plt.axis('off') # 关闭坐标轴
            
            # 创建保存结果的目录 (如果不存在)
            result_dir = "results"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
                print(f"创建结果保存目录: {result_dir}")
                
            # 保存结果图像
            base_filename = os.path.basename(args.image_path)
            result_image_filename = f"recognition_{args.model_type}_{base_filename}"
            result_file_path = os.path.join(result_dir, result_image_filename)
            plt.savefig(result_file_path)
            print(f"结果图像已保存至: {result_file_path}")
            
            # 尝试显示图像 (如果环境支持)
            plt.show()
        except ImportError:
            print("警告: matplotlib.pyplot 未能正确导入或缺少中文字体支持，无法显示图像，但结果已保存。")
            print("如果需要显示中文，请确保已安装中文字体，如'SimHei'，并配置matplotlib。")
        except Exception as e:
            print(f"可视化过程中发生错误: {e}。结果图像可能已保存。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='人脸识别推理脚本')
    
    # 输入和输出相关参数
    parser.add_argument('--image_path', type=str, required=True, help='待识别的输入图像路径')
    parser.add_argument('--model_path', type=str, default='model', help='训练好的模型文件所在的目录路径') # 改为目录
    parser.add_argument('--label_file', type=str, default='data/face/readme.json', help='类别标签与名称映射的json文件路径 (通常是数据集的readme.json)')
    
    # 模型和推理参数
    parser.add_argument('--model_type', type=str, default='vgg', choices=['vgg', 'resnet'], help='选择模型类型: vgg 或 resnet')
    parser.add_argument('--image_size', type=int, default=64, help='输入图像预处理后的统一大小 (需与训练时一致)')
    parser.add_argument('--num_classes', type=int, default=5, help='模型的分类数量 (需与训练时一致)') # 推理时主要用于模型结构定义
    parser.add_argument('--use_gpu', type=bool, default=False, help='是否使用GPU进行推理') # 建议使用 action='store_true'
    parser.add_argument('--visualize', type=bool, default=True, help='是否可视化识别结果并保存图像') # 建议使用 action='store_true'

    # ResNet特定参数 (仅当 model_type='resnet' 时有效，主要用于模型结构定义)
    parser.add_argument('--nf', type=int, default=32, help='ResNet初始卷积核数量 (nf)')
    parser.add_argument('--n_resnet_blocks', type=int, default=3, help='ResNet每个残差块组中BasicBlock的数量 (n)')

    args = parser.parse_args()
    
    # 解决matplotlib中文显示问题的一个常见方法是指定字体
    # 但更通用的方法是用户自行配置matplotlib的字体设置
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为SimHei
        plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题
    except Exception as e:
        print(f"设置matplotlib中文字体失败: {e}。如果标题中文显示乱码，请手动配置matplotlib。")

    print(f"开始推理图像: {args.image_path} 使用 {args.model_type} 模型")
    infer(args) 