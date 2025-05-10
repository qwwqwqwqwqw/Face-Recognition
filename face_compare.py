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
from resnet_new import ResNetFace # 导入新版ResNet模型
from infer import process_image # 从infer模块导入统一的图像处理函数

def extract_feature(model, img_tensor):
    """
    使用给定模型提取图像张量的特征
    Args:
        model (paddle.nn.Layer): 已加载参数的PaddlePaddle模型
        img_tensor (paddle.Tensor): 预处理后的图像张量，形状如 (1, 3, H, W)
    Returns:
        numpy.ndarray: 提取到的特征向量
    """
    model.eval() # 确保模型处于评估模式
    with paddle.no_grad(): # 推理时不需要计算梯度
        feature, _ = model(img_tensor) # 模型返回 (feature, logits)，这里只需要feature
    return feature.numpy()

def compute_similarity(feature1, feature2):
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
    similarity = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
    return similarity

def show_comparison_result(img1_path, img2_path, similarity, model_type, threshold=0.8):
    """
    可视化人脸对比结果并保存图像
    Args:
        img1_path (str): 第一张图像的路径
        img2_path (str): 第二张图像的路径
        similarity (float): 计算得到的相似度得分
        model_type (str): 使用的模型类型 ('vgg' 或 'resnet')，用于结果文件名
        threshold (float): 判断是否为同一人的相似度阈值
    """
    # 读取图像
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print(f"错误: 无法读取图像进行对比可视化 (img1: {img1_path}, img2: {img2_path})")
        return

    # BGR转换为RGB以便matplotlib正确显示
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 6))
    
    # 显示第一张图像
    plt.subplot(1, 2, 1)
    plt.imshow(img1_rgb)
    plt.title("图像1", fontproperties="SimHei")
    plt.axis('off')
    
    # 显示第二张图像
    plt.subplot(1, 2, 2)
    plt.imshow(img2_rgb)
    plt.title("图像2", fontproperties="SimHei")
    plt.axis('off')
    
    # 判断是否为同一个人
    is_same_person = similarity >= threshold
    judgment_text = "同一个人" if is_same_person else "不同人"
    title_color = 'green' if is_same_person else 'red'
    
    # 设置整体标题显示结果
    result_text = f"相似度: {similarity:.4f}\n判断: {judgment_text} (阈值: {threshold})"
    plt.suptitle(result_text, fontsize=16, color=title_color, fontproperties="SimHei")
    
    # 创建保存结果的目录
    result_dir = "results"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"创建结果保存目录: {result_dir}")
    
    # 生成结果文件名 (包含模型类型)
    base1 = os.path.basename(img1_path).split('.')[0]
    base2 = os.path.basename(img2_path).split('.')[0]
    result_filename = f"compare_{model_type}_{base1}_vs_{base2}.png"
    result_file_path = os.path.join(result_dir, result_filename)
    
    plt.savefig(result_file_path)
    print(f"对比结果图像已保存至: {result_file_path}")
    
    # 尝试显示图像 (如果环境支持)
    try:
        plt.show()
    except Exception as e:
        print(f"无法显示对比图像 (可能在无头服务器上运行)，但结果已保存: {e}")

def compare_faces(args):
    """人脸对比主函数"""
    # 检查输入图像文件是否存在
    if not os.path.exists(args.img1):
        print(f"错误: 图像1不存在: {args.img1}")
        return
    if not os.path.exists(args.img2):
        print(f"错误: 图像2不存在: {args.img2}")
        return
    
    # 设置运行设备 (GPU或CPU)
    if args.use_gpu and paddle.is_compiled_with_cuda():
        paddle.set_device('gpu')
        print("使用 GPU 进行人脸对比")
    else:
        paddle.set_device('cpu')
        print("使用 CPU 进行人脸对比")

    # -------------------- 模型选择与实例化 --------------------
    if args.model_type == 'vgg':
        # 注意：VGGFace的num_classes在对比任务中不直接使用，但定义模型时需要
        # 可以使用一个默认值或与训练时相似的值。提取的特征维度与此无关。
        model = VGGFace(num_classes=args.num_classes, dropout_rate=0.5)
        print(f"使用 VGG 模型进行对比 (num_classes设置为 {args.num_classes} 用于模型结构定义)")
    elif args.model_type == 'resnet':
        model = ResNetFace(num_classes=args.num_classes, nf=args.nf, n=args.n_resnet_blocks)
        print(f"使用 ResNet 模型进行对比 (num_classes={args.num_classes}, nf={args.nf}, n={args.n_resnet_blocks})")
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    # ---------------------------------------------------------

    # ----------- 构建模型参数文件路径 (根据模型类型) -----------
    # args.model_path 现在是模型文件所在的目录
    model_filename = f"face_model_{args.model_type}.pdparams"
    actual_model_path = os.path.join(args.model_path, model_filename)
    # ---------------------------------------------------------

    # 加载模型参数
    if not os.path.exists(actual_model_path):
        print(f"错误: 找不到模型文件 {actual_model_path}")
        print(f"请确保已使用 'python train.py --model_type {args.model_type} ...' 完成训练。")
        return
    
    try:
        state_dict = paddle.load(actual_model_path)
        model.set_state_dict(state_dict)
        print(f"模型加载成功: {actual_model_path}")
    except Exception as e:
        print(f"加载模型参数 {actual_model_path} 失败: {e}")
        return
        
    model.eval() # 设置为评估模式
    
    # 处理两张图像
    try:
        img1_tensor_np = process_image(args.img1, args.image_size)
        img2_tensor_np = process_image(args.img2, args.image_size)
    except FileNotFoundError as e:
        print(e)
        return
    except Exception as e:
        print(f"处理图像时发生错误: {e}")
        return

    img1_tensor = paddle.to_tensor(img1_tensor_np)
    img2_tensor = paddle.to_tensor(img2_tensor_np)
    
    # 提取特征
    feature1 = extract_feature(model, img1_tensor)
    feature2 = extract_feature(model, img2_tensor)
    
    # 计算相似度
    similarity = compute_similarity(feature1, feature2)
    
    # 判断结果并输出
    is_same_person = similarity >= args.threshold
    judgment = "是同一个人" if is_same_person else "不是同一个人"
    print(f"图像1: {args.img1}")
    print(f"图像2: {args.img2}")
    print(f"模型类型: {args.model_type}")
    print(f"计算得到的相似度: {similarity:.4f}")
    print(f"判断结果 (阈值 {args.threshold}): {judgment}")
    
    # 可视化结果 (如果启用)
    if args.visualize:
        show_comparison_result(args.img1, args.img2, similarity, args.model_type, args.threshold)
    
    return similarity, is_same_person

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='人脸对比工具脚本')
    
    # 输入图像参数
    parser.add_argument('--img1', type=str, required=True, help='第一张待对比的人脸图像路径')
    parser.add_argument('--img2', type=str, required=True, help='第二张待对比的人脸图像路径')
    
    # 模型和路径参数
    parser.add_argument('--model_path', type=str, default='model', help='训练好的模型文件所在的目录路径') # 改为目录
    parser.add_argument('--model_type', type=str, default='vgg', choices=['vgg', 'resnet'], help='选择模型类型: vgg 或 resnet')
    parser.add_argument('--image_size', type=int, default=64, help='输入图像预处理后的统一大小 (需与训练时一致)')
    # num_classes在对比时仅用于模型结构定义，不影响特征提取，但需与模型保存时的结构匹配
    parser.add_argument('--num_classes', type=int, default=5, help='模型的分类数量 (用于模型结构定义，需与训练时一致)')
    
    # ResNet特定参数 (仅当 model_type='resnet' 时有效，用于模型结构定义)
    parser.add_argument('--nf', type=int, default=32, help='ResNet初始卷积核数量 (nf)')
    parser.add_argument('--n_resnet_blocks', type=int, default=3, help='ResNet每个残差块组中BasicBlock的数量 (n)')

    # 对比和输出参数
    parser.add_argument('--threshold', type=float, default=0.8, help='判断为同一人的相似度阈值')
    parser.add_argument('--use_gpu', type=bool, default=False, help='是否使用GPU进行特征提取') # 建议使用 action='store_true'
    parser.add_argument('--visualize', type=bool, default=True, help='是否可视化对比结果并保存图像') # 建议使用 action='store_true'
    
    args = parser.parse_args()

    # 尝试设置matplotlib中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"设置matplotlib中文字体失败: {e}。如果标题中文显示乱码，请手动配置matplotlib。")

    print(f"开始对比图像: \n1. {args.img1}\n2. {args.img2}")
    print(f"使用 {args.model_type} 模型进行特征提取")
    compare_faces(args)