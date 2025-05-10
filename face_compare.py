# coding:utf-8
import os
import cv2
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import paddle
from vgg import VGGFace
from infer import process_image

def extract_feature(model, img_tensor):
    """
    Extract image features
    
    Args:
        model: Model
        img_tensor: Image tensor
        
    Returns:
        Feature vector
    """
    with paddle.no_grad():
        feature, _ = model(img_tensor)
    return feature.numpy()

def compute_similarity(feature1, feature2):
    """
    Calculate cosine similarity between two feature vectors
    
    Args:
        feature1: Feature vector 1
        feature2: Feature vector 2
        
    Returns:
        Similarity score (between 0-1)
    """
    # 确保向量形状正确并计算余弦相似度
    return np.dot(feature1.flatten(), feature2.flatten()) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))

def show_comparison_result(img1_path, img2_path, similarity, threshold=0.8):
    """
    Visualize comparison results
    
    Args:
        img1_path: Image 1 path
        img2_path: Image 2 path
        similarity: Similarity score
        threshold: Judgment threshold
    """
    # 读取图像
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # BGR转RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # 创建图形
    plt.figure(figsize=(10, 5))
    
    # 显示图像1
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title("Image 1")
    plt.axis('off')
    
    # 显示图像2
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title("Image 2")
    plt.axis('off')
    
    # 判断是否为同一人
    is_same_person = similarity >= threshold
    
    # 设置标题颜色
    title_color = 'green' if is_same_person else 'red'
    
    # 显示结果
    result_text = f"Similarity: {similarity:.4f}\nJudgment: {'Same Person' if is_same_person else 'Different Person'}"
    plt.suptitle(result_text, fontsize=16, color=title_color)
    
    # 保存结果
    result_dir = "results"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 生成结果文件名
    base1 = os.path.basename(img1_path).split('.')[0]
    base2 = os.path.basename(img2_path).split('.')[0]
    result_file = os.path.join(result_dir, f"compare_{base1}_{base2}.png")
    
    plt.savefig(result_file)
    print(f"Comparison result saved to: {result_file}")
    
    # 显示图形
    plt.tight_layout()
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display image, but result saved: {e}")

def compare_faces(args):
    """
    Face comparison main function
    """
    # 检查文件是否存在
    if not os.path.exists(args.img1):
        print(f"Error: Image 1 not found: {args.img1}")
        return
    
    if not os.path.exists(args.img2):
        print(f"Error: Image 2 not found: {args.img2}")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    # 设置设备
    if args.use_gpu and paddle.is_compiled_with_cuda():
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')
    
    # 创建模型
    model = VGGFace(num_classes=args.num_classes, dropout_rate=0.5)
    
    # 加载模型参数
    model_state_dict = paddle.load(args.model_path)
    model.set_state_dict(model_state_dict)
    model.eval()
    
    # 处理图像
    img1_tensor = paddle.to_tensor(process_image(args.img1, args.image_size))
    img2_tensor = paddle.to_tensor(process_image(args.img2, args.image_size))
    
    # 提取特征
    feature1 = extract_feature(model, img1_tensor)
    feature2 = extract_feature(model, img2_tensor)
    
    # 计算相似度
    similarity = compute_similarity(feature1, feature2)
    
    # 判断结果
    is_same_person = similarity >= args.threshold
    print(f"Similarity: {similarity:.4f}")
    print(f"Judgment: {'Same Person' if is_same_person else 'Different Person'}")
    
    # 可视化结果
    if args.visualize:
        show_comparison_result(args.img1, args.img2, similarity, args.threshold)
    
    return similarity, is_same_person

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Comparison Tool')
    parser.add_argument('--img1', type=str, required=True, 
                        help='Path to first face image')
    parser.add_argument('--img2', type=str, required=True, 
                        help='Path to second face image')
    parser.add_argument('--model_path', type=str, 
                        default='model/face_model.pdparams', 
                        help='Model loading path')
    parser.add_argument('--image_size', type=int, default=64, 
                        help='Image size')
    parser.add_argument('--num_classes', type=int, default=5, 
                        help='Number of classes')
    parser.add_argument('--threshold', type=float, default=0.8, 
                        help='Threshold for same person judgment')
    parser.add_argument('--use_gpu', type=bool, default=False, 
                        help='Whether to use GPU')
    parser.add_argument('--visualize', type=bool, default=True, 
                        help='Whether to visualize results')
    
    args = parser.parse_args()
    
    print(f"Starting comparison of images: \n1. {args.img1}\n2. {args.img2}")
    compare_faces(args)