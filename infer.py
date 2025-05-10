# coding:utf-8
import os
import cv2
import argparse
import numpy as np
import paddle
from vgg import VGGFace
import json
import matplotlib.pyplot as plt

def process_image(img_path, size=64):
    """
    处理图像，调整大小并进行预处理
    """
    # 读取图像
    img = cv2.imread(img_path)
    # 调整大小
    img = cv2.resize(img, (size, size))
    # 转换为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 转换为float32，并归一化到[0, 1]
    img = img.astype('float32') / 255.0
    # 标准化
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img = (img.transpose((2, 0, 1)) - mean) / std
    # 添加批维度
    img = np.expand_dims(img, axis=0)
    return img

def infer(args):
    """推理函数"""
    # 设置设备
    if args.use_gpu and paddle.is_compiled_with_cuda():
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')
    
    # 创建模型
    model = VGGFace(num_classes=args.num_classes, dropout_rate=0.5)
    
    # 加载模型参数
    if os.path.exists(args.model_path):
        state_dict = paddle.load(args.model_path)
        model.set_state_dict(state_dict)
        print(f"模型加载成功: {args.model_path}")
    else:
        print(f"找不到模型文件: {args.model_path}")
        return
    
    # 设置为评估模式
    model.eval()
    
    # 加载标签映射
    label_dict = {}
    with open(args.label_file, 'r', encoding='utf-8') as f:
        label_data = json.load(f)
        for class_info in label_data.get('class_detail', []):
            label_dict[class_info['class_label']] = class_info['class_name']
    
    # 处理输入图像
    img_tensor = process_image(args.image_path, args.image_size)
    img_tensor = paddle.to_tensor(img_tensor)
    
    # 推理
    with paddle.no_grad():
        features, logits = model(img_tensor)
        probs = paddle.nn.functional.softmax(logits, axis=1).numpy()
    
    # 获取预测结果
    pred_label = np.argmax(probs[0])
    pred_score = probs[0][pred_label]
    
    # 输出预测结果
    if pred_label in label_dict:
        print(f"预测的人脸类别: {label_dict[pred_label]}, 置信度: {pred_score:.4f}")
    else:
        print(f"预测的人脸类别ID: {pred_label}, 置信度: {pred_score:.4f}")
        print("注意: 找不到该ID对应的标签名称")
    
    # 如果需要可视化
    if args.visualize:
        # 读取原始图像
        img = cv2.imread(args.image_path)
        # BGR转RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 创建图形
        plt.figure(figsize=(10, 6))
        plt.imshow(img_rgb)
        
        # 添加预测标签和置信度
        result_text = f"{label_dict.get(pred_label, f'ID:{pred_label}')} {pred_score:.4f}"
        plt.title(result_text)
        
        # 创建保存目录
        result_dir = "results"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            
        # 保存结果图像
        result_file = os.path.join(result_dir, f"recognition_{os.path.basename(args.image_path)}")
        plt.savefig(result_file)
        print(f"结果图像已保存至: {result_file}")
        
        # 显示图像（如果有显示环境）
        try:
            plt.show()
        except Exception as e:
            print(f"无法显示图像，但已保存结果: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='人脸识别推理')
    parser.add_argument('--image_path', type=str, required=True, 
                        help='输入图像路径')
    parser.add_argument('--model_path', type=str, 
                        default='model/face_model.pdparams', 
                        help='模型加载路径')
    parser.add_argument('--label_file', type=str, 
                        default='data/face/readme.json', 
                        help='标签映射文件')
    parser.add_argument('--image_size', type=int, default=64, 
                        help='图像大小')
    parser.add_argument('--num_classes', type=int, default=5, 
                        help='分类数量')
    parser.add_argument('--use_gpu', type=bool, default=False, 
                        help='是否使用GPU')
    parser.add_argument('--visualize', type=bool, default=True, 
                        help='是否可视化结果')
    
    args = parser.parse_args()
    
    print(f"开始推理图像: {args.image_path}")
    infer(args) 