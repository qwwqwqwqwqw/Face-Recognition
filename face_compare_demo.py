# coding:utf-8
import cv2
import numpy as np
import paddle.v2 as paddle
import matplotlib.pyplot as plt
from vgg import vgg_bn_drop
from resnet import resnet_face
import argparse
import os

def get_parameters(parameters_path):
    """
    加载模型参数
    """
    with open(parameters_path, 'r') as f:
        parameters = paddle.parameters.Parameters.from_tar(f)
    return parameters

def get_feature_extractor(model_type, parameters, datadim, class_num):
    """
    获取特征提取器
    """
    if model_type.lower() == 'vgg':
        feature, _ = vgg_bn_drop(datadim=datadim, type_size=class_num)
    elif model_type.lower() == 'resnet':
        image = paddle.layer.data(name="image", type=paddle.data_type.dense_vector(datadim))
        feature, _ = resnet_face(ipt=image, class_dim=class_num)
    else:
        raise ValueError("不支持的模型类型：%s" % model_type)
    
    return feature

def process_image(img_path, size):
    """
    处理图像
    """
    img = paddle.image.load_image(img_path)
    img = paddle.image.simple_transform(img, 70, size, False)
    return img.flatten().astype('float32')

def extract_feature(img_path, feature_extractor, parameters, size):
    """
    提取图像特征
    """
    img_data = [(process_image(img_path, size),)]
    feature = paddle.inference.Inference(
        output_layer=feature_extractor, 
        parameters=parameters).infer(input=img_data)[0]
    return feature

def compute_similarity(feature1, feature2):
    """
    计算两个特征向量的余弦相似度
    """
    similarity = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    return similarity

def display_result(img1_path, img2_path, similarity, threshold):
    """
    可视化展示对比结果
    """
    # 读取图像
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # BGR转RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # 创建画布
    plt.figure(figsize=(10, 5))
    
    # 显示图像1
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title("图像1")
    plt.axis('off')
    
    # 显示图像2
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title("图像2")
    plt.axis('off')
    
    # 显示相似度结果
    is_same = similarity >= threshold
    result_text = "相似度: {:.4f}\n".format(similarity)
    result_text += "判断: " + ("同一个人" if is_same else "不是同一个人")
    
    plt.suptitle(result_text, fontsize=16, color='blue' if is_same else 'red')
    plt.tight_layout()
    plt.show()

def compare_faces(img1_path, img2_path, model_path, model_type, class_num, image_size, threshold):
    """
    人脸对比主函数
    """
    # 初始化PaddlePaddle
    paddle.init(use_gpu=False, trainer_count=1)
    
    # 数据维度
    datadim = 3 * image_size * image_size
    
    # 加载参数
    parameters = get_parameters(model_path)
    
    # 获取特征提取器
    feature_extractor = get_feature_extractor(
        model_type, parameters, datadim, class_num)
    
    # 提取特征
    feature1 = extract_feature(img1_path, feature_extractor, parameters, image_size)
    feature2 = extract_feature(img2_path, feature_extractor, parameters, image_size)
    
    # 计算相似度
    similarity = compute_similarity(feature1, feature2)
    
    # 显示结果
    display_result(img1_path, img2_path, similarity, threshold)
    
    return similarity

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='人脸对比演示程序')
    parser.add_argument('--img1', type=str, required=True, help='第一张人脸图片路径')
    parser.add_argument('--img2', type=str, required=True, help='第二张人脸图片路径')
    parser.add_argument('--model', type=str, default='vgg', help='模型类型: vgg 或 resnet')
    parser.add_argument('--class_num', type=int, default=100, help='类别数量')
    parser.add_argument('--image_size', type=int, default=64, help='图像大小')
    parser.add_argument('--threshold', type=float, default=0.8, help='判断为同一人的阈值')
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.img1):
        print("错误：图片1不存在: %s" % args.img1)
        exit(1)
    
    if not os.path.exists(args.img2):
        print("错误：图片2不存在: %s" % args.img2)
        exit(1)
    
    # 模型路径
    if args.model.lower() == 'vgg':
        model_path = "model/vgg_face_model.tar"
    else:
        model_path = "model/resnet_face_model.tar"
    
    if not os.path.exists(model_path):
        print("错误：模型文件不存在: %s" % model_path)
        print("请先运行训练程序 train_model.py")
        exit(1)
    
    # 比较人脸
    similarity = compare_faces(
        img1_path=args.img1,
        img2_path=args.img2,
        model_path=model_path,
        model_type=args.model,
        class_num=args.class_num,
        image_size=args.image_size,
        threshold=args.threshold)
    
    print("相似度: {:.4f}".format(similarity)) 