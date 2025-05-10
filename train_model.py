# coding:utf-8
import os
import sys
import paddle.v2 as paddle
from MyReader import MyReader
from vgg import vgg_bn_drop
from resnet import resnet_face
import argparse

class FaceTrainer:
    """人脸识别训练类"""
    
    def __init__(self, use_gpu=False):
        # 初始化PaddlePaddle
        paddle.init(use_gpu=use_gpu, trainer_count=2)
        
    def get_model(self, model_type, datadim, class_num):
        """
        获取模型
        Args:
            model_type: 模型类型，'vgg'或'resnet'
            datadim: 数据维度
            class_num: 分类数量
        Returns:
            特征输出层和分类输出层
        """
        if model_type.lower() == 'vgg':
            return vgg_bn_drop(datadim=datadim, type_size=class_num)
        elif model_type.lower() == 'resnet':
            # 图像数据层
            image = paddle.layer.data(
                name="image", type=paddle.data_type.dense_vector(datadim))
            return resnet_face(ipt=image, class_dim=class_num)
        else:
            raise ValueError("不支持的模型类型: %s" % model_type)
    
    def get_trainer(self, model_type, datadim, class_num, parameters_path=None):
        """
        创建训练器
        """
        # 标签数据层
        label = paddle.layer.data(
            name="label", type=paddle.data_type.integer_value(class_num))
        
        # 获取特征提取器和分类器
        feature, output = self.get_model(model_type, datadim, class_num)
        
        # 分类损失函数
        cost = paddle.layer.classification_cost(input=output, label=label)
        
        # 获取参数
        if parameters_path and os.path.exists(parameters_path):
            # 加载已有参数
            with open(parameters_path, 'r') as f:
                parameters = paddle.parameters.Parameters.from_tar(f)
            print("加载已有参数：%s" % parameters_path)
        else:
            # 创建新参数
            parameters = paddle.parameters.create(cost)
            print("创建新参数")
        
        # 定义优化方法
        optimizer = paddle.optimizer.Momentum(
            momentum=0.9,
            regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128),
            learning_rate=0.001 / 128,
            learning_rate_decay_a=0.1,
            learning_rate_decay_b=128000 * 35,
            learning_rate_schedule="discexp")
        
        # 创建训练器
        trainer = paddle.trainer.SGD(
            cost=cost, parameters=parameters, update_equation=optimizer)
        
        return trainer
    
    def train(self, trainer, num_passes, save_model_path, train_list_path, test_list_path, image_size=64):
        """
        开始训练
        """
        # 创建数据读取器
        reader = MyReader(imageSize=image_size)
        train_reader = reader.train_reader(train_list=train_list_path)
        test_reader = reader.test_reader(test_list=test_list_path)
        
        # 确保保存模型的目录存在
        model_dir = os.path.dirname(save_model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 批处理数据
        batch_reader = paddle.batch(
            paddle.reader.shuffle(train_reader, buf_size=10000),
            batch_size=128)
        
        # 喂入数据映射
        feeding = {"image": 0, "label": 1}
        
        # 定义训练事件处理函数
        def event_handler(event):
            if isinstance(event, paddle.event.EndIteration):
                if event.batch_id % 100 == 0:
                    print("\n轮次 %d, 批次 %d, 损失 %f, 错误率 %s" % (
                        event.pass_id, event.batch_id, event.cost, 
                        event.metrics['classification_error_evaluator']))
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()
            
            # 每完成一轮训练
            if isinstance(event, paddle.event.EndPass):
                # 保存模型参数
                with open(save_model_path, 'w') as f:
                    trainer.save_parameter_to_tar(f)
                
                # 测试模型效果
                result = trainer.test(
                    reader=paddle.batch(test_reader, batch_size=128),
                    feeding=feeding)
                
                print("\n测试 轮次 %d, 分类错误率 %s" % (
                    event.pass_id, result.metrics['classification_error_evaluator']))
        
        # 开始训练
        trainer.train(
            reader=batch_reader,
            num_passes=num_passes,
            event_handler=event_handler,
            feeding=feeding)
        
        print("\n训练完成！模型已保存到：%s" % save_model_path)

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='人脸识别训练程序')
    parser.add_argument('--model', type=str, default='vgg', help='模型类型: vgg 或 resnet')
    parser.add_argument('--class_num', type=int, default=100, help='人脸类别数量')
    parser.add_argument('--image_size', type=int, default=64, help='图像大小')
    parser.add_argument('--passes', type=int, default=50, help='训练轮数')
    parser.add_argument('--gpu', action='store_true', help='是否使用GPU')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的模型路径')
    args = parser.parse_args()
    
    # 设置类别数量、图像大小等参数
    class_num = args.class_num
    image_size = args.image_size
    model_type = args.model
    datadim = 3 * image_size * image_size
    
    # 数据集路径
    data_dir = "data/face"
    train_list = "%s/trainer.list" % data_dir
    test_list = "%s/test.list" % data_dir
    
    # 模型保存路径
    if model_type.lower() == 'vgg':
        model_path = "model/vgg_face_model.tar"
    else:
        model_path = "model/resnet_face_model.tar"
    
    # 创建训练器
    trainer = FaceTrainer(use_gpu=args.gpu)
    
    # 获取训练器
    paddle_trainer = trainer.get_trainer(
        model_type=model_type,
        datadim=datadim,
        class_num=class_num,
        parameters_path=args.resume)
    
    # 开始训练
    print("开始训练：使用%s模型，类别数量：%d，图像大小：%d" % (model_type, class_num, image_size))
    trainer.train(
        trainer=paddle_trainer,
        num_passes=args.passes,
        save_model_path=model_path,
        train_list_path=train_list,
        test_list_path=test_list,
        image_size=image_size) 