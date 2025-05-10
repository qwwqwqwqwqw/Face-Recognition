# coding:utf-8
import os
import argparse
import paddle
import paddle.nn as nn
import paddle.optimizer as optimizer
from vgg import VGGFace
import MyReader

def continue_train(args):
    """从检查点继续训练"""
    # 设置随机种子
    paddle.seed(args.seed)
    
    # 设置设备
    if args.use_gpu and paddle.is_compiled_with_cuda():
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')
    
    # 检查点路径
    checkpoint_path = os.path.join(os.path.dirname(args.model_path), "checkpoint.pdparams")
    
    # 检查检查点是否存在
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到检查点文件 {checkpoint_path}")
        print("请先使用 train.py 进行训练并生成检查点")
        return
    
    # 加载检查点
    checkpoint = paddle.load(checkpoint_path)
    
    # 检查检查点格式
    required_keys = ['model', 'optimizer', 'lr_scheduler', 'epoch', 'best_acc', 'global_step', 'args']
    for key in required_keys:
        if key not in checkpoint:
            print(f"错误: 检查点文件缺少关键信息 '{key}'")
            return
    
    # 从检查点获取训练参数
    ckpt_args = checkpoint['args']
    num_classes = ckpt_args['num_classes']
    image_size = ckpt_args['image_size']
    dropout_rate = ckpt_args.get('dropout_rate', 0.5)  # 使用默认值0.5防止旧检查点缺少此字段
    
    print(f"检查点信息: 类别数量={num_classes}, 图像大小={image_size}")
    
    # 数据集路径
    train_list = os.path.join(args.data_dir, args.class_name, "trainer.list")
    test_list = os.path.join(args.data_dir, args.class_name, "test.list")
    
    # 创建数据加载器
    train_loader = MyReader.create_dataloader(
        train_list, 
        image_size=image_size,  # 使用检查点中的图像大小
        batch_size=args.batch_size, 
        mode='train'
    )
    
    test_loader = MyReader.create_dataloader(
        test_list, 
        image_size=image_size,  # 使用检查点中的图像大小
        batch_size=args.batch_size, 
        mode='test'
    )
    
    # 创建模型 - 使用检查点中的参数
    model = VGGFace(num_classes=num_classes, dropout_rate=dropout_rate)
    
    # 加载模型权重
    model.set_state_dict(checkpoint['model'])
    print("成功加载模型权重")
    
    # 获取训练状态
    last_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    global_step = checkpoint['global_step']
    
    # 打印恢复信息
    print(f"从epoch {last_epoch + 1}继续训练，当前最佳准确率={best_acc:.4f}")
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器 - 初始创建
    lr_scheduler = optimizer.lr.CosineAnnealingDecay(
        learning_rate=args.learning_rate, 
        T_max=args.epochs)
    
    # 定义优化器
    opt = optimizer.Momentum(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        momentum=0.9,
        weight_decay=5e-4)
    
    # 恢复优化器和调度器状态
    opt.set_state_dict(checkpoint['optimizer'])
    lr_scheduler.set_state_dict(checkpoint['lr_scheduler'])
    print("成功加载优化器和学习率调度器状态")
    
    # 确保模型保存目录存在
    model_dir = os.path.dirname(args.model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 从最后一个epoch继续训练
    start_epoch = last_epoch + 1
    
    # 训练模型
    for epoch in range(start_epoch, args.epochs):
        # 训练模式
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for batch_id, data in enumerate(train_loader):
            images, labels = data
            
            # 前向传播
            features, logits = model(images)
            loss = criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            opt.step()
            opt.clear_grad()
            
            # 更新学习率
            lr_scheduler.step()
            
            # 计算准确率
            pred = paddle.argmax(logits, axis=1)
            correct += (pred == labels).sum().item()
            total_samples += labels.shape[0]
            
            # 累计损失
            total_loss += loss.item()
            
            # 打印训练信息
            if batch_id % 100 == 0:
                print(f"Epoch {epoch}/{args.epochs}, Batch {batch_id}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Accuracy: {correct/total_samples:.4f}, "
                      f"LR: {opt.get_lr():.6f}")
            
            global_step += 1
        
        # 计算训练集平均损失和准确率
        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / total_samples
        
        print(f"Epoch {epoch}/{args.epochs} Training: "
              f"Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
        
        # 测试模式
        model.eval()
        test_correct = 0
        test_total = 0
        
        with paddle.no_grad():
            for data in test_loader:
                images, labels = data
                features, logits = model(images)
                pred = paddle.argmax(logits, axis=1)
                test_correct += (pred == labels).sum().item()
                test_total += labels.shape[0]
        
        test_acc = test_correct / test_total
        print(f"Epoch {epoch}/{args.epochs} Test: Accuracy: {test_acc:.4f}")
        
        # 更新检查点
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc,
            'global_step': global_step,
            'args': {
                'num_classes': num_classes,
                'image_size': image_size,
                'dropout_rate': dropout_rate,
                'batch_size': args.batch_size
            }
        }
        
        # 保存最新检查点
        paddle.save(checkpoint, checkpoint_path)
        print(f"检查点已更新")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            # 保存模型
            paddle.save(model.state_dict(), args.model_path)
            print(f"最佳模型已保存，epoch {epoch}，准确率 {best_acc:.4f}")
    
    print(f"继续训练完成。最佳准确率: {best_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='继续训练人脸识别模型')
    parser.add_argument('--data_dir', type=str, default='data', 
                        help='数据集根目录')
    parser.add_argument('--class_name', type=str, default='face', 
                        help='分类名称')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='批大小')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='总训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.0005, 
                        help='初始学习率（可选调整）')
    parser.add_argument('--use_gpu', type=bool, default=False, 
                        help='是否使用GPU')
    parser.add_argument('--model_path', type=str, 
                        default='model/face_model.pdparams', 
                        help='模型保存/加载路径')
    parser.add_argument('--seed', type=int, default=42, 
                        help='随机种子')
    
    args = parser.parse_args()
    
    print("开始继续训练...")
    print(f"批大小: {args.batch_size}, 目标训练轮数: {args.epochs}")
    
    continue_train(args)