# coding:utf-8
import os
import argparse
import paddle
import paddle.nn as nn
import paddle.optimizer as optimizer
from vgg import VGGFace         # 导入VGG模型
from resnet_new import ResNetFace # 导入新版ResNet模型
import MyReader                 # 导入数据读取器

def train(args):
    """训练函数"""
    # 设置随机种子，确保实验可复现
    paddle.seed(args.seed)
    
    # 设置运行设备 (GPU或CPU)
    if args.use_gpu and paddle.is_compiled_with_cuda():
        paddle.set_device('gpu')
        print("使用 GPU 进行训练")
    else:
        paddle.set_device('cpu')
        print("使用 CPU 进行训练")
        
    # 构建数据集文件路径
    train_list = os.path.join(args.data_dir, args.class_name, "trainer.list")
    test_list = os.path.join(args.data_dir, args.class_name, "test.list")
    
    # 创建数据加载器
    train_loader = MyReader.create_dataloader(
        train_list, 
        image_size=args.image_size, 
        batch_size=args.batch_size, 
        mode='train'
    )
    
    test_loader = MyReader.create_dataloader(
        test_list, 
        image_size=args.image_size, 
        batch_size=args.batch_size, 
        mode='test'
    )
    
    # -------------------- 模型选择与实例化 --------------------
    if args.model_type == 'vgg':
        model = VGGFace(num_classes=args.num_classes, dropout_rate=0.5) # VGG特有的dropout_rate
        print(f"使用 VGG 模型进行训练，分类数量: {args.num_classes}")
    elif args.model_type == 'resnet':
        # ResNetFace实例化，nf和n可以使用默认值或从命令行参数传入
        model = ResNetFace(num_classes=args.num_classes, nf=args.nf, n=args.n_resnet_blocks)
        print(f"使用 ResNet 模型进行训练，分类数量: {args.num_classes}, nf: {args.nf}, n: {args.n_resnet_blocks}")
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    # ---------------------------------------------------------

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 定义初始训练状态
    start_epoch = 0
    best_acc = 0.0
    global_step = 0
    
    # ----------- 检查点文件路径 (根据模型类型命名) -------------
    # 确保模型保存目录存在
    model_dir = args.model_path # args.model_path 直接作为目录
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"创建模型保存目录: {model_dir}")

    checkpoint_filename = f"checkpoint_{args.model_type}.pdparams"
    checkpoint_path = os.path.join(model_dir, checkpoint_filename)
    # ---------------------------------------------------------
    
    # 学习率调度器 (在加载检查点逻辑之前先定义，以便能够恢复其状态)
    lr_scheduler = optimizer.lr.CosineAnnealingDecay(
        learning_rate=args.learning_rate, T_max=args.epochs
    )
    
    # 定义优化器 (同样在加载检查点逻辑之前定义)
    opt = optimizer.Momentum(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        momentum=0.9,
        weight_decay=5e-4
    )

    # -------------------- 检查点加载逻辑 --------------------
    if args.resume and os.path.exists(checkpoint_path):
        print(f"尝试从检查点 {checkpoint_path} 恢复训练...")
        checkpoint = paddle.load(checkpoint_path)
        
        # 校验检查点中的模型类型与当前参数是否匹配
        ckpt_args = checkpoint.get('args', {})
        ckpt_model_type = ckpt_args.get('model_type')

        if ckpt_model_type == args.model_type:
            model.set_state_dict(checkpoint['model'])
            # 校验和恢复ResNet的nf和n参数 (如果存在于检查点中)
            if args.model_type == 'resnet':
                if ckpt_args.get('nf') != args.nf or ckpt_args.get('n_resnet_blocks') != args.n_resnet_blocks:
                    print(f"警告: 检查点中的ResNet参数 (nf={ckpt_args.get('nf')}, n={ckpt_args.get('n_resnet_blocks')}) "
                          f"与当前参数 (nf={args.nf}, n={args.n_resnet_blocks}) 不匹配。可能导致问题。")
            
            opt.set_state_dict(checkpoint['optimizer'])
            lr_scheduler.set_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            global_step = checkpoint['global_step']
            print(f"成功从检查点恢复。继续从 epoch {start_epoch} 开始，最佳准确率={best_acc:.4f}")
        else:
            print(f"警告: 检查点模型类型 ({ckpt_model_type}) 与当前指定模型类型 ({args.model_type}) 不匹配。将从头开始训练。")
            start_epoch = 0
            best_acc = 0.0
            global_step = 0
            # 如果模型类型不匹配，优化器和学习率调度器也应该使用新创建的，而不是尝试加载不兼容的状态
            lr_scheduler = optimizer.lr.CosineAnnealingDecay(
                learning_rate=args.learning_rate, T_max=args.epochs
            )
            opt = optimizer.Momentum(
                learning_rate=lr_scheduler,
                parameters=model.parameters(),
                momentum=0.9,
                weight_decay=5e-4
            )

    elif args.resume and not os.path.exists(checkpoint_path):
        print(f"警告: 指定了 --resume 但检查点文件 {checkpoint_path} 不存在。将从头开始训练。")
    else:
        print("不使用检查点，从头开始训练。")
    # ---------------------------------------------------------
    
    # ---------- 最佳模型文件路径 (根据模型类型命名) -----------
    best_model_filename = f"face_model_{args.model_type}.pdparams"
    current_best_model_path = os.path.join(model_dir, best_model_filename)
    # ---------------------------------------------------------

    # 训练模型
    print(f"开始训练，总共 {args.epochs} 个 epochs...")
    for epoch in range(start_epoch, args.epochs):
        model.train() # 设置为训练模式
        total_loss = 0
        correct_samples = 0 # 修改变量名以示区分
        total_samples = 0
        
        print(f"--- Epoch {epoch + 1}/{args.epochs} ---")
        for batch_id, data in enumerate(train_loader):
            images, labels = data
            
            features, logits = model(images)
            loss = criterion(logits, labels)
            
            loss.backward()
            opt.step()
            opt.clear_grad()
            
            lr_scheduler.step() # 每个batch后更新学习率
            
            pred = paddle.argmax(logits, axis=1)
            correct_samples += (pred == labels).sum().item()
            total_samples += labels.shape[0]
            
            total_loss += loss.item()
            global_step += 1
            
            if batch_id % 100 == 0:
                current_lr = opt.get_lr()
                print(f"  Batch {batch_id}/{len(train_loader)}, Loss: {loss.item():.4f}, "
                      f"Train Acc: {correct_samples/total_samples:.4f}, LR: {current_lr:.6f}")
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = correct_samples / total_samples
        print(f"Epoch {epoch + 1} Training Summary: Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")
        
        # 测试模式
        model.eval() # 设置为评估模式
        test_correct_samples = 0
        test_total_samples = 0
        
        with paddle.no_grad(): # 测试时不需要计算梯度
            for data in test_loader:
                images, labels = data
                _, logits = model(images) # 测试时通常只关心分类结果
                pred = paddle.argmax(logits, axis=1)
                test_correct_samples += (pred == labels).sum().item()
                test_total_samples += labels.shape[0]
        
        test_acc = test_correct_samples / test_total_samples
        print(f"Epoch {epoch + 1} Test Summary: Accuracy: {test_acc:.4f}")
        
        # -------------------- 保存检查点 --------------------
        checkpoint_content = {
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc,
            'global_step': global_step,
            'args': { # 保存训练时的部分重要参数
                'model_type': args.model_type,
                'num_classes': args.num_classes,
                'image_size': args.image_size,
                'batch_size': args.batch_size,
                # 对于VGG，可以保存dropout_rate；对于ResNet，保存nf和n
                'dropout_rate': 0.5 if args.model_type == 'vgg' else None, 
                'nf': args.nf if args.model_type == 'resnet' else None,
                'n_resnet_blocks': args.n_resnet_blocks if args.model_type == 'resnet' else None
            }
        }
        paddle.save(checkpoint_content, checkpoint_path)
        print(f"检查点已保存到: {checkpoint_path}")
        # -----------------------------------------------------
        
        # -------------------- 保存最佳模型 -------------------
        if test_acc > best_acc:
            best_acc = test_acc
            paddle.save(model.state_dict(), current_best_model_path)
            print(f"最佳模型已更新并保存到: {current_best_model_path} (Epoch {epoch + 1}, Accuracy: {best_acc:.4f})")
        # -----------------------------------------------------
    
    print(f"训练完成。最佳测试准确率: {best_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='人脸识别训练脚本')
    
    # 数据和模型相关参数
    parser.add_argument('--data_dir', type=str, default='data', help='数据集根目录')
    parser.add_argument('--class_name', type=str, default='face', help='数据集中的分类名称 (子目录名)')
    parser.add_argument('--model_path', type=str, default='model', help='模型保存的目录路径') # 改为目录
    parser.add_argument('--model_type', type=str, default='vgg', choices=['vgg', 'resnet'], help='选择模型类型: vgg 或 resnet')
    
    # 训练超参数
    parser.add_argument('--num_classes', type=int, default=5, help='分类数量 (必须与数据集匹配)')
    parser.add_argument('--image_size', type=int, default=64, help='输入图像统一调整后的大小 (像素)')
    parser.add_argument('--batch_size', type=int, default=32, help='训练批大小')
    parser.add_argument('--epochs', type=int, default=50, help='总训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='初始学习率')
    
    # ResNet特定参数 (仅当 model_type='resnet' 时有效)
    parser.add_argument('--nf', type=int, default=32, help='ResNet初始卷积核数量 (nf)')
    parser.add_argument('--n_resnet_blocks', type=int, default=3, help='ResNet每个残差块组中BasicBlock的数量 (n)')

    # 其他设置
    parser.add_argument('--use_gpu', type=bool, default=False, help='是否使用GPU进行训练 (若可用)') # bool类型的argparse参数建议使用action='store_true'
    parser.add_argument('--seed', type=int, default=42, help='随机种子，用于保证实验可复现性')
    parser.add_argument('--resume', action='store_true', help='是否从检查点恢复训练')
    
    args = parser.parse_args()
    
    print("开始训练...")
    if args.resume:
        print(f"将尝试从检查点恢复 {args.model_type} 模型训练")
    else:
        print(f"将开始新的 {args.model_type} 模型训练")
        
    print(f"参数配置: 类别数量={args.num_classes}, 图像大小={args.image_size}, "
          f"批大小={args.batch_size}, 训练轮数={args.epochs}, 模型类型={args.model_type}")
    if args.model_type == 'resnet':
        print(f"ResNet特定参数: nf={args.nf}, n_resnet_blocks={args.n_resnet_blocks}")

    train(args) 