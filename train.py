# coding:utf-8
import os
import argparse
import paddle
import paddle.nn as nn
import paddle.optimizer as optimizer
# import paddle.nn.functional as F # F 已在 resnet_new 和其他地方导入，此处若非直接使用可不显式导入
from vgg import VGGFace         # 导入VGG模型
from resnet_new import ResNetFace, ArcFaceHead # 导入新版ResNet模型 和 ArcFaceHead
import MyReader                 # 导入数据读取器
from config_utils import load_config # 导入配置加载工具

def train(config, cmd_args):
    """训练函数，参数通过config对象传递"""
    # 设置随机种子，确保实验可复现
    paddle.seed(config.seed)
    
    # 设置运行设备 (GPU或CPU)
    if config.use_gpu and paddle.is_compiled_with_cuda():
        paddle.set_device('gpu')
        print("使用 GPU 进行训练")
    else:
        paddle.set_device('cpu')
        print("使用 CPU 进行训练")
        
    # 构建数据集文件路径
    # 使用 config 对象中的参数
    train_list = os.path.join(config.data_dir, config.class_name, "trainer.list")
    test_list = os.path.join(config.data_dir, config.class_name, "test.list")
    
    # 创建数据加载器
    train_loader = MyReader.create_dataloader(
        train_list, 
        image_size=config.image_size, 
        batch_size=config.batch_size, 
        mode='train'
    )
    
    test_loader = MyReader.create_dataloader(
        test_list, 
        image_size=config.image_size, 
        batch_size=config.batch_size, 
        mode='test'
    )
    
    # -------------------- 模型、头部和损失函数的定义与实例化 ---------------------
    backbone = None
    head_module = None
    model_instance = None  # Renamed from 'model' to avoid confusion with config.model
    criterion = None

    if config.model_type == 'vgg':
        vgg_params = config.model.get('vgg_params', {})
        dropout_rate = vgg_params.get('dropout_rate', 0.5)
        model_instance = VGGFace(num_classes=config.num_classes, dropout_rate=dropout_rate)
        criterion = nn.CrossEntropyLoss()
        print(f"使用 VGG 模型进行训练，分类数量: {config.num_classes}, Dropout: {dropout_rate}, 使用 CrossEntropyLoss")
    elif config.model_type == 'resnet':
        resnet_params = config.model.get('resnet_params', {})
        nf_val = resnet_params.get('nf', 32)
        n_blocks_val = resnet_params.get('n_resnet_blocks', 3)
        feat_dim_val = resnet_params.get('feature_dim', 512)

        backbone = ResNetFace(nf=nf_val, n=n_blocks_val, feature_dim=feat_dim_val)
        print(f"使用 ResNet 模型骨干，特征维度: {feat_dim_val}, nf: {nf_val}, n: {n_blocks_val}")

        if config.loss_type == 'arcface':
            arcface_params = config.loss.get('arcface_params', {})
            m1 = arcface_params.get('arcface_m1', 1.0)
            m2 = arcface_params.get('arcface_m2', 0.5)
            m3 = arcface_params.get('arcface_m3', 0.0)
            s = arcface_params.get('arcface_s', 64.0)
            head_module = ArcFaceHead(
                in_features=feat_dim_val, 
                out_features=config.num_classes,
                margin1=m1, margin2=m2, margin3=m3, scale=s
            )
            criterion = None # ArcFaceHead 内部计算损失
            print(f"  L--> 使用 ArcFaceHead，分类数量: {config.num_classes}")
            print(f"       ArcFace参数: m1={m1}, m2={m2}, m3={m3}, s={s}")
        elif config.loss_type == 'cross_entropy':
            head_module = nn.Linear(feat_dim_val, config.num_classes)
            criterion = nn.CrossEntropyLoss()
            print(f"  L--> 使用普通分类头 (Linear) 和 CrossEntropyLoss，分类数量: {config.num_classes}")
        else:
            raise ValueError(f"ResNet 模型不支持的损失函数类型: {config.loss_type}")
    else:
        raise ValueError(f"不支持的模型类型: {config.model_type}")
    # ---------------------------------------------------------------------------

    # 定义初始训练状态
    start_epoch = 0
    best_acc = 0.0
    global_step = 0
    
    # ----------- 检查点文件路径 (根据模型类型命名) ------------- 
    # model_save_dir 来自 config
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
        print(f"创建模型保存目录: {config.model_save_dir}")

    checkpoint_filename = f"checkpoint_{config.model_type}_{config.loss_type if config.model_type == 'resnet' else 'ce'}.pdparams"
    checkpoint_path = os.path.join(config.model_save_dir, checkpoint_filename)
    # ---------------------------------------------------------
    
    # 学习率调度器
    lr_scheduler_params = config.get('lr_scheduler_params', {})
    lr_scheduler = optimizer.lr.CosineAnnealingDecay(
        learning_rate=config.learning_rate, 
        T_max=lr_scheduler_params.get('T_max', config.epochs) 
    )
    
    # 定义优化器
    params_to_optimize = []
    if model_instance: params_to_optimize.extend(model_instance.parameters())
    if backbone: params_to_optimize.extend(backbone.parameters())
    if head_module: params_to_optimize.extend(head_module.parameters())

    optimizer_p = config.get('optimizer_params', {})
    opt_type = config.get('optimizer_type', 'Momentum')

    if opt_type == 'Momentum':
        opt = optimizer.Momentum(
            learning_rate=lr_scheduler,
            parameters=params_to_optimize,
            momentum=optimizer_p.get('momentum', 0.9),
            weight_decay=optimizer_p.get('weight_decay', 0.0005)
        )
    elif opt_type == 'AdamW':
        opt = optimizer.AdamW(
            learning_rate=lr_scheduler,
            parameters=params_to_optimize,
            weight_decay=optimizer_p.get('weight_decay', 0.0001)
        )
    # Add other optimizers like Adam if needed
    else:
        raise ValueError(f"不支持的优化器类型: {opt_type}")
    print(f"使用优化器: {opt_type}，学习率: {config.learning_rate}, WeightDecay: {optimizer_p.get('weight_decay')}")

    # -------------------- 检查点加载逻辑 --------------------
    checkpoint_exists = os.path.exists(checkpoint_path)
    attempt_resume_from_checkpoint = False

    if checkpoint_exists:
        if cmd_args.resume is False: # 用户通过命令行 --no-resume 明确指示不恢复
            print(f"检查点 {checkpoint_path} 存在，但用户通过命令行 (--no-resume) 指示从头开始训练。")
        elif cmd_args.resume is True: # 用户通过命令行 --resume 明确指示恢复
            print(f"检查点 {checkpoint_path} 存在，用户通过命令行 (--resume) 指示恢复训练。将尝试恢复。")
            attempt_resume_from_checkpoint = True
        else: # cmd_args.resume is None (命令行未指定恢复选项)
            if config.resume is False: # 配置文件明确指示不恢复 (resume: false)
                print(f"检查点 {checkpoint_path} 存在，但配置文件指示不恢复 (resume: False)，且命令行未指定 --resume。将从头开始训练。")
            else: # 配置文件指示恢复 (resume: true) 或未指定 (默认为自动恢复逻辑)
                print(f"检查点 {checkpoint_path} 存在，将自动尝试恢复训练 (配置文件 resume: {config.resume}, 命令行未指定)。")
                attempt_resume_from_checkpoint = True
    else: # 检查点文件不存在
        if cmd_args.resume is True: # 用户明确指定 --resume 但文件不存在
            print(f"警告: 用户指定了 --resume 但检查点文件 {checkpoint_path} 不存在。将从头开始训练。")
        else: # 文件不存在，且用户未指定 --resume 或指定了 --no-resume
            print(f"检查点文件 {checkpoint_path} 不存在。将从头开始训练。")

    if not attempt_resume_from_checkpoint:
        start_epoch = 0
        best_acc = 0.0
        global_step = 0
        print("将从 epoch 0，best_acc 0.0 开始新的训练。")

    if attempt_resume_from_checkpoint:
        print(f"尝试从检查点 {checkpoint_path} 恢复训练...")
        try:
            checkpoint = paddle.load(checkpoint_path)
            ckpt_config_dict = checkpoint.get('config', {})
            if not isinstance(ckpt_config_dict, dict):
                ckpt_config_dict = vars(ckpt_config_dict)

            load_weights_successful = True
            ckpt_model_type = ckpt_config_dict.get('model_type')
            ckpt_loss_type = ckpt_config_dict.get('loss_type')

            if ckpt_model_type != config.model_type or \
               (config.model_type == 'resnet' and ckpt_loss_type != config.loss_type):
                print(f"警告: 检查点模型/损失类型 ({ckpt_model_type}/{ckpt_loss_type}) 与当前 ({config.model_type}/{config.loss_type}) 不符。将从头训练。")
                load_weights_successful = False

            if load_weights_successful:
                if config.model_type == 'vgg' and model_instance:
                    if 'model' in checkpoint: model_instance.set_state_dict(checkpoint['model'])
                    else: print("警告: 检查点缺少VGG 'model'权重。"); load_weights_successful = False
                elif config.model_type == 'resnet':
                    if backbone and 'backbone' in checkpoint: backbone.set_state_dict(checkpoint['backbone'])
                    else: print("警告: 检查点缺少ResNet 'backbone'权重。"); load_weights_successful = False
                    if head_module and 'head' in checkpoint: head_module.set_state_dict(checkpoint['head'])
                    elif head_module and 'head' not in checkpoint: print("警告: 当前配置需要ResNet 'head'但检查点中未找到。"); load_weights_successful = False
            else:
                print("权重加载未完全成功或模型/损失类型不匹配，将重置训练状态并从头开始。")
                start_epoch = 0; best_acc = 0.0; global_step = 0
        except Exception as e:
            print(f"从检查点 {checkpoint_path} 恢复失败: {e}。将从头开始训练。")
            start_epoch = 0; best_acc = 0.0; global_step = 0
    # ---------------------------------------------------------
    
    # ---------- 最佳模型文件路径 (根据模型类型命名) ----------- 
    best_model_filename = f"face_model_{config.model_type}_{config.loss_type if config.model_type == 'resnet' else 'ce'}.pdparams"
    current_best_model_path = os.path.join(config.model_save_dir, best_model_filename)
    # ---------------------------------------------------------

    # 训练模型
    print(f"开始训练，总共 {config.epochs} 个 epochs... 从 epoch {start_epoch} 开始")
    for epoch in range(start_epoch, config.epochs):
        if model_instance: model_instance.train()
        if backbone: backbone.train()
        if head_module: head_module.train()
        
        total_loss_val = 0
        correct_samples = 0
        total_samples = 0
        
        print(f"--- Epoch {epoch + 1}/{config.epochs} ---")
        for batch_id, data in enumerate(train_loader):
            images, labels = data
            
            loss_val = None
            current_logits_for_acc = None

            if config.model_type == 'vgg':
                features, logits = model_instance(images)
                loss_val = criterion(logits, labels)
                current_logits_for_acc = logits
            elif config.model_type == 'resnet':
                features = backbone(images)
                if config.loss_type == 'arcface' and head_module:
                    loss_val, softmax_output = head_module(features, labels)
                    current_logits_for_acc = softmax_output
                elif config.loss_type == 'cross_entropy' and head_module:
                    logits = head_module(features) 
                    loss_val = criterion(logits, labels)
                    current_logits_for_acc = logits
            
            if loss_val is not None:
                loss_val.backward()
                opt.step()
                opt.clear_grad()
                total_loss_val += loss_val.item()
            else:
                print(f"警告: epoch {epoch}, batch {batch_id}, loss计算结果为None")
                continue
            
            lr_scheduler.step() 
            
            if current_logits_for_acc is not None:
                pred = paddle.argmax(current_logits_for_acc, axis=1)
                correct_samples += (pred == labels).sum().item()
            
            total_samples += labels.shape[0]
            global_step += 1
            
            if batch_id % config.log_interval == 0:
                current_lr = opt.get_lr()
                print(f"  Batch {batch_id}/{len(train_loader)}, Loss: {loss_val.item():.4f}, "
                      f"Train Acc: {correct_samples/total_samples:.4f}, LR: {current_lr:.6f}")
        
        avg_loss = total_loss_val / len(train_loader) if len(train_loader) > 0 else 0
        avg_acc = correct_samples / total_samples if total_samples > 0 else 0
        print(f"Epoch {epoch + 1} Training Summary: Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")
        
        # 测试模式
        if model_instance: model_instance.eval()
        if backbone: backbone.eval()
        if head_module: head_module.eval()
        test_correct_samples = 0
        test_total_samples = 0
        
        with paddle.no_grad():
            for data in test_loader:
                images, labels = data
                current_logits_for_acc_test = None
                if config.model_type == 'vgg':
                    _, logits_test = model_instance(images)
                    current_logits_for_acc_test = logits_test
                elif config.model_type == 'resnet':
                    features_test = backbone(images)
                    if config.loss_type == 'arcface' and head_module:
                        _, softmax_output_test = head_module(features_test, labels) 
                        current_logits_for_acc_test = softmax_output_test
                    elif config.loss_type == 'cross_entropy' and head_module:
                        logits_test = head_module(features_test) 
                        current_logits_for_acc_test = logits_test
                
                if current_logits_for_acc_test is not None:
                    pred_test = paddle.argmax(current_logits_for_acc_test, axis=1)
                    test_correct_samples += (pred_test == labels).sum().item()
                test_total_samples += labels.shape[0]
        
        test_acc = test_correct_samples / test_total_samples if test_total_samples > 0 else 0
        print(f"Epoch {epoch + 1} Test Summary: Accuracy: {test_acc:.4f}")
        
        # -------------------- 保存检查点 --------------------
        # 保存当前的配置对象 (config 是 ConfigObject，可以转为dict保存)
        checkpoint_content = {
            'optimizer': opt.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch, 'best_acc': best_acc, 'global_step': global_step,
            'config': config.to_dict() # Save the currently used *merged* config
        }
        
        if model_instance: checkpoint_content['model'] = model_instance.state_dict()
        if backbone: checkpoint_content['backbone'] = backbone.state_dict()
        if head_module: checkpoint_content['head'] = head_module.state_dict()
            
        paddle.save(checkpoint_content, checkpoint_path)
        print(f"检查点已保存到: {checkpoint_path}")
        # -----------------------------------------------------
        
        # -------------------- 保存最佳模型 -------------------
        if test_acc > best_acc:
            best_acc = test_acc
            model_to_save = {'config': config.to_dict()}
            if model_instance: model_to_save['model'] = model_instance.state_dict()
            if backbone: model_to_save['backbone'] = backbone.state_dict()
            if head_module: model_to_save['head'] = head_module.state_dict()
            
            paddle.save(model_to_save, current_best_model_path)
            print(f"最佳模型已更新并保存到: {current_best_model_path} (Epoch {epoch + 1}, Accuracy: {best_acc:.4f})")
        # -----------------------------------------------------
    
    print(f"训练完成。最佳测试准确率: {best_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='人脸识别训练脚本')
    
    # 关键的命令行参数，用于控制配置加载和基本运行模式
    parser.add_argument('--config_path', type=str, default=None, help='YAML 配置文件路径。如果未提供，则使用默认路径。')
    parser.add_argument('--use_gpu', action=argparse.BooleanOptionalAction, help='是否使用GPU进行训练 (若可用，此命令行参数会覆盖配置文件中的设置)。')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, help='是否从检查点恢复训练 (此命令行参数会覆盖配置文件中的设置)。')

    # 以下参数如果包含在配置文件中，它们的值将被配置文件覆盖，除非用户在命令行显式指定了这些参数。
    # 如果配置文件中没有这些参数，则使用这里的默认值。
    # 为了清晰，建议将这些参数的主要默认值维护在 YAML 文件中。
    # 命令行中定义的默认值，实际上是 argparse 在未从配置文件读取到相应值时的"最终默认"。

    # 数据和模型相关参数 (这些参数的默认值应主要由 YAML 文件提供)
    parser.add_argument('--data_dir', type=str, help='数据集根目录 (覆盖配置文件)')
    parser.add_argument('--class_name', type=str, help='数据集中的分类名称 (覆盖配置文件)')
    parser.add_argument('--model_save_dir', type=str, help='模型保存的目录路径 (覆盖配置文件)')
    parser.add_argument('--model_type', type=str, choices=['vgg', 'resnet'], help='选择模型类型 (覆盖配置文件)')
    
    # 训练超参数 (这些参数的默认值应主要由 YAML 文件提供)
    parser.add_argument('--num_classes', type=int, help='分类数量 (覆盖配置文件)')
    parser.add_argument('--image_size', type=int, help='输入图像大小 (覆盖配置文件)')
    parser.add_argument('--batch_size', type=int, help='训练批大小 (覆盖配置文件)')
    parser.add_argument('--epochs', type=int, help='总训练轮数 (覆盖配置文件)')
    parser.add_argument('--learning_rate', type=float, help='初始学习率 (覆盖配置文件)')
    parser.add_argument('--log_interval', type=int, help='打印日志的间隔 (覆盖配置文件)')
    parser.add_argument('--seed', type=int, help='随机种子 (覆盖配置文件)')
    
    # ResNet特定参数 (这些参数的默认值应主要由 YAML 文件提供)
    parser.add_argument('--loss_type', type=str, choices=['cross_entropy', 'arcface'], help='损失函数类型 (覆盖配置文件)')
    
    args = parser.parse_args()

    # 加载配置: default_yaml_path 是此脚本认为的默认配置文件位置
    # cmd_args_namespace 是解析后的命令行参数，其中的 config_path (如果提供) 会被 load_config优先使用
    config = load_config(default_yaml_path='configs/default_config.yaml', cmd_args_namespace=args)
    
    print("开始训练前的最终配置确认 (来自YAML并由命令行更新后):")
    # 打印关键配置信息，可以使用 config 对象访问
    print(f"模型类型: {config.model_type}, 损失类型: {config.loss_type}, GPU使用: {config.use_gpu}, 恢复训练: {config.resume}")
    print(f"学习率: {config.learning_rate}, Epochs: {config.epochs}, 批大小: {config.batch_size}")
    if config.model_type == 'resnet':
        resnet_p = config.model.get('resnet_params', {})
        print(f"ResNet参数: 特征维度={resnet_p.get('feature_dim')}, nf={resnet_p.get('nf')}, n_blocks={resnet_p.get('n_resnet_blocks')}")
        if config.loss_type == 'arcface':
            arc_p = config.loss.get('arcface_params', {})
            print(f"  使用 ArcFace Loss, 参数: m1={arc_p.get('arcface_m1')}, m2={arc_p.get('arcface_m2')}, m3={arc_p.get('arcface_m3')}, s={arc_p.get('arcface_s')}")
        else:
            print("  ResNet 使用 CrossEntropy Loss")
    elif config.model_type == 'vgg':
        vgg_p = config.model.get('vgg_params', {})
        print(f"VGG 参数: dropout_rate={vgg_p.get('dropout_rate')}")

    train(config, args) # 将加载好的config对象传递给训练函数 