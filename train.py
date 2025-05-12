# train.py
# 本脚本是人脸识别项目的核心训练模块。
# 它负责根据配置文件中的设定，执行模型的训练流程，包括：
# 1. 设置运行环境（CPU/GPU）和随机种子。
# 2. 使用 MyReader.py 中的 create_dataloader 创建训练和测试数据加载器。
# 3. 使用 model_factory.py 中的工厂函数 (get_backbone, get_head) 构建骨干网络和模型头部。
# 4. 使用 utils.lr_scheduler_factory.py 中的工厂函数 (get_lr_scheduler) 构建学习率调度器。
# 5. 定义优化器 (AdamW, Momentum)。
# 6. 实现检查点加载逻辑，支持从断点恢复训练或加载预训练模型（部分权重）。
# 7. 执行训练循环 (epochs)，包括前向传播、损失计算、反向传播、参数优化。
# 8. 在每个epoch结束后，在测试集上进行评估。
# 9. 根据评估结果保存最佳模型和训练检查点。
# 10. 详细的日志输出，记录训练过程中的损失、准确率、学习率等信息。
#
# 脚本通过解析命令行参数和YAML配置文件来获取所有必要的参数，
# 由 config_utils.py 中的 load_config 函数统一管理配置的加载和合并。

import os
import argparse
import paddle
import paddle.nn as nn # 引入nn以备不时之需，尽管工厂函数已封装模型创建
import paddle.optimizer as optimizer
# import paddle.nn.functional as F # F 已在模型定义或头部模块中按需导入
import MyReader                 # 导入数据读取和预处理模块
from config_utils import load_config, ConfigObject # 导入配置加载工具和配置对象类型
from model_factory import get_backbone, get_head   # 导入模型构建工厂函数
from utils.lr_scheduler_factory import get_lr_scheduler # 导入学习率调度器工厂函数

def train(config: ConfigObject, cmd_args: argparse.Namespace):
    """模型训练的主函数。

    根据传入的配置对象 `config` 和命令行参数对象 `cmd_args` 来执行完整的模型训练流程。

    Args:
        config (ConfigObject): 包含所有训练所需参数的配置对象。
                               这些参数通常来自YAML文件和命令行的合并结果。
        cmd_args (argparse.Namespace): 包含从命令行直接解析得到的参数的对象。
                                       主要用于判断如 `--resume` 等覆盖配置文件的行为。
    """
    # 设置随机种子，以确保实验结果的可复现性
    paddle.seed(config.seed)
    
    # 设置运行设备 (GPU或CPU)
    # 优先使用GPU；如果GPU不可用或配置为不使用GPU，则使用CPU。
    if config.use_gpu and paddle.is_compiled_with_cuda():
        paddle.set_device('gpu')
        print("使用 GPU 进行训练")
    else:
        paddle.set_device('cpu')
        print("使用 CPU 进行训练")
        
    # 构建训练和测试数据列表文件的完整路径
    # 这些路径基于配置文件中的 data_dir (数据根目录) 和 class_name (数据集子目录名)。
    # train_list_path = os.path.join(config.data_dir, config.class_name, "trainer.list") # MyReader.create_dataloader现在直接接收config
    # test_list_path = os.path.join(config.data_dir, config.class_name, "test.list")
    
    # --- 创建数据加载器 (DataLoader) ---
    # 使用 MyReader.py 中的 create_data_loader 函数创建训练数据加载器。
    # 它会根据配置处理数据增强、批处理等。
    # 注意: MyReader.create_data_loader 函数现在直接接收完整的config对象和模式字符串。
    print("正在创建训练数据加载器...")
    train_loader = MyReader.create_data_loader(
        config=config, # 传递完整的配置对象
        mode='train'   # 指定为训练模式
    )
    
    # 创建测试数据加载器
    print("正在创建测试数据加载器...")
    test_loader = MyReader.create_data_loader(
        config=config, # 传递完整的配置对象
        mode='eval'    # 指定为评估模式 (通常测试集在验证时称为eval)
    )
    
    # --- 使用模型工厂 (model_factory.py) 获取骨干网络和头部模块 ---
    model_type_str = config.model_type         # 如 'vgg', 'resnet'
    loss_type_str = config.loss_type           # 如 'cross_entropy', 'arcface'，也决定头部类型
    num_classes = config.num_classes           # 数据集中的类别总数
    image_size = config.image_size             # 输入图像尺寸，某些骨干网络构建时可能需要

    # 从主配置对象中提取骨干网络所需的特定参数字典。
    # 例如，config.model 可能包含 config.model.vgg_params 或 config.model.resnet_params。
    backbone_specific_params = config.model.get(f'{model_type_str}_params', {})
    print(f"为骨干网络 '{model_type_str}' 提取的参数: {backbone_specific_params}")

    backbone_instance, feature_dim_from_backbone = get_backbone(
        config_model_params=backbone_specific_params, 
        model_type_str=model_type_str, 
        image_size=image_size
    )
    if not backbone_instance:
        raise RuntimeError(f"从工厂创建骨干网络 '{model_type_str}' 失败。")
    print(f"骨干网络 ({model_type_str.upper()}) 加载成功，输出特征维度: {feature_dim_from_backbone}")

    # 从主配置对象中提取头部模块/损失函数所需的特定参数字典。
    # 例如，config.loss 可能包含 config.loss.arcface_params。
    head_specific_params = config.loss.get(f'{loss_type_str}_params', {})
    print(f"为头部模块 '{loss_type_str}' 提取的参数: {head_specific_params}")

    head_module_instance = get_head(
        config_loss_params=head_specific_params, 
        loss_type_str=loss_type_str, 
        in_features=feature_dim_from_backbone, # 头部输入维度需与骨干输出维度一致
        num_classes=num_classes
    )
    if not head_module_instance:
        raise RuntimeError(f"从工厂创建头部模块 '{loss_type_str}' 失败。")
    print(f"头部模块 ({loss_type_str.upper()}) 加载成功，输入特征维度: {feature_dim_from_backbone}, 输出类别数: {num_classes}")
    # ---------------------------------------------------------------------------

    # 初始化训练状态变量
    start_epoch = 0      # 训练起始轮次
    best_acc = 0.0       # 记录训练过程中在测试集上的最佳准确率
    global_step = 0      # 记录总的训练步数 (batch数量)
    
    # ----------- 定义检查点和最佳模型的保存路径 ------------- 
    # 模型保存目录由配置文件中的 model_save_dir 指定。
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
        print(f"创建模型保存目录: {config.model_save_dir}")

    # 检查点和最佳模型的文件名格式包含模型类型和损失类型，以区分不同配置的产物。
    # filename_suffix = f"{model_type_str}_{loss_type_str}" # 更通用的后缀
    # # 对于VGG，历史上可能只用 'ce' 作为损失后缀，除非明确用了arcface
    # if model_type_str == 'vgg' and loss_type_str != 'arcface':
    #     filename_suffix = f"{model_type_str}_ce"
    # elif model_type_str == 'vgg' and loss_type_str == 'arcface':
    #     filename_suffix = f"{model_type_str}_arcface" # 明确的VGG+ArcFace
    # # 对于ResNet，通常会带上损失类型

    # 统一使用 model_type 和 loss_type 组合作为文件名后缀
    filename_suffix = f"{config.model_type}_{config.loss_type}"

    checkpoint_filename = f"checkpoint_{filename_suffix}.pdparams"
    checkpoint_path = os.path.join(config.model_save_dir, checkpoint_filename)
    
    best_model_filename = f"best_model_{filename_suffix}.pdparams"
    best_model_path = os.path.join(config.model_save_dir, best_model_filename)
    print(f"训练检查点将保存至: {checkpoint_path}")
    print(f"最佳模型将保存至: {best_model_path}")
    # ---------------------------------------------------------
    
    # --- 定义学习率调度器 (Learning Rate Scheduler) ---
    # 使用 utils.lr_scheduler_factory.py 中的 get_lr_scheduler 函数创建。
    # 该函数会根据配置文件中的 lr_scheduler_type 和 lr_scheduler_params 创建具体的调度器实例。
    # 同时支持Warmup策略。
    lr_scheduler = get_lr_scheduler(config, initial_learning_rate=config.learning_rate)
    if lr_scheduler is None:
        # get_lr_scheduler 在无法创建时应抛出异常，这里理论上不会执行。
        # 但作为防御性编程，如果真的返回None，则应停止或使用固定学习率。
        raise RuntimeError("学习率调度器未能从工厂中创建。请检查配置和工厂实现。")

    # --- 定义优化器 (Optimizer) --- 
    # 将骨干网络和头部模块的参数都加入到优化列表中。
    params_to_optimize = []
    if backbone_instance: params_to_optimize.extend(backbone_instance.parameters())
    if head_module_instance: params_to_optimize.extend(head_module_instance.parameters())

    if not params_to_optimize:
        raise ValueError("没有可优化的模型参数。请检查骨干网络和头部模块是否已正确加载。")

    # 从配置文件获取优化器类型 (optimizer_type) 及其特定参数 (optimizer_params)。
    optimizer_specific_params = config.get('optimizer_params', {}) # 如 weight_decay, momentum
    optimizer_type_str = config.get('optimizer_type', 'Momentum')   # 默认为 Momentum

    if optimizer_type_str.lower() == 'momentum':
        opt = optimizer.Momentum(
            learning_rate=lr_scheduler, # 学习率可以是一个调度器对象
            parameters=params_to_optimize,
            momentum=optimizer_specific_params.get('momentum', 0.9),
            weight_decay=optimizer_specific_params.get('weight_decay', 0.0005)
        )
    elif optimizer_type_str.lower() == 'adamw':
        opt = optimizer.AdamW(
            learning_rate=lr_scheduler,
            parameters=params_to_optimize,
            weight_decay=optimizer_specific_params.get('weight_decay', 0.0001)
            # AdamW的其他参数如 beta1, beta2, epsilon 可以按需从 optimizer_specific_params 获取
        )
    else:
        raise ValueError(f"不支持的优化器类型: '{optimizer_type_str}'. 支持 'Momentum' 或 'AdamW'。")
    print(f"使用优化器: {optimizer_type_str}，初始学习率(来自配置): {config.learning_rate}, WeightDecay: {optimizer_specific_params.get('weight_decay')}")

    # -------------------- 检查点加载逻辑 --------------------
    # 该逻辑决定是从头开始训练，还是从已有的检查点恢复训练。
    # 优先级: 命令行参数 > YAML配置 > 默认行为 (存在则恢复)
    checkpoint_exists = os.path.exists(checkpoint_path)
    should_resume_from_checkpoint = False # 标记是否最终决定从检查点恢复

    if checkpoint_exists:
        if cmd_args.resume is False: # 命令行明确指定不恢复 (--no-resume)
            print(f"检查点 {checkpoint_path} 存在，但用户通过命令行 (--no-resume) 指示从头开始训练。")
        elif cmd_args.resume is True: # 命令行明确指定恢复 (--resume)
            print(f"检查点 {checkpoint_path} 存在，用户通过命令行 (--resume) 指示恢复训练。将尝试恢复。")
            should_resume_from_checkpoint = True
        else: # 命令行未指定resume行为，则参考配置文件中的 resume 设置
            if config.resume is False:
                print(f"检查点 {checkpoint_path} 存在，但配置文件指示不恢复 (resume: False)，且命令行未指定。将从头开始训练。")
            elif config.resume is True:
                print(f"检查点 {checkpoint_path} 存在，配置文件指示恢复 (resume: True)，将尝试恢复。")
                should_resume_from_checkpoint = True
            else: # config.resume 为 None 或未定义 (视为默认行为：存在即恢复)
                print(f"检查点 {checkpoint_path} 存在，将自动尝试恢复训练 (配置文件 resume 未明确设为False，命令行未指定)。")
                should_resume_from_checkpoint = True
    else: # 检查点文件不存在
        if cmd_args.resume is True: # 用户指定了 --resume 但文件不存在
            print(f"警告: 用户指定了 --resume 但检查点文件 {checkpoint_path} 不存在。将从头开始训练。")
        elif config.resume is True and cmd_args.resume is None: # 配置文件要求恢复但文件不存在
             print(f"警告: 配置文件要求恢复训练 (resume: True) 但检查点文件 {checkpoint_path} 不存在。将从头开始训练。")
        else: # 其他情况，都是从头开始
            print(f"检查点文件 {checkpoint_path} 不存在。将从头开始训练。")

    if not should_resume_from_checkpoint:
        start_epoch = 0
        best_acc = 0.0
        global_step = 0
        print("将从 epoch 0，best_acc 0.0 开始新的训练。")
    else: # should_resume_from_checkpoint is True，尝试加载检查点
        print(f"尝试从检查点 {checkpoint_path} 恢复训练...")
        try:
            checkpoint_data = paddle.load(checkpoint_path)
            
            # 恢复训练状态变量
            # 如果检查点中没有这些键，则使用默认值，以增加兼容性
            loaded_epoch = checkpoint_data.get('epoch', -1) 
            start_epoch = loaded_epoch + 1 # 下一个epoch开始
            best_acc = checkpoint_data.get('best_acc', 0.0)
            global_step = checkpoint_data.get('global_step', 0)
            
            # 检查检查点中的配置是否与当前配置兼容
            # 检查点中保存的配置有助于追溯和兼容性判断
            ckpt_saved_config_dict = checkpoint_data.get('config', {}) # 应为字典
            if not isinstance(ckpt_saved_config_dict, dict):
                # 尝试兼容旧的 argparse.Namespace 保存方式
                ckpt_saved_config_dict = vars(ckpt_saved_config_dict) if hasattr(ckpt_saved_config_dict, '__dict__') else {}
            
            ckpt_model_type = ckpt_saved_config_dict.get('model_type')
            ckpt_loss_type = ckpt_saved_config_dict.get('loss_type')
            ckpt_num_classes = ckpt_saved_config_dict.get('num_classes')

            # 标记权重加载是否成功以及配置是否兼容
            weights_loaded_successfully = True
            is_config_compatible_for_full_resume = True 

            # 核心兼容性检查：模型类型、损失类型、类别数
            if ckpt_model_type != model_type_str or \
               ckpt_loss_type != loss_type_str or \
               (ckpt_num_classes is not None and ckpt_num_classes != num_classes):
                print(f"警告: 检查点配置与当前配置不完全兼容。")
                print(f"  检查点: model_type='{ckpt_model_type}', loss_type='{ckpt_loss_type}', num_classes={ckpt_num_classes}")
                print(f"  当前  : model_type='{model_type_str}', loss_type='{loss_type_str}', num_classes={num_classes}")
                print("将仅尝试加载匹配的模型权重（骨干和头部），训练状态 (epoch, best_acc, optimizer, lr_scheduler) 将被重置。")
                is_config_compatible_for_full_resume = False
                start_epoch = 0 # 重置训练状态
                best_acc = 0.0
                global_step = 0

            # 加载骨干网络权重
            if 'backbone' in checkpoint_data and backbone_instance:
                try:
                    backbone_instance.set_state_dict(checkpoint_data['backbone'])
                    print("骨干网络权重从检查点加载成功。")
                except Exception as e:
                    print(f"警告: 加载骨干网络权重失败: {e}。骨干网络将使用初始权重。")
                    weights_loaded_successfully = False
            elif backbone_instance:
                print(f"警告: 检查点中未找到 'backbone' 键对应的骨干网络权重。骨干网络将使用初始权重。")
                weights_loaded_successfully = False

            # 加载头部模块权重
            if 'head' in checkpoint_data and head_module_instance:
                try:
                    head_module_instance.set_state_dict(checkpoint_data['head'])
                    print("头部模块权重从检查点加载成功。")
                except Exception as e:
                    print(f"警告: 加载头部模块权重失败: {e}。头部模块将使用初始权重。")
                    weights_loaded_successfully = False # 标记权重加载不完全成功
            elif head_module_instance:
                print(f"警告: 检查点中未找到 'head' 键对应的头部模块权重。头部模块将使用初始权重。")
                weights_loaded_successfully = False
            
            if not weights_loaded_successfully and is_config_compatible_for_full_resume:
                # 如果配置兼容，但权重加载失败，这通常是个问题
                print("警告: 模型配置兼容，但部分或全部模型权重加载失败。请检查检查点文件完整性。")
            
            # 如果配置兼容，则恢复优化器和学习率调度器状态
            if is_config_compatible_for_full_resume:
                if 'optimizer' in checkpoint_data: opt.set_state_dict(checkpoint_data['optimizer'])
                if 'lr_scheduler' in checkpoint_data: lr_scheduler.set_state_dict(checkpoint_data['lr_scheduler'])
                print(f"从检查点恢复成功。将从 epoch {start_epoch} 开始，当前最佳准确率: {best_acc:.4f}, 全局步数: {global_step}")
            else: # 配置不兼容，已重置训练状态
                print(f"由于配置不兼容，训练状态已重置。模型权重已尝试加载。将从 epoch 0 开始新训练。")

        except Exception as e:
            print(f"加载检查点 {checkpoint_path} 失败: {e}。将从头开始训练。")
            start_epoch = 0; best_acc = 0.0; global_step = 0
    # -----------------------------------------------------------------

    print(f"开始训练，总共 {config.epochs} 个 epochs... 从 epoch {start_epoch} 开始")
    # --- 主训练循环 (Outer loop for epochs) ---
    for epoch_idx in range(start_epoch, config.epochs):
        # 设置模型为训练模式 (启用dropout, batchnorm更新等)
        if backbone_instance: backbone_instance.train()
        if head_module_instance: head_module_instance.train()
        
        # 初始化每个epoch的统计变量
        epoch_total_loss = 0.0    # 当前epoch的总损失
        epoch_correct_samples = 0 # 当前epoch训练集中预测正确的样本数
        epoch_total_samples = 0   # 当前epoch训练集中处理的总样本数
        
        print(f"--- Epoch {epoch_idx + 1}/{config.epochs} --- LR: {opt.get_lr():.6f} ---")
        # --- 内部循环遍历训练数据加载器 (Inner loop for batches) ---
        for batch_id, train_data_batch in enumerate(train_loader):
            images, labels = train_data_batch # 解包图像和标签
            
            # 确保 labels 的形状是 [batch_size] 而不是 [batch_size, 1]
            if labels.ndim > 1 and labels.shape[1] == 1:
                labels = paddle.squeeze(labels, axis=1)
            
            # 1. 前向传播: 获取骨干网络输出的特征
            features = backbone_instance(images)
            # 2. 前向传播: 获取头部模块输出的损失和用于计算准确率的输出
            #    头部模块 (如ArcFaceHead, CrossEntropyHead) 内部会计算损失。
            loss_value, accuracy_output = head_module_instance(features, labels)
            
            # 3. 反向传播和优化
            if loss_value is not None and loss_value.item() > 0: # 确保损失有效
                loss_value.backward() # 计算梯度
                opt.step()            # 更新模型参数
                opt.clear_grad()      # 清除梯度，为下一次迭代做准备
                epoch_total_loss += loss_value.item() * images.shape[0] # 累加批次损失 (乘以bs得到总损失)
            else:
                # 如果损失为None或无效，打印警告并跳过优化步骤
                print(f"警告: Epoch {epoch_idx+1}, Batch {batch_id}, 损失计算结果为None或无效 ({loss_value})，跳过优化。")
                continue # 跳到下一个batch
            
            # 4. （如果不是ReduceLROnPlateau）在每个batch后更新学习率调度器状态
            if not isinstance(lr_scheduler, paddle.optimizer.lr.ReduceOnPlateau):
                lr_scheduler.step() 

            # 5. 计算当前批次的训练准确率 (可选，用于日志)
            if accuracy_output is not None:
                # accuracy_output 通常是logits或softmax概率，需要argmax获取预测类别
                predicted_labels = paddle.argmax(accuracy_output, axis=1)
                epoch_correct_samples += (predicted_labels == labels).sum().item()
            
            epoch_total_samples += labels.shape[0] # 更新已处理的样本总数
            global_step += 1 # 全局训练步数增加
            
            # 6. 定期打印训练日志
            if batch_id % config.log_interval == 0:
                current_lr = opt.get_lr() # 获取当前学习率
                # 训练到目前为止的平均准确率
                train_acc_so_far_in_epoch = epoch_correct_samples / epoch_total_samples if epoch_total_samples > 0 else 0
                # 当前批次的平均损失
                current_batch_avg_loss = loss_value.item() 
                print(f"  Batch {batch_id+1}/{len(train_loader)}, AvgLoss(batch): {current_batch_avg_loss:.4f}, "
                      f"TrainAcc(epoch_so_far): {train_acc_so_far_in_epoch:.4f}, LR: {current_lr:.6f}")
        # --- 单个Epoch训练结束 ---
        
        # 计算当前epoch在整个训练集上的平均损失和平均准确率
        avg_epoch_train_loss = epoch_total_loss / epoch_total_samples if epoch_total_samples > 0 else 0
        avg_epoch_train_acc = epoch_correct_samples / epoch_total_samples if epoch_total_samples > 0 else 0
        print(f"Epoch {epoch_idx + 1} Training Summary: AvgLoss: {avg_epoch_train_loss:.4f}, AvgAcc: {avg_epoch_train_acc:.4f}")
        
        # --- 在测试集上进行评估 (Validation/Testing after each epoch) ---
        # 设置模型为评估模式 (关闭dropout, batchnorm使用固定均值方差等)
        if backbone_instance: backbone_instance.eval()
        if head_module_instance: head_module_instance.eval()
        
        eval_correct_samples = 0
        eval_total_samples = 0
        eval_total_loss = 0.0 # 用于ReduceLROnPlateau的验证损失
        
        with paddle.no_grad(): # 评估时不需要计算梯度
            for eval_data_batch in test_loader:
                images, labels = eval_data_batch
                # 确保 labels 的形状是 [batch_size] 而不是 [batch_size, 1]
                if labels.ndim > 1 and labels.shape[1] == 1:
                    labels = paddle.squeeze(labels, axis=1)
                    
                features_eval = backbone_instance(images)
                # 头部模块在评估时也应能返回损失 (用于ReduceLROnPlateau) 和准确率计算的输出
                loss_eval_value, acc_output_eval = head_module_instance(features_eval, labels) 
                
                if loss_eval_value is not None: # 累加验证集损失
                    eval_total_loss += loss_eval_value.item() * images.shape[0]

                if acc_output_eval is not None:
                    predicted_labels_eval = paddle.argmax(acc_output_eval, axis=1)
                    eval_correct_samples += (predicted_labels_eval == labels).sum().item()
                eval_total_samples += labels.shape[0]
        
        # 计算在整个测试集上的平均准确率和平均损失
        current_eval_acc = eval_correct_samples / eval_total_samples if eval_total_samples > 0 else 0
        avg_eval_loss = eval_total_loss / eval_total_samples if eval_total_samples > 0 else float('inf')
        print(f"Epoch {epoch_idx + 1} Test Summary: Accuracy: {current_eval_acc:.4f}, AvgLoss: {avg_eval_loss:.4f}")

        # --- （如果是ReduceLROnPlateau）在每个epoch评估后，根据验证指标更新学习率 --- 
        if isinstance(lr_scheduler, paddle.optimizer.lr.ReduceOnPlateau):
            # 从配置中获取 ReduceLROnPlateau 监控的指标名 (如 'loss' 或 'acc')
            plateau_scheduler_params = config.lr_scheduler_params.get('reduce_lr_on_plateau', {})
            plateau_metric_to_monitor = plateau_scheduler_params.get('metric_name', 'loss') # 默认为loss
            
            if plateau_metric_to_monitor == 'loss':
                lr_scheduler.step(avg_eval_loss) # 传入验证集平均损失
                print(f"ReduceOnPlateau: Stepped with validation loss {avg_eval_loss:.4f}")
            elif plateau_metric_to_monitor == 'acc':
                lr_scheduler.step(current_eval_acc) # 传入验证集准确率
                print(f"ReduceOnPlateau: Stepped with validation accuracy {current_eval_acc:.4f}")
            else:
                # 如果配置了未知的监控指标，默认使用损失，并打印警告
                print(f"警告: ReduceOnPlateau 配置了未知的 metric_name: '{plateau_metric_to_monitor}'。将使用验证损失进行 step。")
                lr_scheduler.step(avg_eval_loss)
        
        # --- 保存检查点 (Checkpointing) ---
        # 将当前训练配置转换为字典以便保存。这有助于后续恢复或分析。
        current_config_dict_to_save = config.to_dict() 
        
        # # (可选) 准备更详细的骨干网络参数用于保存，如果工厂函数有修改或添加默认值。
        # # 这里假设config对象中的backbone_specific_params已经是最终生效的参数。
        # current_config_dict_to_save['backbone_params_used'] = backbone_specific_params
        # current_config_dict_to_save['head_params_used'] = head_specific_params

        checkpoint_content_to_save = {
            'optimizer': opt.state_dict(), 
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch_idx, 
            'best_acc': best_acc, 
            'global_step': global_step,
            'config': current_config_dict_to_save # 保存当前训练的完整配置
        }
        if backbone_instance: checkpoint_content_to_save['backbone'] = backbone_instance.state_dict()
        if head_module_instance: checkpoint_content_to_save['head'] = head_module_instance.state_dict()
            
        paddle.save(checkpoint_content_to_save, checkpoint_path)
        print(f"检查点已保存到: {checkpoint_path}")
        
        # --- 保存最佳模型 (Save best model based on validation accuracy) ---
        if current_eval_acc > best_acc:
            best_acc = current_eval_acc
            # 最佳模型文件也应包含配置信息，以及骨干和头部的权重。
            best_model_content_to_save = {'config': current_config_dict_to_save} 
            if backbone_instance: best_model_content_to_save['backbone'] = backbone_instance.state_dict()
            if head_module_instance: best_model_content_to_save['head'] = head_module_instance.state_dict()
            
            paddle.save(best_model_content_to_save, best_model_path)
            print(f"最佳模型已更新并保存到: {best_model_path} (Epoch {epoch_idx + 1}, Test Accuracy: {best_acc:.4f})")
    # --- 所有Epoch训练结束 ---
    
    print(f"训练完成。在测试集上的最佳准确率: {best_acc:.4f}")
    print(f"最终模型检查点位于: {checkpoint_path}")
    print(f"性能最佳的模型位于: {best_model_path}")

if __name__ == '__main__':
    # --- 命令行参数解析 --- 
    # 使用 argparse 定义脚本运行时可以接受的命令行参数。
    parser = argparse.ArgumentParser(description='人脸识别模型训练脚本')
    
    # 核心控制参数
    parser.add_argument('--config_path', type=str, default=None, 
                        help='指定YAML配置文件的路径。如果未提供，则使用脚本内部定义的默认路径。')
    parser.add_argument('--use_gpu', action=argparse.BooleanOptionalAction, default=None,
                        help='是否使用GPU进行训练。此命令行开关会覆盖配置文件中的设置。'
                             '例如: --use_gpu (设为True), --no-use_gpu (设为False)。如果未指定，则遵循配置文件。')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=None,
                        help='是否从检查点恢复训练。此命令行开关会覆盖配置文件中的设置。'
                             '例如: --resume (设为True), --no-resume (设为False)。如果未指定，则遵循配置文件。')
    parser.add_argument('--active_config', type=str, default=None,
                        help='通过命令行指定要激活的配置块名称，覆盖YAML文件中的active_config设置。')

    # --- 其他可覆盖配置文件的参数 ---
    # 以下参数如果用户在命令行中指定了，其值将覆盖从YAML配置文件中加载的同名参数。
    # 如果命令行未指定，则使用YAML中的值。如果YAML中也没有，则使用argparse中定义的default值（如果提供了）。
    # 最佳实践是主要在YAML文件中维护这些参数的默认值。

    # 数据和模型保存路径相关 (通常由YAML配置)
    parser.add_argument('--data_dir', type=str, help='数据集根目录 (覆盖配置文件中的 global_settings.data_dir)')
    parser.add_argument('--class_name', type=str, help='数据集子目录名 (覆盖配置文件中的 global_settings.class_name)')
    parser.add_argument('--model_save_dir', type=str, help='模型和检查点保存目录 (覆盖配置文件中的 global_settings.model_save_dir)')
    
    # 模型结构相关 (通常由YAML中 active_config 指向的配置块定义)
    parser.add_argument('--model_type', type=str, choices=['vgg', 'resnet'], help='选择骨干网络类型 (覆盖配置文件中的 model_type)')
    parser.add_argument('--loss_type', type=str, choices=['cross_entropy', 'arcface'], help='选择损失函数/头部类型 (覆盖配置文件中的 loss_type)')
    parser.add_argument('--num_classes', type=int, help='数据集中的类别总数 (覆盖配置文件中的 num_classes)')
    parser.add_argument('--image_size', type=int, help='输入图像预处理后的统一尺寸 (覆盖配置文件中的 image_size)')
    
    # 训练超参数 (通常由YAML中 active_config 指向的配置块定义)
    parser.add_argument('--batch_size', type=int, help='训练批大小 (覆盖配置文件中的 batch_size)')
    parser.add_argument('--epochs', type=int, help='总训练轮数 (覆盖配置文件中的 epochs)')
    parser.add_argument('--learning_rate', type=float, help='优化器的初始学习率 (覆盖配置文件中的 learning_rate)')
    parser.add_argument('--log_interval', type=int, help='训练时打印日志的间隔批次数 (覆盖配置文件中的 global_settings.log_interval)')
    parser.add_argument('--seed', type=int, help='随机种子，用于实验复现 (覆盖配置文件中的 global_settings.seed)')
    
    # ResNet骨干网络的特定参数 (通常在YAML的 resnet_params 中配置)
    # 为了简单，这里不将所有嵌套参数都暴露到命令行，主要依赖YAML。如需命令行覆盖，可按需添加。
    # parser.add_argument('--resnet_feature_dim', type=int, help='ResNet输出特征维度 (覆盖 model.resnet_params.feature_dim)')

    # ArcFace损失的特定参数 (通常在YAML的 arcface_params 中配置)
    # parser.add_argument('--arcface_s', type=float, help='ArcFace的尺度因子s (覆盖 loss.arcface_params.arcface_s)')
    # parser.add_argument('--arcface_m2', type=float, help='ArcFace的角度间隔m (覆盖 loss.arcface_params.arcface_m2)')

    cmd_line_args = parser.parse_args()

    # --- 配置加载与合并 ---
    # 使用 config_utils.load_config 函数加载配置。
    # 它会首先尝试加载命令行 --config_path 指定的YAML文件，如果未指定，则使用 default_yaml_path。
    # 然后，它会将命令行解析的参数 cmd_line_args 合并到YAML配置中，命令行参数具有更高优先级。
    # 返回一个 ConfigObject 实例，方便通过属性访问配置项。
    final_config = load_config(
        default_yaml_path='configs/default_config.yaml', # 脚本默认查找的YAML配置文件
        cmd_args_namespace=cmd_line_args                 # 解析后的命令行参数
    )
    
    # 打印最终生效的关键配置信息，方便用户确认
    print("\n--- 最终生效的训练配置 (YAML与命令行合并后) ---")
    print(f"  模型类型 (model_type): {final_config.model_type}")
    print(f"  损失/头部类型 (loss_type): {final_config.loss_type}")
    print(f"  GPU 使用 (use_gpu): {final_config.use_gpu}")
    print(f"  从检查点恢复 (resume): {final_config.resume}")
    print(f"  学习率 (learning_rate): {final_config.learning_rate}")
    print(f"  总轮数 (epochs): {final_config.epochs}")
    print(f"  批大小 (batch_size): {final_config.batch_size}")
    print(f"  类别数 (num_classes): {final_config.num_classes}")
    print(f"  图像尺寸 (image_size): {final_config.image_size}")
    print(f"  随机种子 (seed): {final_config.seed}")
    
    if final_config.model_type == 'resnet':
        resnet_p = final_config.model.get('resnet_params', {})
        print(f"  ResNet参数: 特征维度={resnet_p.get('feature_dim')}, nf={resnet_p.get('nf')}, n_blocks={resnet_p.get('n_resnet_blocks')}")
        if final_config.loss_type == 'arcface':
            arc_p = final_config.loss.get('arcface_params', {})
            print(f"    ArcFace参数: m1={arc_p.get('arcface_m1')}, m2={arc_p.get('arcface_m2')}, m3={arc_p.get('arcface_m3')}, s={arc_p.get('arcface_s')}")
    elif final_config.model_type == 'vgg':
        vgg_p = final_config.model.get('vgg_params', {})
        print(f"  VGG参数: dropout_rate={vgg_p.get('dropout_rate')}")
    print("---------------------------------------------------\n")

    # 调用训练函数，开始训练
    train(final_config, cmd_line_args) 