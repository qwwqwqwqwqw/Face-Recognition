# acceptance_test.py
# 该脚本用于在"验收集" (acceptance set) 上评估已训练的人脸识别模型的性能。
#
# 主要流程:
# 1.  加载配置文件 (YAML + 命令行参数)。
# 2.  设置运行设备 (CPU/GPU)。
# 3.  根据配置实例化骨干网络 (backbone) 和模型头部 (head)。
# 4.  从指定的检查点文件加载训练好的模型权重。
# 5.  使用 MyReader.py 创建验收集的数据加载器。
# 6.  执行评估循环：
#     - 将模型设置为评估模式。
#     - 在 paddle.no_grad() 上下文中进行。
#     - 遍历验收数据加载器中的所有批次。
#     - 执行前向传播，获取特征和头部输出。
#     - 计算损失值 (可选，但有助于了解模型在验收集上的表现)。
#     - 计算准确率。
# 7.  汇总并打印在整个验收集上的平均损失和准确率。
# 8.  (可选) 可以扩展此脚本以保存每个样本的预测结果或失败案例。

import os
import argparse
import paddle
import numpy as np # type: ignore
import json # 用于加载模型元数据或保存结果
import time

import MyReader # 数据读取模块
from config_utils import load_config, ConfigObject # 配置加载工具
from model_factory import get_backbone, get_head   # 模型构建工厂

def run_acceptance_evaluation(config: ConfigObject, cmd_args: argparse.Namespace):
    """
    执行在验收集上的模型评估。

    Args:
        config (ConfigObject): 合并后的配置对象。
        cmd_args (argparse.Namespace): 命令行参数。
    """
    # 1. 设置运行设备
    if config.use_gpu and paddle.is_compiled_with_cuda():
        paddle.set_device('gpu')
        print("使用 GPU 进行验收测试。")
    else:
        paddle.set_device('cpu')
        print("使用 CPU 进行验收测试。")

    # 2. 创建验收数据加载器
    print("正在创建验收数据加载器...")
    try:
        acceptance_loader = MyReader.create_data_loader(
            config=config,
            mode='acceptance' # 关键：指定模式为 'acceptance'
        )
        print(f"验收数据加载器创建成功，共 {len(acceptance_loader.dataset)} 个样本。")
    except Exception as e:
        print(f"创建验收数据加载器失败: {e}")
        return

    if len(acceptance_loader.dataset) == 0:
        print("警告: 验收数据集为空，无法进行评估。请检查验收列表文件和路径配置。")
        return

    # 3. 加载模型权重并实例化模型
    #    trained_model_path 应指向包含骨干网络和头部权重的 .pdparams 文件。
    #    该文件也可能包含训练时的配置快照。
    model_weights_path = cmd_args.trained_model_path or config.get('trained_model_path')
    if not model_weights_path:
        print("错误: 未通过命令行参数 --trained_model_path 或配置文件指定模型权重路径。")
        return
    if not os.path.exists(model_weights_path):
        print(f"错误: 指定的模型权重文件不存在: {model_weights_path}")
        return

    print(f"将从以下路径加载模型权重: {model_weights_path}")
    try:
        checkpoint_data = paddle.load(model_weights_path)
    except Exception as e:
        print(f"加载模型权重文件 {model_weights_path} 失败: {e}")
        return

    # --- 重点修改：从元数据文件或当前配置中获取模型参数 ---
    model_type_to_use = None
    loss_type_to_use = None
    num_classes_to_use = None
    image_size_to_use = None
    backbone_specific_params_to_use = {}
    head_specific_params_to_use = {}
    source_of_model_config = "未确定"

    # 尝试加载与模型权重配对的 .json 元数据文件
    metadata_json_path = model_weights_path.replace('.pdparams', '.json')
    if os.path.exists(metadata_json_path):
        print(f"尝试从元数据文件加载模型配置: {metadata_json_path}")
        try:
            with open(metadata_json_path, 'r', encoding='utf-8') as f_meta:
                metadata = json.load(f_meta)
            # 从元数据提取所需配置 (与 train.py 中 CheckpointManager 保存的元数据对应)
            model_type_to_use = metadata.get('model_type')
            loss_type_to_use = metadata.get('loss_type')
            num_classes_to_use = metadata.get('num_classes')
            image_size_to_use = metadata.get('image_size')
            # model_specific_params 和 loss_specific_params 在元数据中可能是嵌套的字典
            # config.model.get(f'{final_config.model_type}_params', {})
            if isinstance(metadata.get('model_specific_params'), dict):
                backbone_specific_params_to_use = metadata['model_specific_params']
            if isinstance(metadata.get('loss_specific_params'), dict):
                head_specific_params_to_use = metadata['loss_specific_params']
            
            if all([model_type_to_use, loss_type_to_use, num_classes_to_use is not None, image_size_to_use is not None]):
                source_of_model_config = f"元数据文件 ({metadata_json_path})"
                print(f"成功从元数据文件加载模型配置。")
            else:
                print(f"警告: 元数据文件 {metadata_json_path} 不完整，缺少核心模型配置。将回退。")
                # 部分加载成功也可能导致后续使用 fallback 值，所以重置确保一致性
                model_type_to_use, loss_type_to_use, num_classes_to_use, image_size_to_use = None, None, None, None 
        except Exception as e_meta_load:
            print(f"加载或解析元数据文件 {metadata_json_path} 失败: {e_meta_load}。将回退。")
            model_type_to_use, loss_type_to_use, num_classes_to_use, image_size_to_use = None, None, None, None 

    # 如果从元数据文件加载失败或不完整，则回退到当前脚本的配置
    if source_of_model_config == "未确定":
        source_of_model_config = "当前脚本的 final_config (回退机制)"
        print(f"警告: 未能从配对的 .json 文件成功加载完整模型配置。将使用 {source_of_model_config}。"
              f"请确保此配置与被测模型 {os.path.basename(model_weights_path)} 的训练时配置严格匹配，否则可能导致错误或不准确的评估。")
        model_type_to_use = config.model_type
        loss_type_to_use = config.loss_type
        num_classes_to_use = config.num_classes
        image_size_to_use = config.image_size
        # 从当前 config 对象安全地获取参数
        backbone_params_obj = config.model.get(f'{model_type_to_use}_params', ConfigObject({}))
        backbone_specific_params_to_use = backbone_params_obj.to_dict() if isinstance(backbone_params_obj, ConfigObject) else backbone_params_obj
        
        head_params_obj = config.loss.get(f'{loss_type_to_use}_params', ConfigObject({}))
        head_specific_params_to_use = head_params_obj.to_dict() if isinstance(head_params_obj, ConfigObject) else head_params_obj

    print(f"--- 模型实例化配置来源: {source_of_model_config} ---")
    print(f"  模型类型: {model_type_to_use}, 损失类型: {loss_type_to_use}, 类别数: {num_classes_to_use}, 图像尺寸: {image_size_to_use}")
    print(f"  骨干参数: {backbone_specific_params_to_use}")
    print(f"  头部参数: {head_specific_params_to_use}")
    print("---------------------------------------------------")

    if not all([model_type_to_use, loss_type_to_use, num_classes_to_use is not None, image_size_to_use is not None]):
        print("错误: 无法最终确定模型构建所需的核心配置 (model_type, loss_type, num_classes, image_size)。请检查元数据或脚本配置。")
        return

    # 实例化骨干网络
    try:
        backbone_instance, feature_dim_from_backbone = get_backbone(
            config_model_params=backbone_specific_params_to_use,
            model_type_str=model_type_to_use,
            image_size=image_size_to_use
        )
        # 修改权重加载逻辑
        # if 'backbone' in checkpoint_data: # 旧的逻辑
        #     backbone_instance.set_state_dict(checkpoint_data['backbone'])
        #     print(f"骨干网络 ({model_type_to_use}) 实例化并加载权重成功。输出特征维度: {feature_dim_from_backbone}")
        # else:
        #     print(f"警告: 模型检查点中未找到 'backbone' 权重。骨干网络 ({model_type_to_use}) 将使用随机初始化权重，这可能不是期望的行为。")
        
        # 新的加载逻辑，兼容 CombinedModel 的 state_dict
        if isinstance(checkpoint_data, dict): # 确保 checkpoint_data 是 state_dict
            backbone_state_dict = {k.replace('backbone.', '', 1): v for k, v in checkpoint_data.items() if k.startswith('backbone.')}
            if backbone_state_dict:
                backbone_instance.set_state_dict(backbone_state_dict)
                print(f"骨干网络 ({model_type_to_use}) 实例化并从检查点加载 'backbone.' 权重成功。输出特征维度: {feature_dim_from_backbone}")
            else:
                # 尝试将整个 state_dict 加载到 backbone (如果模型只包含 backbone)
                try:
                    backbone_instance.set_state_dict(checkpoint_data)
                    print(f"骨干网络 ({model_type_to_use}) 实例化并尝试直接加载整个检查点权重成功。")
                except Exception as e_direct_bb_load:
                    print(f"警告: 在检查点中未找到 'backbone.' 前缀的权重，且直接加载整个状态字典到骨干网络失败: {e_direct_bb_load}。骨干网络将使用随机初始化权重。")
        else:
            print(f"警告: 加载的检查点数据不是预期的字典格式。骨干网络 ({model_type_to_use}) 将使用随机初始化权重。")

    except Exception as e:
        print(f"创建或加载骨干网络 ({model_type_to_use}) 失败: {e}")
        return

    # 实例化头部模块
    try:
        head_module_instance = get_head(
            config_loss_params=head_specific_params_to_use,
            loss_type_str=loss_type_to_use,
            in_features=feature_dim_from_backbone,
            num_classes=num_classes_to_use
        )
        # 修改权重加载逻辑
        # if 'head' in checkpoint_data: # 旧的逻辑
        #     head_module_instance.set_state_dict(checkpoint_data['head'])
        #     print(f"头部模块 ({loss_type_to_use}) 实例化并加载权重成功。")
        # else:
        #     print(f"警告: 模型检查点中未找到 'head' 权重。头部模块 ({loss_type_to_use}) 将使用随机初始化权重。")

        # 新的加载逻辑，兼容 CombinedModel 的 state_dict
        if isinstance(checkpoint_data, dict): # 确保 checkpoint_data 是 state_dict
            head_state_dict = {k.replace('head.', '', 1): v for k, v in checkpoint_data.items() if k.startswith('head.')}
            if head_state_dict:
                head_module_instance.set_state_dict(head_state_dict)
                print(f"头部模块 ({loss_type_to_use}) 实例化并从检查点加载 'head.' 权重成功。")
            else:
                # 尝试将整个 state_dict 加载到 head (如果模型只包含 head，虽然不太可能)
                # 或者如果 loss_type_to_use 是一个不需要特定头部训练的类型 (例如，如果骨干直接输出分类)
                # 但通常对于分类任务，总会有一个head。
                print(f"警告: 在检查点中未找到 'head.' 前缀的权重。头部模块 ({loss_type_to_use}) 将使用随机初始化权重。")
        else:
             print(f"警告: 加载的检查点数据不是预期的字典格式。头部模块 ({loss_type_to_use}) 将使用随机初始化权重。")

    except Exception as e:
        print(f"创建或加载头部模块 ({loss_type_to_use}) 失败: {e}")
        return

    # 4. 执行评估循环
    backbone_instance.eval()
    head_module_instance.eval()

    total_loss = 0.0
    correct_samples = 0
    total_samples = 0
    
    print(f"\\n开始在验收集上评估模型 (共 {len(acceptance_loader.dataset)} 个样本)...")
    eval_start_time = time.time()

    with paddle.no_grad():
        for batch_id, data_batch in enumerate(acceptance_loader):
            images, labels = data_batch
            if labels.ndim > 1 and labels.shape[1] == 1:
                labels = paddle.squeeze(labels, axis=1)

            features = backbone_instance(images)
            loss_value, acc_output = head_module_instance(features, labels)

            if loss_value is not None:
                total_loss += loss_value.item() * images.shape[0]
            
            # acc_output 可能是 logits (ArcFaceHead返回的是logits) 或 经过softmax的概率
            # 对于分类准确率，通常在logits上取argmax
            if acc_output is not None:
                predicted_labels = paddle.argmax(acc_output, axis=1)
                correct_samples += (predicted_labels == labels).sum().item()
            
            total_samples += labels.shape[0]

            if (batch_id + 1) % config.get('log_interval', 20) == 0:
                current_acc = correct_samples / total_samples if total_samples > 0 else 0
                current_avg_loss = total_loss / total_samples if total_samples > 0 else 0
                print(f"  批次 {batch_id + 1}/{len(acceptance_loader)}: "
                      f"当前平均损失 {current_avg_loss:.4f}, 当前准确率 {current_acc:.4f}")

    eval_duration = time.time() - eval_start_time

    # 5. 计算并打印最终结果
    final_avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
    final_accuracy = correct_samples / total_samples if total_samples > 0 else float('nan')

    print("\\n--- 验收测试结果 ---")
    print(f"  模型文件: {model_weights_path}")
    print(f"  测试样本总数: {total_samples}")
    print(f"  正确预测样本数: {correct_samples}")
    print(f"  平均损失: {final_avg_loss:.4f}")
    print(f"  准确率: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"  评估耗时: {eval_duration:.2f} 秒")
    print("----------------------\\n")

    # (可选) 保存详细结果到文件
    results_to_save = {
        'model_path': model_weights_path,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_samples': total_samples,
        'correct_samples': correct_samples,
        'average_loss': final_avg_loss,
        'accuracy': final_accuracy,
        'config_used_for_eval': config.to_dict(), # 保存评估时使用的完整配置
        'model_config_snapshot': {
            'model_type': model_type_to_use,
            'loss_type': loss_type_to_use,
            'num_classes': num_classes_to_use,
            'image_size': image_size_to_use,
            'model_specific_params': backbone_specific_params_to_use,
            'loss_specific_params': head_specific_params_to_use
        }
    }
    
    # 确定结果文件名和路径
    # results_dir = config.get("results_save_dir", "acceptance_results") # 从配置读取或使用默认 # 旧逻辑
    results_dir_from_config = config.get("results_save_dir")
    results_dir = results_dir_from_config if results_dir_from_config is not None else "acceptance_results"

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"创建验收结果保存目录: {results_dir}")

    # 基于模型文件名生成结果文件名
    model_basename = os.path.splitext(os.path.basename(model_weights_path))[0]
    result_filename = f"acceptance_summary_{model_basename}_{time.strftime('%Y%m%d%H%M%S')}.json"
    result_filepath = os.path.join(results_dir, result_filename)

    try:
        with open(result_filepath, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=4, ensure_ascii=False, default=lambda o: '<not serializable>')
        print(f"验收测试摘要已保存至: {result_filepath}")
    except Exception as e:
        print(f"警告: 保存验收测试摘要到 {result_filepath} 失败: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='人脸识别模型验收测试脚本')

    # 核心控制参数
    parser.add_argument('--config_path', type=str, default=None,
                        help='指定YAML配置文件的路径。')
    parser.add_argument('--active_config', type=str, default=None,
                        help='通过命令行指定要激活的配置块名称，覆盖YAML文件中的active_config设置。')
    parser.add_argument('--trained_model_path', type=str, required=False,
                        help='必需：指定已训练模型的权重文件路径 (.pdparams)。如果配置文件中也指定了，命令行优先。')
    
    # 可覆盖配置文件的参数
    parser.add_argument('--use_gpu', action=argparse.BooleanOptionalAction, default=None,
                        help='是否使用GPU。覆盖配置文件设置。')
    parser.add_argument('--batch_size', type=int, help='验收测试批大小 (覆盖配置文件)。')
    parser.add_argument('--image_size', type=int, help='输入图像尺寸 (覆盖配置文件)。')
    parser.add_argument('--num_classes', type=int, help='类别数 (覆盖配置文件, 影响模型加载)。')
    parser.add_argument('--data_dir', type=str, help='数据集根目录 (覆盖配置文件)。')
    parser.add_argument('--class_name', type=str, help='数据集子目录名 (覆盖配置文件)。')
    parser.add_argument('--log_interval', type=int, help='打印日志的间隔批次数 (覆盖配置文件)。')
    parser.add_argument('--results_save_dir', type=str, help='验收结果保存目录 (覆盖配置文件)。')

    cmd_line_args = parser.parse_args()

    # 加载配置
    final_config = load_config(
        default_yaml_path='configs/default_config.yaml',
        cmd_args_namespace=cmd_line_args
    )

    # 确保 trained_model_path 已提供
    if not cmd_line_args.trained_model_path and not final_config.get('trained_model_path'):
        parser.error("错误: 必须通过 --trained_model_path 命令行参数或在配置文件中 "
                     "(e.g., global_settings.trained_model_path or active_config.trained_model_path) "
                     "指定模型权重路径。")
    
    print("\\n--- 最终生效的验收测试配置 ---")
    print(f"  模型路径: {cmd_line_args.trained_model_path or final_config.get('trained_model_path')}")
    print(f"  使用GPU: {final_config.use_gpu}")
    print(f"  批大小: {final_config.batch_size}")
    print(f"  类别数: {final_config.num_classes}") # 确保这里能正确打印
    print(f"  图像尺寸: {final_config.image_size}")
    print(f"  结果保存目录: {final_config.get('results_save_dir', 'acceptance_results')}")
    print("---------------------------------\\n")
    
    if final_config.get('num_classes') is None: # 再次检查 num_classes
        parser.error("错误: 最终配置中未能确定 'num_classes'。请检查YAML文件和命令行参数 --num_classes。")

    run_acceptance_evaluation(final_config, cmd_line_args) 