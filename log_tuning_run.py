# log_tuning_run.py
# 该脚本用于记录机器学习实验的参数调优过程，并将结果追加到CSV文件中。

import csv
import os
import argparse
from datetime import datetime

# 定义CSV文件的列名 (表头)
CSV_FIELD_NAMES = [
    'timestamp',
    'experiment_name',
    'model_type',
    'loss_type',
    'changed_params',
    'lr_schedule',
    'optimizer',
    'batch_size',
    'epochs',
    'acceptance_accuracy',
    'test_accuracy',
    'git_hash',
    'notes'
]

def log_experiment(
    csv_path: str,
    experiment_name: str,
    model_type: str,
    loss_type: str,
    changed_params: str, # 格式如 "param1=value1,param2=value2;param3=new_value"
    lr_schedule: str,
    optimizer: str,
    batch_size: int,
    epochs: int,
    acceptance_accuracy: float,
    test_accuracy: float = None, # 可选
    git_hash: str = None,    # 可选
    notes: str = None        # 可选
):
    """
    将一次实验的参数和结果记录到CSV文件中。

    Args:
        csv_path (str): CSV文件的保存路径。
        experiment_name (str): 实验的简短描述或名称。
        model_type (str): 使用的模型类型 (e.g., 'vgg', 'resnet')。
        loss_type (str): 使用的损失类型 (e.g., 'cross_entropy', 'arcface')。
        changed_params (str): 描述本次实验中主要更改的参数及其新值。
                              例如: "feature_dim=256,lr=0.0005" 或 "ResNet_blocks=4;ArcFace_margin=0.4"
        lr_schedule (str): 使用的学习率调度策略 (e.g., 'StepDecay', 'CosineAnnealing')。
        optimizer (str): 使用的优化器 (e.g., 'AdamW', 'Momentum')。
        batch_size (int): 训练时使用的批处理大小。
        epochs (int): 训练的总轮数。
        acceptance_accuracy (float): 在验收集上达到的准确率。
        test_accuracy (float, optional): 在测试集上达到的准确率。默认为 None。
        git_hash (str, optional): 与本次实验相关的Git commit hash。默认为 None。
        notes (str, optional): 关于本次实验的任何额外备注或发现。默认为 None。
    """
    file_exists = os.path.isfile(csv_path)

    row_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'experiment_name': experiment_name,
        'model_type': model_type,
        'loss_type': loss_type,
        'changed_params': changed_params,
        'lr_schedule': lr_schedule,
        'optimizer': optimizer,
        'batch_size': batch_size,
        'epochs': epochs,
        'acceptance_accuracy': f"{acceptance_accuracy:.4f}" if acceptance_accuracy is not None else 'N/A',
        'test_accuracy': f"{test_accuracy:.4f}" if test_accuracy is not None else 'N/A',
        'git_hash': git_hash if git_hash else 'N/A',
        'notes': notes if notes else ''
    }

    try:
        with open(csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELD_NAMES)
            if not file_exists: # 如果文件是新创建的，则写入表头
                writer.writeheader()
            writer.writerow(row_data)
        print(f"实验记录已成功追加到: {csv_path}")
    except IOError as e:
        print(f"错误: 无法写入CSV文件 {csv_path}: {e}")
    except Exception as e:
        print(f"记录实验时发生未知错误: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="记录机器学习实验参数和结果到CSV文件。")

    parser.add_argument("--csv_path", type=str, default="tuning_log.csv",
                        help="CSV文件的路径 (默认: tuning_log.csv)")
    parser.add_argument("--experiment_name", type=str, required=True,
                        help="实验的简短描述或名称 (必需)")
    parser.add_argument("--model_type", type=str, required=True,
                        help="使用的模型类型 (e.g., vgg, resnet) (必需)")
    parser.add_argument("--loss_type", type=str, required=True,
                        help="使用的损失/头部类型 (e.g., cross_entropy, arcface) (必需)")
    parser.add_argument("--changed_params", type=str, required=True,
                        help='本次实验主要更改的参数及其值。格式: "param1=value1;param2=value2" (必需)')
    parser.add_argument("--lr_schedule", type=str, required=True,
                        help="学习率调度策略 (e.g., StepDecay) (必需)")
    parser.add_argument("--optimizer", type=str, required=True,
                        help="优化器 (e.g., AdamW, Momentum) (必需)")
    parser.add_argument("--batch_size", type=int, required=True,
                        help="批处理大小 (必需)")
    parser.add_argument("--epochs", type=int, required=True,
                        help="训练轮数 (必需)")
    parser.add_argument("--acceptance_accuracy", type=float, required=True,
                        help="在验收集上的准确率 (必需)")
    parser.add_argument("--test_accuracy", type=float, default=None,
                        help="(可选) 在测试集上的准确率")
    parser.add_argument("--git_hash", type=str, default=None,
                        help="(可选) 本次实验的 Git commit hash")
    parser.add_argument("--notes", type=str, default=None,
                        help="(可选) 关于本次实验的额外备注")

    args = parser.parse_args()

    log_experiment(
        csv_path=args.csv_path,
        experiment_name=args.experiment_name,
        model_type=args.model_type,
        loss_type=args.loss_type,
        changed_params=args.changed_params,
        lr_schedule=args.lr_schedule,
        optimizer=args.optimizer,
        batch_size=args.batch_size,
        epochs=args.epochs,
        acceptance_accuracy=args.acceptance_accuracy,
        test_accuracy=args.test_accuracy,
        git_hash=args.git_hash,
        notes=args.notes
    ) 