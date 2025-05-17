import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def visualize(csv_path="acceptance_results.csv", output_dir="acceptance_plots"):
    """
    读取验收结果CSV，生成可视化图表。
    """
    if not os.path.exists(csv_path):
        print(f"错误：未找到结果文件 {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
        print(f"成功读取 {len(df)} 条结果。")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return

    # 确保准确率是数值类型，处理可能的非数值条目 (如 N/A, ERROR)
    df['acceptance_accuracy'] = pd.to_numeric(df['acceptance_accuracy'], errors='coerce')
    df_valid = df.dropna(subset=['acceptance_accuracy'])

    if df_valid.empty:
        print("没有有效的准确率数据可供可视化。")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 示例可视化：不同损失类型的平均准确率
    plt.figure(figsize=(10, 6))
    sns.barplot(x='loss_type', y='acceptance_accuracy', data=df_valid)
    plt.title('Average Acceptance Accuracy by Loss Type')
    plt.ylabel('Accuracy')
    plt.xlabel('Loss Type')
    plt.ylim(0, 1.0) # 准确率范围0-1
    plot_path = os.path.join(output_dir, 'avg_accuracy_by_loss_type.png')
    plt.savefig(plot_path)
    print(f"图表已保存到 {plot_path}")

    # 示例可视化：按实验名称（完整配置组合）排序的准确率
    # 可以按准确率降序排序以便找到最佳配置
    df_sorted = df_valid.sort_values(by='acceptance_accuracy', ascending=False)
    plt.figure(figsize=(12, len(df_sorted) * 0.5)) # 根据条目数调整图表高度
    sns.barplot(x='acceptance_accuracy', y='experiment_name', data=df_sorted)
    plt.title('Acceptance Accuracy by Configuration')
    plt.xlabel('Accuracy')
    plt.ylabel('Configuration Name')
    plt.xlim(0, 1.0)
    plt.tight_layout() # 自动调整布局以防止标签重叠
    plot_path = os.path.join(output_dir, 'accuracy_by_config.png')
    plt.savefig(plot_path)
    print(f"图表已保存到 {plot_path}")

    # 根据需要添加更多图表，例如按optimizer, lr_schedule等分组

    # --- 添加更多可视化图表 ---

    # 1. 按骨干网络 (model_type) 对比平均准确率
    if 'model_type' in df_valid.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model_type', y='acceptance_accuracy', data=df_valid)
        plt.title('Average Acceptance Accuracy by Backbone Type')
        plt.ylabel('Accuracy')
        plt.xlabel('Backbone Type')
        plt.ylim(0, 1.0)
        plot_path = os.path.join(output_dir, 'avg_accuracy_by_model_type.png')
        plt.savefig(plot_path)
        print(f"图表已保存到 {plot_path}")

    # 2. 按优化器 (optimizer) 对比平均准确率
    if 'optimizer' in df_valid.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='optimizer', y='acceptance_accuracy', data=df_valid)
        plt.title('Average Acceptance Accuracy by Optimizer')
        plt.ylabel('Accuracy')
        plt.xlabel('Optimizer')
        plt.ylim(0, 1.0)
        plot_path = os.path.join(output_dir, 'avg_accuracy_by_optimizer.png')
        plt.savefig(plot_path)
        print(f"图表已保存到 {plot_path}")

    # 3. 按学习率调度器 (lr_schedule) 对比平均准确率
    if 'lr_schedule' in df_valid.columns:
        plt.figure(figsize=(14, 6)) # 调度器类型可能较多，增加图表宽度
        sns.barplot(x='lr_schedule', y='acceptance_accuracy', data=df_valid)
        plt.title('Average Acceptance Accuracy by LR Scheduler')
        plt.ylabel('Accuracy')
        plt.xlabel('LR Scheduler Type')
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45, ha='right') # 旋转x轴标签以避免重叠
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'avg_accuracy_by_lr_schedule.png')
        plt.savefig(plot_path)
        print(f"图表已保存到 {plot_path}")

    # 4. 按学习率 (learning_rate) 和权重衰减 (weight_decay) 对比准确率 (散点图)
    # 确保 learning_rate 和 weight_decay 是数值类型
    df_valid['learning_rate'] = pd.to_numeric(df_valid['learning_rate'], errors='coerce')
    df_valid['weight_decay'] = pd.to_numeric(df_valid['weight_decay'], errors='coerce')
    df_numeric_params = df_valid.dropna(subset=['learning_rate', 'weight_decay'])

    if not df_numeric_params.empty:
        # 学习率 vs 准确率
        if len(df_numeric_params['learning_rate'].unique()) > 1:
            plt.figure(figsize=(10, 6))
            # 可以使用hue来区分不同的loss_type或model_type
            sns.scatterplot(x='learning_rate', y='acceptance_accuracy', hue='loss_type', style='model_type', data=df_numeric_params)
            plt.title('Acceptance Accuracy vs. Learning Rate')
            plt.ylabel('Accuracy')
            plt.xlabel('Learning Rate')
            plt.ylim(0, 1.0)
            plt.xscale('log') # 学习率通常跨越多个数量级，使用log缩放
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'accuracy_vs_learning_rate.png')
            plt.savefig(plot_path)
            print(f"图表已保存到 {plot_path}")

        # 权重衰减 vs 准确率
        if len(df_numeric_params['weight_decay'].unique()) > 1:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='weight_decay', y='acceptance_accuracy', hue='loss_type', style='model_type', data=df_numeric_params)
            plt.title('Acceptance Accuracy vs. Weight Decay')
            plt.ylabel('Accuracy')
            plt.xlabel('Weight Decay')
            plt.ylim(0, 1.0)
            plt.xscale('log') # 权重衰减也通常跨越多个数量级
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'accuracy_vs_weight_decay.png')
            plt.savefig(plot_path)
            print(f"图表已保存到 {plot_path}")

    print(f"所有图表已生成并保存到目录: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='可视化验收结果脚本')
    parser.add_argument('--csv_path', type=str, default='acceptance_results.csv',
                        help='验收结果CSV文件的路径')
    parser.add_argument('--output_dir', type=str, default='acceptance_plots',
                        help='图表保存目录')
    cmd_args = parser.parse_args()

    visualize(csv_path=cmd_args.csv_path, output_dir=cmd_args.output_dir)
