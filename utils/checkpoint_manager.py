# utils/checkpoint_manager.py
# 该模块包含 CheckpointManager 类，用于处理模型训练过程中的检查点保存和加载。
# 它能保存和恢复模型的权重、优化器状态以及学习率调度器状态，
# 同时管理最佳模型和训练元数据（如当前epoch、最佳准确率等）。

import os
import paddle
import json
import time # Optional, for timestamp if needed, though we use it in train.py logdir

class CheckpointManager:
    """
    负责模型训练检查点的保存和加载。
    管理最新的检查点、最佳模型及其元数据。
    """
    def __init__(self, model_save_dir: str, model_name: str):
        """
        初始化检查点管理器。

        Args:
            model_save_dir (str): 保存模型和检查点文件的根目录。
            model_name (str): 模型的名称，用于生成文件名。
                                 通常包含模型类型、损失函数、调度器等信息。
                                 例如: "vgg_ce_steplr_gpu_manual"
        """
        self.model_save_dir = model_save_dir
        self.model_name = model_name

        # Ensure the save directory exists
        os.makedirs(self.model_save_dir, exist_ok=True)

        # Define standard paths for the latest checkpoint and best model
        self.checkpoint_path = os.path.join(self.model_save_dir, f"checkpoint_{self.model_name}.pdparams")
        self.metadata_path = os.path.join(self.model_save_dir, f"checkpoint_{self.model_name}.json")

        self.best_model_path = os.path.join(self.model_save_dir, f"best_model_{self.model_name}.pdparams")
        self.best_metadata_path = os.path.join(self.model_save_dir, f"best_model_{self.model_name}.json")

        print(f"训练检查点将保存至: {self.checkpoint_path} (元数据: {self.metadata_path})")
        print(f"最佳模型将保存至: {self.best_model_path} (元数据: {self.best_metadata_path})")


    def save_checkpoint(self, model: paddle.nn.Layer, optimizer: paddle.optimizer.Optimizer, lr_scheduler, metadata: dict, is_best: bool = False):
        """
        保存当前的训练检查点。

        Args:
            model (paddle.nn.Layer): 当前的模型实例。
            optimizer (paddle.optimizer.Optimizer): 当前的优化器实例。
            lr_scheduler: 当前的学习率调度器实例 (可以是 None)。
            metadata (dict): 包含训练状态的元数据字典 (e.g., epoch, best_acc)。
            is_best (bool, optional): 如果当前模型是迄今为止最好的模型，则设置为 True。
                                      默认为 False。
        """
        try:
            # Save model state dict
            paddle.save(model.state_dict(), self.checkpoint_path)
            # Save optimizer state dict
            paddle.save(optimizer.state_dict(), self.checkpoint_path.replace('.pdparams', '_optimizer.pdparams'))
            # Save lr_scheduler state dict if it exists and has state_dict method
            if hasattr(lr_scheduler, 'state_dict') and lr_scheduler.state_dict() is not None:
                 paddle.save(lr_scheduler.state_dict(), self.checkpoint_path.replace('.pdparams', '_lr_scheduler.pdparams'))
            else:
                 # Remove old lr_scheduler checkpoint if it exists and we are not saving a new one
                 if os.path.exists(self.checkpoint_path.replace('.pdparams', '_lr_scheduler.pdparams')):
                      os.remove(self.checkpoint_path.replace('.pdparams', '_lr_scheduler.pdparams'))
                      print(f"警告: LR调度器没有 state_dict 或 state_dict() 返回 None，未保存其状态。已移除旧的调度器检查点文件 (如果存在)。")

            # Save metadata
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)

            print(f"检查点及元数据已保存到: {self.checkpoint_path}, {self.metadata_path}")

            # If it's the best model so far, save a copy
            if is_best:
                paddle.save(model.state_dict(), self.best_model_path)
                 # Optionally save optimizer/lr_scheduler for best model if needed for later analysis,
                 # but typically best model only saves the weights.
                with open(self.best_metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=4)
                print(f"最佳模型已更新并保存到: {self.best_model_path}, {self.best_metadata_path} (Epoch {metadata.get('epoch', '?')}, Test Acc: {metadata.get('previous_best_acc', 0.0):.4f} -> {metadata.get('best_acc', 0.0):.4f})")

        except Exception as e:
            print(f"保存检查点失败: {e}")


    def load_checkpoint(self, model: paddle.nn.Layer, optimizer: paddle.optimizer.Optimizer, lr_scheduler, resume: bool = True) -> tuple[int, float, dict]:
        """
        加载最新的检查点以恢复训练。

        Args:
            model (paddle.nn.Layer): 需要加载状态字典的模型实例。
            optimizer (paddle.optimizer.Optimizer): 需要加载状态字典的优化器实例。
            lr_scheduler: 需要加载状态字典的学习率调度器实例 (可以是 None)。
            resume (bool, optional): 是否尝试加载检查点。如果为 False，则不加载。默认为 True。

        Returns:
            tuple[int, float, dict]: 返回开始训练的 epoch (通常从检查点元数据中获取)，
                                     加载的最佳准确率 (从检查点元数据中获取)，
                                     以及完整的加载的元数据字典。
                                     如果 resume=False 或检查点不存在，则返回 (0, 0.0, {})。
        """
        start_epoch = 0
        best_acc = 0.0
        loaded_meta_data = {}
        checkpoint_exists = os.path.exists(self.checkpoint_path)
        metadata_exists = os.path.exists(self.metadata_path)

        if resume and checkpoint_exists and metadata_exists:
            print(f"正在从检查点加载: {self.checkpoint_path}")
            try:
                # Load model state dict
                model_state_dict = paddle.load(self.checkpoint_path)
                model.set_state_dict(model_state_dict)

                # Load optimizer state dict
                optimizer_checkpoint_path = self.checkpoint_path.replace('.pdparams', '_optimizer.pdparams')
                if os.path.exists(optimizer_checkpoint_path):
                     optimizer_state_dict = paddle.load(optimizer_checkpoint_path)
                     optimizer.set_state_dict(optimizer_state_dict)
                else:
                     print(f"警告: 优化器检查点文件 {optimizer_checkpoint_path} 不存在，未加载优化器状态。")

                # Load lr_scheduler state dict
                lr_scheduler_checkpoint_path = self.checkpoint_path.replace('.pdparams', '_lr_scheduler.pdparams')
                if os.path.exists(lr_scheduler_checkpoint_path):
                     try:
                         lr_scheduler_state_dict = paddle.load(lr_scheduler_checkpoint_path)
                         if hasattr(lr_scheduler, 'set_state_dict') and lr_scheduler_state_dict is not None:
                              lr_scheduler.set_state_dict(lr_scheduler_state_dict)
                         else:
                              print(f"警告: LR调度器没有 set_state_dict 方法或状态字典为空，未加载其状态。")
                     except Exception as e:
                         print(f"警告: 加载LR调度器检查点文件 {lr_scheduler_checkpoint_path} 失败: {e}。未加载调度器状态。")
                else:
                     print(f"警告: LR调度器检查点文件 {lr_scheduler_checkpoint_path} 不存在，未加载调度器状态。")


                # Load metadata
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    loaded_meta_data = json.load(f)

                start_epoch = loaded_meta_data.get('epoch', 0) # Resume from the next epoch
                best_acc = loaded_meta_data.get('best_acc', 0.0)
                print(f"成功从检查点恢复。将从 epoch {start_epoch} 开始训练，上次最佳准确率: {best_acc:.4f}")

            except Exception as e:
                print(f"加载检查点失败: {e}. 将从头开始训练。")
                start_epoch = 0
                best_acc = 0.0
                loaded_meta_data = {}
        else:
            if resume: # User specified resume but checkpoint does not exist
                 print(f"警告: 用户指定了 --resume 但检查点文件 {self.checkpoint_path} 或元数据文件 {self.metadata_path} 不存在。将从头开始训练。")
            else: # User did not specify resume
                 print(f"未指定 --resume 或未找到检查点。将从头开始训练。")

        # Add previous best acc to metadata for logging the update
        loaded_meta_data['previous_best_acc'] = best_acc

        return start_epoch, best_acc, loaded_meta_data

# Example usage within a training script:
# checkpoint_manager = CheckpointManager(model_save_dir='model', model_name='my_model_vgg_ce')
# start_epoch, best_acc = checkpoint_manager.load_checkpoint(model, optimizer, lr_scheduler, resume=True)
# ... training loop ...
# checkpoint_manager.save_checkpoint(model, optimizer, lr_scheduler, {'epoch': current_epoch, 'best_acc': current_best_acc}, is_best=is_current_best)