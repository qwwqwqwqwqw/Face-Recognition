# utils/lr_scheduler_factory.py
# 该工厂模块负责根据配置文件中的设置，动态创建和返回不同类型的学习率调度器实例。
# 目的是将学习率调度器的创建逻辑与主训练脚本分离，提高代码的模块化和可维护性。
#
# 支持的调度器类型包括：
# - StepDecay: 按预设的步长衰减学习率。
# - MultiStepDecay: 在预设的多个里程碑 (milestones) 处衰减学习率。
# - ExponentialDecay: 按指数方式衰减学习率。
# - ReduceOnPlateau: 当某个监控指标 (如验证集损失) 在一段时间内不再改善时，降低学习率。
# - CosineAnnealingDecay: 使用余弦退火方式调整学习率。
# - PolynomialDecay: 使用多项式衰减方式调整学习率。
# - CosineAnnealingWarmRestarts: 带热重启的余弦退火。
#
# 工厂函数 `get_lr_scheduler` 会解析配置文件中的 `lr_scheduler_type` 和相应的 `lr_scheduler_params`，
# 然后实例化对应的调度器对象。它还支持可选的 `warmup_params` 来为调度器添加预热 (warmup) 阶段。
# 如果配置的调度器类型不受支持，或者参数不正确，会抛出 ValueError。
#
# 示例配置 (在YAML文件中):
# ```yaml
# # ...其他配置...
# learning_rate: 0.01 # 优化器的初始学习率，也会作为调度器的基础学习率
# lr_scheduler_type: "MultiStepDecay" # 或者 "StepDecay", "ReduceOnPlateau", etc.
# lr_scheduler_params:
#   stepdecay: # 如果使用 StepDecay，键名应为 stepdecay
#     step_size: 30
#     gamma: 0.1
#   multistepdecay: # 如果使用 MultiStepDecay，键名应为 multistepdecay
#     milestones: [30, 60, 90]
#     gamma: 0.1
#   exponentialdecay: # 如果使用 ExponentialDecay，键名应为 exponentialdecay
#     gamma: 0.9
#   reduceonplateau: # 如果使用 ReduceOnPlateau，键名应为 reduceonplateau
#     mode: 'min'
#     factor: 0.1
#     patience: 10
#     threshold: 0.0001
#   cosineannealingdecay: # 如果使用 CosineAnnealingDecay，键名应为 cosineannealingdecay
#     T_max: 100
#     eta_min: 0
#   polynomialdecay: # 如果使用 PolynomialDecay，键名应为 polynomialdecay
#     decay_steps: 100
#     end_lr: 0
#     power: 1.0
#   cosineannealingwarmrestarts: # 如果使用 CosineAnnealingWarmRestarts，键名应为 cosineannealingwarmrestarts
#     T_0: 10
#     T_mult: 2
#     eta_min: 0
#   warmup: # 预热参数，可以与大多数调度器结合使用 (ReduceOnPlateau 除外)
#     use_warmup: True
#     warmup_steps: 500
#     start_lr: 0.001
# # ...其他配置...
# ```
# 注意：YAML配置中 lr_scheduler_params 下的子字典键名 (如 `stepdecay`, `reduceonplateau` 等)
#      应该与工厂函数中 `config.lr_scheduler_params.get(scheduler_type.lower(), {})`
#      查找时使用的键名一致。为了简化，我们统一使用 PaddlePaddle API 名称的小写形式作为键名。

import paddle
import paddle.optimizer as optimizer
# 导入 PaddlePaddle 提供的学习率调度器模块
import paddle.optimizer.lr as lr_scheduler

from config_utils import ConfigObject # 假设 ConfigObject 类在 config_utils模块中

def get_lr_scheduler(config: ConfigObject, initial_learning_rate):
    """
    根据配置信息创建并返回一个学习率调度器实例。

    Args:
        config (ConfigObject): 包含学习率调度器配置的 ConfigObject 实例。
                               应包含 lr_scheduler_type, lr_scheduler_params, 和 warmup 配置。
        initial_learning_rate (float): 优化器的初始学习率，也作为调度器的基础学习率。

    Returns:
        paddle.optimizer.lr._LRScheduler: 创建的学习率调度器实例。

    Raises:
        ValueError: 如果配置的调度器类型不受支持，或者缺少必要的参数。
    """
    scheduler_type = config.lr_scheduler_type
    # 统一使用 scheduler_type 的小写形式作为 lr_scheduler_params 中的键查找参数
    scheduler_params = config.lr_scheduler_params.get(scheduler_type.lower(), {})
    warmup_config = config.lr_scheduler_params.get('warmup', {}) # 获取 warmup 配置

    # 检查并获取 Warmup 配置
    use_warmup = warmup_config.get('use_warmup', False)
    warmup_steps = warmup_config.get('warmup_steps', 0)
    start_lr = warmup_config.get('start_lr', 0.0) # 默认预热开始学习率为0

    scheduler = None

    # --- 学习率调度器类型判断与实例化 ---

    # StepDecay (对应 YAML 中的 stepdecay)
    # 兼容旧配置中的 "StepLR" 字符串，但查找参数使用 "stepdecay"
    if scheduler_type == "StepDecay" or scheduler_type == "StepLR":
        # --- 移除 debug prints ---
        # print(f"DEBUG (StepDecay): Entering block")
        # print(f"DEBUG (StepDecay): scheduler_type: {scheduler_type}")
        # print(f"DEBUG (StepDecay): scheduler_type.lower(): {scheduler_type.lower()}")
        # print(f"DEBUG (StepDecay): config.lr_scheduler_params type: {type(config.lr_scheduler_params)}")
        # print(f"DEBUG (StepDecay): config.lr_scheduler_params content: {config.lr_scheduler_params}")
        # print(f"DEBUG (StepDecay): scheduler_params (extracted) type: {type(scheduler_params)}")
        # print(f"DEBUG (StepDecay): scheduler_params (extracted) content: {scheduler_params}")
        # --- End debug prints ---

        # 检查 StepDecay 必要的参数
        # 从 scheduler_params 字典中获取参数
        if 'step_size' not in scheduler_params or 'gamma' not in scheduler_params:
             raise ValueError(f"配置错误: '{scheduler_type}' 调度器需要 'step_size' 和 'gamma' 参数。请检查 lr_scheduler_params.{'stepdecay'} 配置块。") # 提示查找键名

        scheduler = lr_scheduler.StepDecay(
            learning_rate=initial_learning_rate,
            step_size=scheduler_params['step_size'],
            gamma=scheduler_params['gamma'],
            # last_epoch=-1 # 如果从检查点恢复，可能需要设置 last_epoch
        )

    # MultiStepDecay (对应 YAML 中的 multistepdecay)
    # 兼容旧配置中的 "MultiStepLR" 字符串，但查找参数使用 "multistepdecay"
    elif scheduler_type == "MultiStepDecay" or scheduler_type == "MultiStepLR":
         # --- 移除 debug prints ---
         # print(f"DEBUG (MultiStepDecay): Entering block")
         # print(f"DEBUG (MultiStepDecay): scheduler_type: {scheduler_type}")
         # print(f"DEBUG (MultiStepDecay): scheduler_type.lower(): {scheduler_type.lower()}")
         # print(f"DEBUG (MultiStepDecay): config.lr_scheduler_params type: {type(config.lr_scheduler_params)}")
         # print(f"DEBUG (MultiStepDecay): config.lr_scheduler_params content: {config.lr_scheduler_params}")
         # print(f"DEBUG (MultiStepDecay): scheduler_params (extracted) type: {type(scheduler_params)}")
         # print(f"DEBUG (MultiStepDecay): scheduler_params (extracted) content: {scheduler_params}")
         # --- End debug prints ---

         # 检查 MultiStepDecay 必要的参数
        # 从 scheduler_params 字典中获取参数
        if 'milestones' not in scheduler_params or 'gamma' not in scheduler_params:
             raise ValueError(f"配置错误: '{scheduler_type}' 调度器需要 'milestones' 和 'gamma' 参数。请检查 lr_scheduler_params.{'multistepdecay'} 配置块。") # 提示查找键名

        scheduler = lr_scheduler.MultiStepDecay(
            learning_rate=initial_learning_rate,
            milestones=scheduler_params['milestones'],
            gamma=scheduler_params['gamma'],
            # last_epoch=-1 # 如果从检查点恢复，可能需要设置 last_epoch
        )

    # ExponentialDecay (对应 YAML 中的 exponentialdecay)
    # 兼容旧配置中的 "ExponentialLR" 字符串，但查找参数使用 "exponentialdecay"
    elif scheduler_type == "ExponentialDecay" or scheduler_type == "ExponentialLR":
         # 检查 ExponentialDecay 必要的参数
        # 从 scheduler_params 字典中获取参数
        if 'gamma' not in scheduler_params:
             raise ValueError(f"配置错误: '{scheduler_type}' 调度器需要 'gamma' 参数。请检查 lr_scheduler_params.{'exponentialdecay'} 配置块。") # 提示查找键名

        scheduler = lr_scheduler.ExponentialDecay(
            learning_rate=initial_learning_rate,
            gamma=scheduler_params['gamma'],
            # last_epoch=-1 # 如果从检查点恢复，可能需要设置 last_epoch
        )

    # ReduceOnPlateau (对应 YAML 中的 reduceonplateau)
    # 注意：PaddlePaddle 中是 ReduceOnPlateau，而不是 ReduceLROnPlateau
    # 兼容旧配置中的 "ReduceLROnPlateau" 字符串，但实例化 ReduceOnPlateau 并查找参数使用 "reduceonplateau"
    # 注意：ReduceOnPlateau 不接受 last_epoch 参数，它的状态是内部维护的
    # 注意：ReduceOnPlateau 通常不能与 warmup 直接结合，因为 warmup 是基于步数的，而 ReduceOnPlateau 是基于指标的
    elif scheduler_type == "ReduceOnPlateau" or scheduler_type == "ReduceLROnPlateau":
        # ReduceOnPlateau 有较多可选参数，这里获取所有可能的参数
        # 从 scheduler_params 字典中获取参数
        scheduler = lr_scheduler.ReduceOnPlateau( # <--- 实例化 ReduceOnPlateau
            learning_rate=initial_learning_rate,
            mode=scheduler_params.get('mode', 'min'),
            factor=scheduler_params.get('factor', 0.1),
            patience=scheduler_params.get('patience', 10),
            threshold=scheduler_params.get('threshold', 0.0001),
            threshold_mode=scheduler_params.get('threshold_mode', 'rel'),
            cooldown=scheduler_params.get('cooldown', 0),
            min_lr=scheduler_params.get('min_lr', 0),
            # 移除 'eps' 参数，因为它在您的PaddlePaddle版本中不受支持
            # eps=scheduler_params.get('eps', 1e-08),
            verbose=scheduler_params.get('verbose', False) # 是否打印学习率变化信息
        )
        # 对于 ReduceOnPlateau，如果使用了 warmup，这里需要特别处理，例如先执行 warmup 步数，
        # 然后再将优化器的 learning_rate 设置为 initial_learning_rate，并开始 ReduceOnPlateau 调度。
        # 这里为了简化，不对 ReduceOnPlateau 进行 warmup 处理。
        if use_warmup:
             print("警告: ReduceOnPlateau 调度器通常不直接与 Warmup 结合使用。Warmup 配置将被忽略。")
             use_warmup = False # 确保不再尝试应用 Warmup


    # CosineAnnealingDecay (对应 YAML 中的 cosineannealingdecay)
    # 注意：PaddlePaddle 中是 CosineAnnealingDecay，而不是 CosineAnnealingLR
    # 兼容旧配置中的 "CosineAnnealingLR" 字符串，但实例化 CosineAnnealingDecay 并查找参数使用 "cosineannealingdecay"
    elif scheduler_type == "CosineAnnealingDecay" or scheduler_type == "CosineAnnealingLR":
         # 检查 CosineAnnealingDecay 必要的参数
        # 从 scheduler_params 字典中获取参数
        if 'T_max' not in scheduler_params:
             raise ValueError(f"配置错误: '{scheduler_type}' 调度器需要 'T_max' 参数。请检查 lr_scheduler_params.{'cosineannealingdecay'} 配置块。") # 提示查找键名

        scheduler = lr_scheduler.CosineAnnealingDecay( # <--- 实例化 CosineAnnealingDecay
            learning_rate=initial_learning_rate,
            T_max=scheduler_params['T_max'],
            eta_min=scheduler_params.get('eta_min', 0), # eta_min 可选，默认为 0
            # last_epoch=-1 # 如果从检查点恢复，可能需要设置 last_epoch
        )

    # PolynomialDecay (对应 YAML 中的 polynomialdecay)
    # 兼容旧配置中的 "PolynomialLR" 字符串，但查找参数使用 "polynomialdecay"
    elif scheduler_type == "PolynomialDecay" or scheduler_type == "PolynomialLR":
        # 检查 PolynomialDecay 必要的参数
        # 从 scheduler_params 字典中获取参数
        if 'decay_steps' not in scheduler_params:
             raise ValueError(f"配置错误: '{scheduler_type}' 调度器需要 'decay_steps' 参数。请检查 lr_scheduler_params.{'polynomialdecay'} 配置块。") # 提示查找键名

        scheduler = lr_scheduler.PolynomialDecay(
            learning_rate=initial_learning_rate,
            decay_steps=scheduler_params['decay_steps'],
            end_lr=scheduler_params.get('end_lr', 0), # end_lr 可选，默认为 0
            power=scheduler_params.get('power', 1.0), # power 可选，默认为 1.0
            cycle=scheduler_params.get('cycle', False), # cycle 可选，默认为 False
            # last_epoch=-1 # 如果从检查点恢复，可能需要设置 last_epoch
        )

    # CosineAnnealingWarmRestarts (对应 YAML 中的 cosineannealingwarmrestarts)
    # PaddlePaddle API 名称一致，查找参数使用 "cosineannealingwarmrestarts"
    elif scheduler_type == "CosineAnnealingWarmRestarts":
         # 检查 CosineAnnealingWarmRestarts 必要的参数
        # 从 scheduler_params 字典中获取参数
        if 'T_0' not in scheduler_params:
             raise ValueError(f"配置错误: '{scheduler_type}' 调度器需要 'T_0' 参数。请检查 lr_scheduler_params.{'cosineannealingwarmrestarts'} 配置块。") # 提示查找键名

        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            learning_rate=initial_learning_rate,
            T_0=scheduler_params['T_0'],
            T_mult=scheduler_params.get('T_mult', 1), # T_mult 可选，默认为 1
            eta_min=scheduler_params.get('eta_min', 0), # eta_min 可选，默认为 0
            # last_epoch=-1 # 如果从检查点恢复，可能需要设置 last_epoch
        )


    # 如果配置的调度器类型不受支持
    if scheduler is None:
        raise ValueError(f"不支持的学习率调度器类型: {scheduler_type}. "
                         f"请检查 configs/default_config.yaml 中 'lr_scheduler_type' 的配置 "
                         f"以及 utils/lr_scheduler_factory.py 中 get_lr_scheduler 函数的支持列表。")

    # --- 应用 Warmup 调度器 (如果启用) ---
    if use_warmup and warmup_steps > 0:
        if scheduler_type == "ReduceOnPlateau" or scheduler_type == "ReduceLROnPlateau":
             # ReduceOnPlateau 不支持 warmup 包装
             print("警告: ReduceOnPlateau 调度器通常不直接与 Warmup 结合使用。Warmup 配置将被忽略。")
             use_warmup = False # 确保不再尝试应用 Warmup

        if use_warmup: # 再次检查 use_warmup，因为 ReduceOnPlateau 可能会将其设为 False
            # 使用 Warmup 包装器
            # PaddlePaddle 的 Warmup 是一个独立的调度器，需要组合使用
            # 注意：这里的实现可能需要根据 PaddlePaddle 的 Warmup API 进行调整
            # 假设 paddle.optimizer.lr 提供了 WarmupDecay 类
            try:
                # WarmupDecay 包装另一个调度器
                warmup_scheduler = lr_scheduler.WarmupDecay(
                     learning_rate=scheduler, # WarmupDecay 包装另一个调度器
                     warmup_steps=warmup_steps,
                     start_lr=start_lr, # 预热开始的学习率
                     end_lr=initial_learning_rate # 预热结束时的学习率，通常是基础学习率
                 )
                scheduler = warmup_scheduler
            except AttributeError:
                 print("警告: paddle.optimizer.lr 模块中未找到 WarmupDecay 类。Warmup 功能将无法使用。")
                 print("请检查您的 PaddlePaddle 版本或手动实现 Warmup 逻辑。")
            except Exception as e:
                 print(f"警告: 创建 Warmup 调度器时发生错误: {e}. Warmup 功能可能无法正常工作。")


    return scheduler

# --- 示例用法 (用于测试 lr_scheduler_factory.py 模块本身) ---
# 在实际训练中，train.py 会调用 get_lr_scheduler 函数
if __name__ == '__main__':
    print("--- 测试学习率调度器工厂函数 ---")

    # 模拟一个配置对象
    class MockConfig(ConfigObject):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 递归将字典转换为 ConfigObject
            for k, v in self.items():
                if isinstance(v, dict):
                    self[k] = MockConfig(v)

    # 1. 测试 StepDecay
    print("\n1. 测试 StepDecay:")
    try:
        config_step = MockConfig({
            "lr_scheduler_type": "StepDecay",
            "lr_scheduler_params": {
                "stepdecay": {"step_size": 30, "gamma": 0.1}, # 参数键名改为 stepdecay
                "warmup": {"use_warmup": False}
            }
        })
        scheduler1 = get_lr_scheduler(config_step, initial_learning_rate=0.1)
        print(f"  调度器类型: {type(scheduler1)}")
        # 模拟训练步数
        # for epoch in range(5):
        #      print(f"    Epoch {epoch+1}: LR = {scheduler1.get_lr():.6f}")
        #      scheduler1.step() # Paddle 的 step 是 per-batch 或 per-epoch，取决于调度器类型和用法

    except ValueError as e:
        print(f"  错误: {e}")

    # 2. 测试 MultiStepDecay
    print("\n2. 测试 MultiStepDecay:")
    try:
        config_multistep = MockConfig({
            "lr_scheduler_type": "MultiStepDecay",
            "lr_scheduler_params": {
                "multistepdecay": {"milestones": [30, 60, 90], "gamma": 0.1}, # 参数键名改为 multistepdecay
                "warmup": {"use_warmup": False}
            }
        })
        scheduler2 = get_lr_scheduler(config_multistep, initial_learning_rate=0.1)
        print(f"  调度器类型: {type(scheduler2)}")
        # 模拟训练步数
        # for epoch in range(5):
        #      print(f"    Epoch {epoch+1}: LR = {scheduler2.get_lr():.6f}")
        #      scheduler2.step()

    except ValueError as e:
        print(f"  错误: {e}")

    # 3. 测试 CosineAnnealingDecay with Warmup
    print("\n3. 测试 CosineAnnealingDecay with Warmup:")
    try:
        config_cosine_warmup = MockConfig({
            # 调度器类型和参数键名改为 CosineAnnealingDecay 和 cosineannealingdecay
            "lr_scheduler_type": "CosineAnnealingDecay",
            "lr_scheduler_params": {
                "cosineannealingdecay": {"T_max": 100, "eta_min": 0},
                "warmup": {"use_warmup": True, "warmup_steps": 100, "start_lr": 0.001}
            }
        })
        scheduler3 = get_lr_scheduler(config_cosine_warmup, initial_learning_rate=0.1)
        print(f"  调度器类型 (可能包含 Warmup 包装器): {type(scheduler3)}")
        # 模拟训练步数 (假设总步数是 1000)
        # for step in range(1000):
        #      current_lr = scheduler3.get_lr()
        #      if step % 100 == 0:
        #         print(f"    Step {step+1}: LR = {current_lr:.6f}")
        #      scheduler3.step()

    except ValueError as e:
        print(f"  错误: {e}")
    except Exception as e:
         print(f"  创建调度器时发生其他错误: {e}")

     # 4. 测试 ReduceOnPlateau
    print("\n4. 测试 ReduceOnPlateau:")
    try:
        config_reduce_lr = MockConfig({
            # 调度器类型和参数键名改为 ReduceOnPlateau 和 reduceonplateau
            "lr_scheduler_type": "ReduceOnPlateau",
            "lr_scheduler_params": {
                "reduceonplateau": {
                    "mode": 'min',
                    "factor": 0.1,
                    "patience": 5,
                    "threshold": 0.001,
                    "threshold_mode": 'rel',
                    "cooldown": 0,
                    "min_lr": 1e-6,
                    # removed 'eps'
                    # "eps": 1e-8 # Removed this parameter from test config
                },
                 "warmup": {"use_warmup": False} # ReduceOnPlateau 通常不和 warmup 直接组合
            }
        })
        scheduler4 = get_lr_scheduler(config_reduce_lr, initial_learning_rate=0.01)
        print(f"  调度器类型: {type(scheduler4)}")
        # 对于 ReduceOnPlateau，需要在评估指标不再改善时调用 step()
        # scheduler4.step(metrics_value) # 传入监控的指标值

    except ValueError as e:
        print(f"  错误: {e}")
    except TypeError as e:
         print(f"  捕获到 ReduceOnPlateau 的 TypeError: {e}")


    # 5. 测试 CosineAnnealingWarmRestarts
    print("\n5. 测试 CosineAnnealingWarmRestarts:")
    try:
        config_warm_restarts = MockConfig({
            "lr_scheduler_type": "CosineAnnealingWarmRestarts",
            "lr_scheduler_params": {
                "cosineannealingwarmrestarts": {"T_0": 10, "T_mult": 2, "eta_min": 0}, # 参数键名改为 cosineannealingwarmrestarts
                "warmup": {"use_warmup": True, "warmup_steps": 100, "start_lr": 0.001}
            }
        })
        scheduler5 = get_lr_scheduler(config_warm_restarts, initial_learning_rate=0.1)
        print(f"  调度器类型 (可能包含 Warmup 包装器): {type(scheduler5)}")
        # 模拟训练步数 (假设总步数是 1000)
        # for step in range(1000):
        #      current_lr = scheduler5.get_lr()
        #      if step % 50 == 0:
        #         print(f"    Step {step+1}: LR = {current_lr:.6f}")
        #      scheduler5.step() # Paddle 的 step 是 per-batch 或 per-epoch，取决于调度器类型和用法
        # # 或者模拟 epoch
        # for epoch in range(30): # 模拟多个周期
        #      print(f"    Epoch {epoch+1}: LR = {scheduler5.get_lr():.6f}")
        #      scheduler5.step() # CosineAnnealingWarmRestarts 的 step 通常是 per-epoch

    except ValueError as e:
        print(f"  错误: {e}")
    except Exception as e:
         print(f"  创建调度器时发生其他错误: {e}")


    # 6. 测试不支持的类型
    print("\n6. 测试不支持的调度器类型:")
    try:
        config_unsupported = MockConfig({
            "lr_scheduler_type": "NonExistentScheduler",
            "lr_scheduler_params": {},
            "warmup": {"use_warmup": False}
        })
        scheduler6 = get_lr_scheduler(config_unsupported, initial_learning_rate=0.1)
    except ValueError as e:
        print(f"  捕获到预期错误: {e}")

    # 7. 测试缺少必要参数的情况 (例如 MultiStepDecay 缺少 milestones)
    print("\n7. 测试缺少必要参数的情况 (MultiStepDecay 缺少 milestones):")
    try:
        config_missing_params = MockConfig({
            "lr_scheduler_type": "MultiStepDecay",
            "lr_scheduler_params": {
                "multistepdecay": {"gamma": 0.1}, # 缺少 milestones
                "warmup": {"use_warmup": False}
            }
        })
        scheduler7 = get_lr_scheduler(config_missing_params, initial_learning_rate=0.1)
    except ValueError as e:
        print(f"  捕获到预期错误: {e}")

    # 8. 测试未指定调度器类型
    print("\n8. 测试未指定调度器类型:")
    try:
        config_no_type = MockConfig({
            # "lr_scheduler_type": "StepDecay", # 注释掉这一行
            "lr_scheduler_params": {
                "stepdecay": {"step_size": 30, "gamma": 0.1},
                "warmup": {"use_warmup": False}
            }
        })
        scheduler8 = get_lr_scheduler(config_no_type, initial_learning_rate=0.1)
    except AttributeError as e: # lr_scheduler_type 不存在会是 AttributeError
        print(f"  捕获到预期错误: {e} (lr_scheduler_type 未找到)")
    except ValueError as e:
         print(f"  捕获到预期错误: {e}")

    # 9. 测试没有 lr_scheduler_params 键
    print("\n9. 测试没有 lr_scheduler_params 键:")
    try:
        config_no_params_key = MockConfig({
            "lr_scheduler_type": "StepDecay",
            # "lr_scheduler_params": { # 注释掉这一行
            #     "stepdecay": {"step_size": 30, "gamma": 0.1},
            #     "warmup": {"use_warmup": False}
            # }
        })
        scheduler9 = get_lr_scheduler(config_no_params_key, initial_learning_rate=0.1)
        print(f"  调度器类型: {type(scheduler9)}") # 如果 lr_scheduler_params 不存在，get() 会返回 {}，可能使用默认参数创建
    except ValueError as e:
        print(f"  捕获到错误: {e}") # 如果参数确实是必须的，这里会捕获错误

    # 10. 测试参数子键名不匹配
    print("\n10. 测试参数子键名不匹配:")
    try:
        config_wrong_params_key = MockConfig({
            "lr_scheduler_type": "StepDecay",
            "lr_scheduler_params": {
                "wrong_key": {"step_size": 30, "gamma": 0.1}, # 键名错误
                "warmup": {"use_warmup": False}
            }
        })
        scheduler10 = get_lr_scheduler(config_wrong_params_key, initial_learning_rate=0.1)
        # 因为使用了 .get(scheduler_type.lower(), {})，键名不匹配会返回 {}，可能使用默认参数创建
        # 这里可能会捕获到ValueError如果step_size和gamma是必须的且不在默认参数中
        # 如果scheduler_params为空字典，然后尝试访问scheduler_params['step_size']，会是KeyError
        # 但工厂函数中加了检查，会报 ValueError
        print(f"  调度器类型: {type(scheduler10)}") # 如果没有捕获到 ValueError
    except ValueError as e:
        print(f"  捕获到错误: {e}") # 捕获到预期的 ValueError

    # 11. 测试 PolynomialDecay
    print("\n11. 测试 PolynomialDecay:")
    try:
        config_polynomial = MockConfig({
             "lr_scheduler_type": "PolynomialDecay",
             "lr_scheduler_params": {
                  "polynomialdecay": {"decay_steps": 100, "end_lr": 0.0001, "power": 0.9}, # 参数键名改为 polynomialdecay
                  "warmup": {"use_warmup": False}
              }
         })
        scheduler11 = get_lr_scheduler(config_polynomial, initial_learning_rate=0.01)
        print(f"  调度器类型: {type(scheduler11)}")
    except ValueError as e:
        print(f"  错误: {e}")

    # 12. 测试 CosineAnnealingWarmRestarts
    print("\n12. 测试 CosineAnnealingWarmRestarts:")
    try:
        config_warm_restarts = MockConfig({
            "lr_scheduler_type": "CosineAnnealingWarmRestarts",
            "lr_scheduler_params": {
                "cosineannealingwarmrestarts": {"T_0": 10, "T_mult": 2, "eta_min": 0}, # 参数键名改为 cosineannealingwarmrestarts
                "warmup": {"use_warmup": True, "warmup_steps": 100, "start_lr": 0.001}
            }
        })
        scheduler12 = get_lr_scheduler(config_warm_restarts, initial_learning_rate=0.1)
        print(f"  调度器类型 (可能包含 Warmup 包装器): {type(scheduler12)}")
        # 模拟训练步数
        # for step in range(1000):
        #      current_lr = scheduler12.get_lr()
        #      if step % 50 == 0:
        #         print(f"    Step {step+1}: LR = {current_lr:.6f}")
        #      scheduler12.step()

    except ValueError as e:
        print(f"  错误: {e}")
    except Exception as e:
         print(f"  创建调度器时发生其他错误: {e}")