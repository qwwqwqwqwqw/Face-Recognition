# utils/lr_scheduler_factory.py
# 该工厂模块负责根据配置文件中的设置，动态创建和返回不同类型的学习率调度器实例。
# 目的是将学习率调度器的创建逻辑与主训练脚本分离，提高代码的模块化和可维护性。
#
# 支持的调度器类型包括：
# - StepLR: 按预设的步长衰减学习率。
# - MultiStepLR: 在预设的多个里程碑 (milestones) 处衰减学习率。
# - ExponentialLR: 按指数方式衰减学习率。
# - ReduceLROnPlateau: 当某个监控指标 (如验证集损失) 在一段时间内不再改善时，降低学习率。
# - CosineAnnealingLR: 使用余弦退火方式调整学习率。
# - PolynomialLR: 使用多项式衰减方式调整学习率。
#
# 工厂函数 `get_lr_scheduler` 会解析配置文件中的 `lr_scheduler_type` 和相应的 `lr_scheduler_params`，
# 然后实例化对应的调度器对象。它还支持可选的 `warmup_params` 来为调度器添加预热 (warmup) 阶段。
# 如果配置的调度器类型不受支持，或者参数不正确，会抛出 ValueError。
#
# 示例配置 (在YAML文件中):
# ```yaml
# # ...其他配置...
# learning_rate: 0.01 # 优化器的初始学习率，也会作为调度器的基础学习率
# lr_scheduler_type: "MultiStepLR" # 或者 "StepLR", "ReduceLROnPlateau", etc.
#
# lr_scheduler_params:
#   # MultiStepLR 特有的参数
#   multi_step_lr:
#     milestones: [30, 60, 90] # 在第30, 60, 90个epoch衰减
#     gamma: 0.1             # 衰减因子
#
#   # StepLR 特有的参数
#   step_lr:
#     step_size: 30
#     gamma: 0.1
#
#   # ReduceLROnPlateau 特有的参数
#   reduce_lr_on_plateau:
#     mode: "min"            # 监控指标越小越好 (如loss) 还是越大越好 (如acc)
#     factor: 0.1            # 学习率衰减因子
#     patience: 10           # 多少个epoch指标没有改善后降低学习率
#     threshold: 0.0001      # 衡量指标改善的阈值
#     verbose: True          # 是否打印学习率变化信息
#     metric_name: "loss"    # (自定义添加) 训练脚本中用于传递给step()的指标名称
#
#   # CosineAnnealingLR 特有的参数
#   cosine_annealing_lr:
#     T_max: 100             # 最大迭代次数 (通常是总epochs数)
#     eta_min: 0.0           # 最小学习率
#
#   # PolynomialLR 特有的参数
#   polynomial_lr:
#     decay_steps: 100000    # 总衰减步数 (通常是总batches数)
#     end_lr: 0.00001        # 最终学习率
#     power: 1.0             # 多项式幂次
#     cycle: False           # 是否循环
#
# warmup_params: # 可选的预热配置
#   use_warmup: True
#   warmup_epochs: 5
#   warmup_start_lr: 0.0001 # 预热阶段的起始学习率
# # ...
# ```

import paddle
from paddle.optimizer.lr import LRScheduler # 基础学习率调度器类
from paddle.optimizer.lr import StepDecay, MultiStepDecay, ExponentialDecay, ReduceOnPlateau, CosineAnnealingDecay, PolynomialDecay, LinearWarmup
from config_utils import ConfigObject # 假设ConfigObject是配置文件加载后生成的对象类型

def get_lr_scheduler(config: ConfigObject, initial_learning_rate: float) -> LRScheduler:
    """学习率调度器工厂函数。

    根据配置对象 `config` 中的 `lr_scheduler_type` 和 `lr_scheduler_params`，
    以及可选的 `warmup_params`，创建并返回一个 PaddlePaddle 学习率调度器实例。

    Args:
        config (ConfigObject): 包含学习率调度器配置的全局配置对象。
                               需要包含 `lr_scheduler_type` (str) 来指定调度器类型，
                               `lr_scheduler_params` (dict) 包含对应类型的参数，
                               以及可选的 `warmup_params` (dict) 用于预热配置。
        initial_learning_rate (float): 优化器设置的初始学习率，
                                       将作为调度器的基础学习率 (base_lr)。

    Returns:
        LRScheduler: 配置好的学习率调度器实例 (可能被LinearWarmup包装)。

    Raises:
        ValueError: 如果配置的 `lr_scheduler_type` 不支持，
                    或者对应类型的参数在 `lr_scheduler_params` 中缺失或无效。
    """
    scheduler_type = config.get('lr_scheduler_type')
    if not scheduler_type:
        # 如果未指定调度器类型，可以考虑返回一个 Constant LR 的调度器，
        # 或者要求用户必须指定。此处选择报错，因为通常期望明确配置。
        raise ValueError("错误: 学习率调度器类型 'lr_scheduler_type' 未在配置中指定。")

    all_scheduler_params = config.get('lr_scheduler_params', {}) # 获取所有调度器参数的字典
    if not isinstance(all_scheduler_params, ConfigObject) and not isinstance(all_scheduler_params, dict):
        raise ValueError(f"错误: 'lr_scheduler_params' 必须是一个字典或ConfigObject，但得到的是 {type(all_scheduler_params)}。")

    # 将ConfigObject转换为普通字典，如果它是ConfigObject的话，方便get操作和**解包
    # 因为ConfigObject的get方法可能与字典的get行为略有不同（例如，对于不存在的键返回None而不是抛KeyError）
    # 但PaddlePaddle的调度器构造函数通常期望普通字典的参数。    
    specific_params_key = ""
    lr_scheduler_instance = None

    print(f"准备创建学习率调度器: 类型 '{scheduler_type}', 初始学习率: {initial_learning_rate}")

    # 核心调度器实例化逻辑
    if scheduler_type.lower() == 'steplr' or scheduler_type.lower() == 'stepdecay':
        specific_params_key = 'step_lr' # 假设参数存储在 lr_scheduler_params.step_lr 下
        params = all_scheduler_params.get(specific_params_key, {})
        if not params or 'step_size' not in params:
            raise ValueError(f"错误: StepLR/StepDecay 调度器需要 '{specific_params_key}.step_size' 参数。")
        lr_scheduler_instance = StepDecay(
            learning_rate=initial_learning_rate,
            step_size=params['step_size'],
            gamma=params.get('gamma', 0.1), # 默认gamma为0.1
            verbose=params.get('verbose', False)
        )
        print(f"  StepLR/StepDecay 创建成功: step_size={params['step_size']}, gamma={params.get('gamma', 0.1)}")

    elif scheduler_type.lower() == 'multisteplr' or scheduler_type.lower() == 'multistepdecay':
        specific_params_key = 'multi_step_lr'
        params = all_scheduler_params.get(specific_params_key, {})
        if not params or 'milestones' not in params:
            raise ValueError(f"错误: MultiStepLR/MultiStepDecay 调度器需要 '{specific_params_key}.milestones' 参数。")
        if not isinstance(params['milestones'], list) or not all(isinstance(m, int) for m in params['milestones']):
            raise ValueError(f"错误: '{specific_params_key}.milestones' 必须是一个整数列表。")
        lr_scheduler_instance = MultiStepDecay(
            learning_rate=initial_learning_rate,
            milestones=params['milestones'],
            gamma=params.get('gamma', 0.1),
            verbose=params.get('verbose', False)
        )
        print(f"  MultiStepLR/MultiStepDecay 创建成功: milestones={params['milestones']}, gamma={params.get('gamma', 0.1)}")

    elif scheduler_type.lower() == 'exponentiallr' or scheduler_type.lower() == 'exponentialdecay':
        specific_params_key = 'exponential_lr'
        params = all_scheduler_params.get(specific_params_key, {})
        if not params or 'gamma' not in params:
            raise ValueError(f"错误: ExponentialLR/ExponentialDecay 调度器需要 '{specific_params_key}.gamma' 参数。")
        lr_scheduler_instance = ExponentialDecay(
            learning_rate=initial_learning_rate,
            gamma=params['gamma'],
            verbose=params.get('verbose', False)
        )
        print(f"  ExponentialLR/ExponentialDecay 创建成功: gamma={params['gamma']}")

    elif scheduler_type.lower() == 'reducelronplateau' or scheduler_type.lower() == 'reduceonplateau':
        specific_params_key = 'reduce_lr_on_plateau'
        params = all_scheduler_params.get(specific_params_key, {})
        # ReduceLROnPlateau 的 learning_rate 参数是优化器或另一个调度器，这里我们用 initial_learning_rate
        # 但实际上 ReduceOnPlateau 包装的是优化器，它的 learning_rate 属性由优化器管理。
        # PaddlePaddle 的 ReduceOnPlateau 通常在训练循环中手动调用 step(metric_value)。
        # 其构造函数中的 learning_rate 参数实际上是优化器对象！
        # 这与其他调度器不同，其他调度器接受float型的初始学习率。
        # 因此，工厂不应该直接实例化它，而是训练脚本在拥有优化器后，再用此参数构造它。
        # 然而，为了保持工厂的通用性，我们假设这里的 initial_learning_rate 就是给它的，
        # 并在训练脚本中正确使用（或者在训练脚本中直接创建此调度器）。
        # 此处我们遵循 PaddlePaddle LRScheduler 的模式，传入float型的 initial_learning_rate。
        # 训练脚本需要注意，如果是 ReduceOnPlateau，step() 需要传入监控的 metric。
        if not params: # mode, factor, patience 是常用参数
            print(f"警告: ReduceLROnPlateau ('{specific_params_key}') 参数未提供，将使用默认值。")
        lr_scheduler_instance = ReduceOnPlateau(
            learning_rate=initial_learning_rate, # 这是float，但ReduceOnPlateau的step由外部调用
            mode=params.get('mode', 'min'),
            factor=params.get('factor', 0.1),
            patience=params.get('patience', 10),
            threshold=params.get('threshold', 1e-4),
            threshold_mode=params.get('threshold_mode', 'rel'),
            cooldown=params.get('cooldown', 0),
            min_lr=params.get('min_lr', 0),
            epsilon=params.get('epsilon', 1e-8),
            verbose=params.get('verbose', False)
        )
        # ReduceOnPlateau 的 metric_name (自定义) 应在训练脚本中使用，而非此处
        print(f"  ReduceLROnPlateau 创建成功: mode={params.get('mode', 'min')}, factor={params.get('factor', 0.1)}, patience={params.get('patience', 10)}")

    elif scheduler_type.lower() == 'cosineannealinglr' or scheduler_type.lower() == 'cosineannealingdecay':
        specific_params_key = 'cosine_annealing_lr'
        params = all_scheduler_params.get(specific_params_key, {})
        if not params or 'T_max' not in params:
            raise ValueError(f"错误: CosineAnnealingLR/Decay 调度器需要 '{specific_params_key}.T_max' 参数。")
        lr_scheduler_instance = CosineAnnealingDecay(
            learning_rate=initial_learning_rate,
            T_max=params['T_max'],
            eta_min=params.get('eta_min', 0),
            verbose=params.get('verbose', False)
        )
        print(f"  CosineAnnealingLR/Decay 创建成功: T_max={params['T_max']}, eta_min={params.get('eta_min', 0)}")

    elif scheduler_type.lower() == 'polynomiallr' or scheduler_type.lower() == 'polynomialdecay':
        specific_params_key = 'polynomial_lr'
        params = all_scheduler_params.get(specific_params_key, {})
        if not params or 'decay_steps' not in params:
            raise ValueError(f"错误: PolynomialLR/Decay 调度器需要 '{specific_params_key}.decay_steps' 参数。")
        lr_scheduler_instance = PolynomialDecay(
            learning_rate=initial_learning_rate,
            decay_steps=params['decay_steps'],
            end_lr=params.get('end_lr', 0.00001), # 官方默认值
            power=params.get('power', 1.0),
            cycle=params.get('cycle', False),
            verbose=params.get('verbose', False)
        )
        print(f"  PolynomialLR/Decay 创建成功: decay_steps={params['decay_steps']}, end_lr={params.get('end_lr', 0.00001)}")
        
    else:
        raise ValueError(f"不支持的学习率调度器类型: '{scheduler_type}'.\n"
                         f"支持的类型包括: StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR, PolynomialLR (及其别名)。")

    # --- Warmup 逻辑 --- 
    # 如果配置了预热 (warmup)，则用 LinearWarmup 包装上面创建的调度器实例。
    warmup_cfg = config.get('warmup_params', {})
    use_warmup = warmup_cfg.get('use_warmup', False)

    if use_warmup and lr_scheduler_instance:
        warmup_epochs = warmup_cfg.get('warmup_epochs')
        warmup_start_lr = warmup_cfg.get('warmup_start_lr')
        
        if warmup_epochs is None or warmup_start_lr is None:
            raise ValueError("错误: 启用了预热 (use_warmup=True)，但缺少 'warmup_epochs' 或 'warmup_start_lr' 参数。")
        if not isinstance(warmup_epochs, int) or warmup_epochs <= 0:
            raise ValueError("错误: 'warmup_epochs' 必须是正整数。")
        if not isinstance(warmup_start_lr, (float, int)) or warmup_start_lr < 0:
            raise ValueError("错误: 'warmup_start_lr' 必须是非负数。")
            
        # 确保 warmup_start_lr 不大于等于 initial_learning_rate (预热应该是从小到大)
        if warmup_start_lr >= initial_learning_rate:
            print(f"警告: 预热起始学习率 warmup_start_lr ({warmup_start_lr}) 大于或等于优化器初始学习率 ({initial_learning_rate})。"
                  f"     这可能不是预期的预热行为。预热将被应用，但效果可能不明显。")

        # LinearWarmup 构造函数的第一个参数是 LRScheduler 实例，第二个是预热的步数(epochs)，第三个是起始LR，第四个是结束LR(即initial_learning_rate)
        lr_scheduler_instance = LinearWarmup(
            learning_rate=lr_scheduler_instance, # 被包装的调度器
            warmup_steps=warmup_epochs,          # 预热的epoch数
            start_lr=warmup_start_lr,            # 预热起始学习率
            end_lr=initial_learning_rate,        # 预热结束学习率 (即主调度器的初始学习率)
            verbose=warmup_cfg.get('verbose', False) # LinearWarmup也有verbose参数
        )
        print(f"  学习率调度器已应用 LinearWarmup: epochs={warmup_epochs}, start_lr={warmup_start_lr}, end_lr={initial_learning_rate}")
    elif use_warmup and lr_scheduler_instance is None:
        # 这理论上不应该发生，因为如果lr_scheduler_instance是None，前面应该已经抛出异常了
        print("警告: 配置了预热，但基础学习率调度器未能创建。预热将不会被应用。")

    if lr_scheduler_instance is None:
        # 再次检查，确保最终返回的是一个有效的调度器实例
        raise RuntimeError(f"最终学习率调度器未能成功创建。类型: '{scheduler_type}'。请检查配置和工厂逻辑。")

    return lr_scheduler_instance

# --- 示例用法 (用于测试或演示，实际使用时由训练脚本调用) ---
if __name__ == '__main__':
    print("--- 学习率调度器工厂模块演示 ---")

    # 模拟一个包含调度器配置的 ConfigObject
    # 注意：ConfigObject 本身需要定义，或者直接使用字典来模拟
    # 为了简单，这里直接使用字典，并假设 get 方法行为类似
    class MockConfig:
        def __init__(self, data):
            self._data = data
        def get(self, key, default=None):
            # 简化版get，如果键是嵌套的 (如 'lr_scheduler_params.step_lr')，则不支持
            # 真实ConfigObject应能处理嵌套访问
            if '.' in key:
                # 模拟嵌套获取，实际ConfigObject会有更健壮的实现
                try:
                    current = self._data
                    for part in key.split('.'):
                        current = current[part]
                    return current
                except KeyError:
                    return default
            return self._data.get(key, default)

    # 1. 测试 StepLR
    print("\n1. 测试 StepLR:")
    try:
        config_steplr = MockConfig({
            "lr_scheduler_type": "StepLR",
            "lr_scheduler_params": {
                "step_lr": {"step_size": 10, "gamma": 0.5, "verbose": True}
            },
            "warmup_params": {"use_warmup": False} # 不使用预热
        })
        scheduler1 = get_lr_scheduler(config_steplr, initial_learning_rate=0.1)
        print(f"  创建的调度器类型: {type(scheduler1)}")
        # 模拟训练过程中的学习率变化
        print("  模拟LR变化 (前35个epochs):")
        current_lr = 0.1
        for epoch in range(35):
            if isinstance(scheduler1, ReduceOnPlateau):
                scheduler1.step(0.1) # 假设metric是0.1，不改变
            else:
                scheduler1.step() # 对大多数调度器，step()不带参数或带epoch索引
            
            # 获取更新后的学习率
            if hasattr(scheduler1, 'last_lr'): # LinearWarmup等包装器有last_lr
                new_lr = scheduler1.last_lr
            elif hasattr(scheduler1, 'get_lr'): # 有些基础调度器有get_lr()
                new_lr = scheduler1.get_lr()
            else: # 对于ReduceOnPlateau等，LR直接通过优化器获取，这里模拟
                # 这是一个简化，实际LR应从优化器获取
                if isinstance(scheduler1, ReduceOnPlateau) and hasattr(scheduler1, 'learning_rate'): 
                    new_lr = scheduler1.learning_rate # ReduceOnPlateau.learning_rate 是float
                else: 
                    new_lr = current_lr # 保持不变如果无法获取
            
            if abs(new_lr - current_lr) > 1e-7: # 如果学习率有显著变化
                print(f"    Epoch {epoch+1}: LR = {new_lr:.6f}")
                current_lr = new_lr
            elif epoch == 0 : # 仅在第一个epoch打印，如果LR未变
                print(f"    Epoch {epoch+1}: LR = {new_lr:.6f}")

    except ValueError as e:
        print(f"  错误: {e}")

    # 2. 测试 MultiStepLR with Warmup
    print("\n2. 测试 MultiStepLR with Warmup:")
    try:
        config_multisteplr_warmup = MockConfig({
            "lr_scheduler_type": "MultiStepLR",
            "lr_scheduler_params": {
                "multi_step_lr": {"milestones": [15, 25], "gamma": 0.1, "verbose": True}
            },
            "warmup_params": {
                "use_warmup": True,
                "warmup_epochs": 5,
                "warmup_start_lr": 0.001,
                "verbose": True
            }
        })
        scheduler2 = get_lr_scheduler(config_multisteplr_warmup, initial_learning_rate=0.01)
        print(f"  创建的调度器类型: {type(scheduler2)}, 内部调度器: {type(scheduler2.learning_rate) if hasattr(scheduler2, 'learning_rate') else 'N/A'}")
        current_lr = 0.01 # 理论上应从warmup_start_lr开始，但打印逻辑依赖scheduler
        for epoch in range(30):
            # 对于被LinearWarmup包装的调度器，其step()方法会自动处理预热和主调度器的step
            scheduler2.step() 
            new_lr = scheduler2.last_lr # LinearWarmup有last_lr
            if abs(new_lr - current_lr) > 1e-7 or epoch < config_multisteplr_warmup.get('warmup_params').get('warmup_epochs') or epoch == 0:
                 print(f"    Epoch {epoch+1}: LR = {new_lr:.6f}")
                 current_lr = new_lr
    except ValueError as e:
        print(f"  错误: {e}")

    # 3. 测试 ReduceLROnPlateau (注意：其step行为不同)
    print("\n3. 测试 ReduceLROnPlateau:")
    try:
        config_reduce_lr = MockConfig({
            "lr_scheduler_type": "ReduceLROnPlateau",
            "lr_scheduler_params": {
                "reduce_lr_on_plateau": {
                    "mode": "min", 
                    "factor": 0.5, 
                    "patience": 3, 
                    "verbose": True,
                    "threshold": 0.01
                }
            },
            "warmup_params": {"use_warmup": False}
        })
        # ReduceLROnPlateau 的 learning_rate 参数是 float (初始LR)
        # 它的 step 方法需要一个 metric 值。
        scheduler3 = get_lr_scheduler(config_reduce_lr, initial_learning_rate=0.1)
        print(f"  创建的调度器类型: {type(scheduler3)}")
        mock_optimizer_lr = 0.1 # 模拟优化器的学习率
        print(f"    Epoch 1: LR = {mock_optimizer_lr:.6f}, Metric = 1.0 (step)")
        scheduler3.step(1.0) # metric_value = 1.0
        # ReduceLROnPlateau 不直接修改自身的LR，而是依赖优化器更新。
        # 它的verbose输出会指示LR是否应该改变。
        # 我们需要模拟优化器实际的LR变化来观察效果。
        # 假设如果scheduler3.verbose打印了降低LR的信息，我们手动调整mock_optimizer_lr
        # PaddlePaddle的ReduceLROnPlateau的_reduce_lr方法会返回新的LR，但step不直接返回。
        # 这里仅演示创建，实际的LR变化由训练循环中的优化器体现。
        for i in range(2, 8):
            metric = 1.0 - (i * 0.001) # 模拟指标轻微改善，但不足以超过阈值
            print(f"    Epoch {i}: LR = {mock_optimizer_lr:.6f} (before step), Metric = {metric:.3f}")
            scheduler3.step(metric) 
            # 实际应用中，如果scheduler的step导致了LR变化，会通过优化器反映出来
            # 这里仅作演示，不模拟优化器的交互
        print(f"    Epoch 8: LR = {mock_optimizer_lr:.6f} (before step), Metric = 0.5 (significant improvement)")
        scheduler3.step(0.5)
        # 此时verbose应有输出，提示学习率降低，mock_optimizer_lr应相应更新 (例如变为 0.1 * 0.5 = 0.05)
        # mock_optimizer_lr = 0.05 # 手动模拟
        print(f"    (如果发生衰减，模拟的LR会降低，此处仅演示调度器创建和调用)")

    except ValueError as e:
        print(f"  错误: {e}")

    # 4. 测试不支持的类型
    print("\n4. 测试不支持的调度器类型:")
    try:
        config_unsupported = MockConfig({
            "lr_scheduler_type": "NonExistentScheduler",
            "lr_scheduler_params": {},
            "warmup_params": {"use_warmup": False}
        })
        scheduler4 = get_lr_scheduler(config_unsupported, initial_learning_rate=0.1)
    except ValueError as e:
        print(f"  捕获到预期错误: {e}")
        
    # 5. 测试缺少必要参数的情况 (例如 MultiStepLR 缺少 milestones)
    print("\n5. 测试 MultiStepLR 缺少 milestones 参数:")
    try:
        config_missing_param = MockConfig({
            "lr_scheduler_type": "MultiStepLR",
            "lr_scheduler_params": {
                "multi_step_lr": {"gamma": 0.1} # 故意缺少 milestones
            },
            "warmup_params": {"use_warmup": False}
        })
        scheduler5 = get_lr_scheduler(config_missing_param, initial_learning_rate=0.01)
    except ValueError as e:
        print(f"  捕获到预期错误: {e}") 