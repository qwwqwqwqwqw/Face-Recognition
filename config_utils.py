# config_utils.py
# 该模块负责加载、合并和管理项目的配置信息。
# 主要功能包括：
# 1. 从YAML文件加载配置。
# 2. 支持全局设置和活动配置块的层级结构。
# 3. 将命令行参数与YAML配置进行合并，命令行参数具有最高优先级。
# 4. 提供一个ConfigObject类，使得配置项可以通过属性点号访问。

import yaml
import argparse
import os
import collections.abc

# 使用自定义类以支持点属性访问，同时仍可作为字典使用
class ConfigObject(dict):
    """
    配置对象类，继承自dict，提供通过属性访问配置项的能力。

    该类旨在将嵌套的字典转换为可以通过点操作符 (e.g., `config.train.batch_size`)
    进行访问的对象结构，同时保留了字典原有的方法。
    当访问一个值为字典的属性时，该字典也会被自动转换为ConfigObject实例，
    从而实现深层次的属性访问。
    """
    def __getattr__(self, name):
        """
        当尝试通过属性点号访问配置项时调用。

        如果 `name` 是字典中的一个键，则返回其对应的值。
        如果该值本身是一个字典但尚未转换为 `ConfigObject`，则会先进行转换。
        如果 `name` 不是一个有效的键，则抛出 `AttributeError`。

        Args:
            name (str): 尝试访问的属性名。

        Returns:
            any: 对应属性的值。如果原值为字典，则返回 `ConfigObject` 实例。

        Raises:
            AttributeError: 如果配置中不存在名为 `name` 的属性。
        """
        if name in self:
            value = self[name]
            # 动态转换嵌套字典为ConfigObject
            if isinstance(value, dict) and not isinstance(value, ConfigObject):
                value = ConfigObject(value)
                self[name] = value # 更新，以便下次直接访问
            return value
        else:
            # 对于不存在的属性，明确抛出AttributeError，行为与普通对象一致
            raise AttributeError(f"配置中不存在属性: '{name}'")

    def __setattr__(self, name, value):
        """
        当尝试通过属性点号设置配置项时调用。

        Args:
            name (str): 要设置的属性名。
            value (any): 要设置的属性值。
        """
        self[name] = value

    def __delattr__(self, name):
        """
        当尝试通过属性点号删除配置项时调用。

        Args:
            name (str): 要删除的属性名。

        Raises:
            AttributeError: 如果配置中不存在名为 `name` 的属性。
        """
        if name in self:
            del self[name]
        else:
            raise AttributeError(f"配置中不存在属性: '{name}'")

    def get(self, key, default=None):
        """
        获取配置项，如果键不存在则返回指定的默认值。

        如果获取到的值是字典，则会确保其作为 `ConfigObject` 返回。

        Args:
            key (str): 要获取的配置项的键。
            default (any, optional): 如果键不存在时返回的默认值。默认为 `None`。

        Returns:
            any: 对应键的值，或默认值。如果原值为字典，则返回 `ConfigObject`。
        """
        value = super().get(key, default)
        if isinstance(value, dict) and not isinstance(value, ConfigObject):
            return ConfigObject(value) # 确保深层get也返回ConfigObject
        return value

    def to_dict(self):
        """
        将ConfigObject实例及其所有嵌套的ConfigObject转换为普通的Python字典。

        这在需要将配置传递给不支持自定义对象序列化或需要标准字典格式的
        外部库或函数时非常有用。

        Returns:
            dict: 包含所有配置项的标准Python字典。
        """
        regular_dict = {}
        for key, value in self.items():
            if isinstance(value, ConfigObject):
                regular_dict[key] = value.to_dict() # 递归转换
            elif isinstance(value, dict): # 处理普通字典（理论上应该在访问时已转为ConfigObject）
                inner_dict = {}
                for k_inner, v_inner in value.items():
                    if isinstance(v_inner, ConfigObject):
                        inner_dict[k_inner] = v_inner.to_dict()
                    else:
                        inner_dict[k_inner] = v_inner
                regular_dict[key] = inner_dict
            else:
                regular_dict[key] = value
        return regular_dict

# 辅助函数：深层合并字典 (to_merge 会覆盖 base 中的值)
def deep_update(base: dict, to_merge: dict) -> dict:
    """
    递归地更新（合并）字典。

    将 `to_merge` 字典中的键值对合并到 `base` 字典中。
    如果某个键在 `base` 和 `to_merge` 中都存在，并且其值都是字典，
    则会递归地对这两个内层字典进行合并。
    否则，`to_merge` 中的值会直接覆盖 `base` 中的值。

    Args:
        base (dict): 基础字典，将被更新。
        to_merge (dict): 包含要合并的键值对的字典。

    Returns:
        dict: 更新后的 `base` 字典。
    """
    for key, value in to_merge.items():
        if isinstance(value, collections.abc.Mapping) and key in base and isinstance(base[key], collections.abc.Mapping):
            # 如果键存在且对应的值都是字典，则递归更新
            base[key] = deep_update(base.get(key, {}), value)
        else:
            # 否则，直接用 to_merge 中的值覆盖 base 中的值
            base[key] = value
    return base

def load_config(default_yaml_path: str, cmd_args_namespace: argparse.Namespace) -> ConfigObject:
    """
    加载项目配置，遵循特定的优先级顺序：
    1. 从YAML文件加载基础配置：
        a. 加载 `global_settings`（如果存在）。
        b. 根据YAML中的 `active_config` 键，加载并深层合并对应的配置块到全局设置中。
    2. 将命令行解析的参数 (`cmd_args_namespace`) 合并到从YAML加载的配置中。
       命令行参数具有最高优先级，会覆盖YAML文件中的同名配置项。

    Args:
        default_yaml_path (str): 当命令行未通过 `--config_path` 指定配置文件时，
                                 使用的默认YAML配置文件路径。
        cmd_args_namespace (argparse.Namespace): 由 `argparse` 解析后的命令行参数对象。
                                                期望包含一个可选的 `config_path` 属性，
                                                以及其他可能覆盖YAML配置的参数。

    Returns:
        ConfigObject: 一个包含最终合并后所有配置项的 `ConfigObject` 实例。
                      可以通过点属性方式访问配置内容。
    """
    # 确定最终要加载的YAML文件路径
    # 优先使用命令行 --config_path 指定的路径，否则使用函数传入的 default_yaml_path
    yaml_file_to_load = getattr(cmd_args_namespace, 'config_path', None) or default_yaml_path

    full_yaml_config = {} # 用于存储从YAML文件完整加载的内容
    if yaml_file_to_load and os.path.exists(yaml_file_to_load):
        print(f"从 YAML 文件加载配置: {yaml_file_to_load}")
        try:
            with open(yaml_file_to_load, 'r', encoding='utf-8') as f:
                loaded_yaml = yaml.safe_load(f)
                if loaded_yaml:
                    full_yaml_config = loaded_yaml
                else:
                    print(f"警告: YAML 配置文件 {yaml_file_to_load} 为空或无效。")
        except yaml.YAMLError as e:
            print(f"错误: 解析 YAML 文件 {yaml_file_to_load} 失败: {e}")
        except Exception as e:
            print(f"错误: 读取 YAML 文件 {yaml_file_to_load} 时发生未知错误: {e}")
    elif yaml_file_to_load: # 指定了路径但文件不存在
        print(f"警告: 指定的 YAML 配置文件 {yaml_file_to_load} 未找到。")
    else: # 没有指定任何YAML文件路径
        print("未提供 YAML 配置文件路径，将仅使用命令行参数和其在脚本中定义的默认值（如果存在）。")

    # 步骤 1: 获取全局设置 (global_settings)
    # global_settings 是YAML文件中的一个顶级键，定义了通用的基础配置。
    global_settings = full_yaml_config.get('global_settings', {})
    if not global_settings and 'use_gpu' in full_yaml_config: # 兼容旧的扁平化YAML结构
        print("提示: 未找到 'global_settings' 块，将尝试从YAML根级别读取全局参数。")
        # 简单启发式：将所有非 '_config' 结尾且非 'active_config' 的顶级键视为全局设置
        potential_global_keys = [k for k in full_yaml_config.keys() if not k.endswith('_config') and k != 'active_config']
        global_settings = {k: full_yaml_config[k] for k in potential_global_keys if k in full_yaml_config}
        if not global_settings:
             print("警告: YAML文件中既没有 'global_settings' 块，也未能从根级别推断出全局参数。全局配置为空。")

    # 初始化合并后的配置字典，从全局设置开始
    merged_config_dict = global_settings.copy()

    # 步骤 2: 确定并获取活动配置块 (active_config)
    # 优先使用命令行传入的 active_config，如果存在且有效的话。
    cmd_active_config = getattr(cmd_args_namespace, 'active_config', None)
    yaml_active_config_name = full_yaml_config.get('active_config')

    active_config_name_to_use = None
    if cmd_active_config: # 如果命令行提供了 active_config
        print(f"从命令行接收到 active_config: '{cmd_active_config}'，将覆盖YAML中的设置。")
        active_config_name_to_use = cmd_active_config
    elif yaml_active_config_name: # 否则，使用YAML文件中的 active_config
        active_config_name_to_use = yaml_active_config_name
        print(f"将使用YAML文件中指定的 active_config: '{yaml_active_config_name}'")
    else: # 两者都没有提供
        print("提示: YAML文件和命令行均未定义 'active_config'。将仅使用全局设置（如果存在）和命令行参数。")

    if active_config_name_to_use:
        active_config_block = full_yaml_config.get(active_config_name_to_use)
        if active_config_block and isinstance(active_config_block, dict):
            print(f"加载并合并活动配置块: {active_config_name_to_use}")
            merged_config_dict = deep_update(merged_config_dict, active_config_block)
        elif active_config_block: # 不是字典
             print(f"警告: 活动配置块 '{active_config_name_to_use}' 的值不是一个有效的配置字典，已忽略。")
        else: # 找不到对应的配置块
            print(f"警告: 在YAML中找不到名为 '{active_config_name_to_use}' 的配置块。")
    
    # 步骤 3: 合并命令行参数 (命令行参数具有最高优先级)
    # 将argparse解析的命名空间转换为字典
    cmd_args_dict = vars(cmd_args_namespace)
    # 创建最终配置的副本，用于合并命令行参数
    final_merged_config = merged_config_dict.copy()

    for key, cmd_value in cmd_args_dict.items():
        # `config_path` 参数本身不作为业务配置项合并到最终结果中，它仅用于指定YAML文件路径
        if key == 'config_path':
            # 理论上，此处的 config_path 应该已经被用于加载YAML，
            # 但为了完整性，可以将其保留在最终配置中以供追溯，或者选择不添加。
            # final_merged_config[key] = cmd_value
            continue

        # 核心逻辑：只有当命令行参数提供了非None值时，才用它来覆盖现有配置。
        # 另一种策略是：即使用户在命令行提供了None，也用它覆盖（除非该参数在argparse中定义了从不为None的action）。
        # 当前实现：如果命令行参数为None，则不覆盖YAML中可能存在的同名有效值。
        #          但如果该键之前完全不存在，则即使命令行值为None，也会添加它。
        if cmd_value is not None:
            # 对于嵌套参数（例如 optimizer_params.weight_decay），如果命令行设计为扁平化参数
            # (例如 --weight_decay)，则这里会直接在顶层设置 `weight_decay`。
            # 如果期望命令行也能深层指定并合并，例如 --optimizer_params '{"weight_decay": 0.01}'，
            # 则需要在此处添加更复杂的解析和深层合并逻辑。
            # 当前假设命令行参数是扁平的，直接覆盖顶层同名键。
            final_merged_config[key] = cmd_value
        elif key not in final_merged_config: # 如果键不存在于之前的配置中，且命令行值为None，也添加
            final_merged_config[key] = cmd_value

    # 将最终合并的字典转换为ConfigObject实例，以便进行属性访问
    final_config_obj = ConfigObject(final_merged_config)
    
    # 打印最终生效的配置项，方便调试和确认
    print("--- 生效的配置项 (YAML与命令行合并后) ---")
    def _print_config_recursively(item: dict, indent_level: int = 0):
        """内部辅助函数，递归打印配置项，保持缩进美观。"""
        indent_str = "  " * indent_level
        # 对键进行排序，确保打印顺序一致，便于比较不同运行间的配置差异
        for k, v in sorted(item.items()):
            if isinstance(v, ConfigObject) or isinstance(v, dict): # ConfigObject也是dict的子类
                print(f"{indent_str}{k}:")
                _print_config_recursively(v, indent_level + 1)
            else:
                print(f"{indent_str}{k}: {v}")
    
    _print_config_recursively(final_config_obj, indent_level=1)
    print("-----------------------------------------")

    return final_config_obj
