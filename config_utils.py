import yaml
import argparse
import os
import collections.abc

# 使用自定义类以支持点属性访问，同时仍可作为字典使用
class ConfigObject(dict):
    """配置对象类，支持通过属性访问配置项。"""
    def __getattr__(self, name):
        if name in self:
            value = self[name]
            if isinstance(value, dict) and not isinstance(value, ConfigObject):
                # 如果值是字典但不是ConfigObject，将其转换为ConfigObject
                # 这样可以实现深层属性访问，例如 config.train.batch_size
                value = ConfigObject(value)
                self[name] = value # 更新回ConfigObject以备后用
            return value
        else:
            # 对于不存在的属性可以返回None或抛出更明确的错误
            # print(f"警告: 尝试访问不存在的配置属性 '{name}'")
            # return None 
            raise AttributeError(f"配置中不存在属性: '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError(f"配置中不存在属性: '{name}'")

    def get(self, key, default=None):
        """获取配置项，如果不存在则返回默认值。"""
        value = super().get(key, default)
        if isinstance(value, dict) and not isinstance(value, ConfigObject):
            return ConfigObject(value)
        return value

    def to_dict(self):
        """Преобразует ConfigObject и все вложенные ConfigObject в обычные словари."""
        regular_dict = {}
        for key, value in self.items():
            if isinstance(value, ConfigObject):
                regular_dict[key] = value.to_dict()
            elif isinstance(value, dict):
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
def deep_update(base, to_merge):
    """递归更新字典。"""
    for key, value in to_merge.items():
        if isinstance(value, collections.abc.Mapping):
            base[key] = deep_update(base.get(key, {}), value)
        else:
            base[key] = value
    return base

def load_config(default_yaml_path, cmd_args_namespace):
    """
    加载配置。顺序：全局YAML -> 活动配置YAML -> 命令行参数。
    后加载的会覆盖先加载的同名参数。
    """
    yaml_file_to_load = getattr(cmd_args_namespace, 'config_path', None) or default_yaml_path

    full_yaml_config = {}
    if yaml_file_to_load and os.path.exists(yaml_file_to_load):
        print(f"从 YAML 文件加载配置: {yaml_file_to_load}")
        with open(yaml_file_to_load, 'r', encoding='utf-8') as f:
            loaded_yaml = yaml.safe_load(f)
            if loaded_yaml:
                full_yaml_config = loaded_yaml
            else:
                print(f"警告: YAML 配置文件 {yaml_file_to_load} 为空或无效。")
    elif yaml_file_to_load:
        print(f"警告: 指定的 YAML 配置文件 {yaml_file_to_load} 未找到。")
    else:
        print("未提供 YAML 配置文件路径，将仅使用命令行参数和其在脚本中定义的默认值。")

    # 1. 获取全局设置
    global_settings = full_yaml_config.get('global_settings', {})
    if not global_settings and 'use_gpu' in full_yaml_config: # 兼容旧的扁平化结构或直接使用根级别参数
        print("提示: 未找到 'global_settings' 块，将尝试从YAML根级别读取全局参数。")
        # 提取所有非配置块的键作为全局设置 (简单启发式方法)
        potential_global_keys = [k for k in full_yaml_config.keys() if not k.endswith('_config') and k != 'active_config']
        global_settings = {k: full_yaml_config[k] for k in potential_global_keys if k in full_yaml_config}
        if not global_settings:
             print("警告: YAML文件中既没有 'global_settings' 块，也未能从根级别推断出全局参数。全局配置为空。")

    merged_config_dict = global_settings.copy() # 从全局设置开始

    # 2. 获取并合并活动配置块
    active_config_name = full_yaml_config.get('active_config')
    if active_config_name:
        active_config_block = full_yaml_config.get(active_config_name)
        if active_config_block:
            print(f"加载活动配置块: {active_config_name}")
            # merged_config_dict.update(active_config_block) # 浅合并
            merged_config_dict = deep_update(merged_config_dict, active_config_block) # 深合并
        else:
            print(f"警告: 在YAML中找到了 active_config 名称 '{active_config_name}'，但找不到对应的配置块。")
    else:
        print("警告: YAML 文件中未定义 'active_config'。将仅使用全局设置（如果存在）和命令行参数。")

    # 3. 合并命令行参数 (命令行参数具有最高优先级)
    cmd_args_dict = vars(cmd_args_namespace)
    final_merged_config = merged_config_dict.copy() # 创建副本以进行命令行参数的合并

    for key, cmd_value in cmd_args_dict.items():
        if key == 'config_path': # config_path 本身不应视为业务配置项被深层合并
            final_merged_config[key] = cmd_value
            continue

        # 只有当命令行参数不是None时，才用它来覆盖
        # 或者，如果该键不存在于之前的合并结果中，也添加它（即使是None）
        if cmd_value is not None:
            # 对于optimizer_params, lr_scheduler_params, model.resnet_params, loss.arcface_params 等嵌套字典，
            # 我们可能希望智能合并而不是完全替换。但简单起见，这里仍是直接替换。
            # 如果需要对特定嵌套字典进行深层合并，则需要更复杂的逻辑。
            # 当前的 deep_update 主要用于 YAML 内部的 global 和 active_config 的合并。
            # 命令行参数通常是扁平的，直接覆盖。
            # 如果命令行参数本身代表一个字典结构（例如通过json字符串传入并解析），则需要特殊处理。
            # 假设这里的命令行参数都是简单类型或 None
            final_merged_config[key] = cmd_value
        elif key not in final_merged_config: # 如果key不存在于之前的配置中，则添加（即使值为None）
            final_merged_config[key] = cmd_value

    final_config_obj = ConfigObject(final_merged_config)
    
    print("--- 生效的配置项 ---")
    # 为了更好地展示嵌套结构，可以自定义打印函数
    def print_config_item(item, indent=0):
        for k, v in sorted(item.items()):
            if isinstance(v, ConfigObject) or isinstance(v, dict):
                print(f"{'  ' * indent}{k}:")
                print_config_item(v, indent + 1)
            else:
                print(f"{'  ' * indent}{k}: {v}")
    
    print_config_item(final_config_obj, indent=1)
    print("--------------------")

    return final_config_obj
