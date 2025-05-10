# coding=utf-8
"""
CreateDataList.py

本模块包含 CreateDataList 类，专门用于扫描指定的数据集根目录，
并根据其中的图像文件自动生成用于模型训练和测试的数据列表文件（通常是 .list 格式）
以及一个元数据文件（通常是 readme.json）。

数据列表文件的格式通常是：`图像文件相对路径\\t类别标签ID`。
元数据文件 (readme.json) 则包含了数据集的整体信息，如总类别数、总图像数、
以及每个类别的详细统计（类别名、标签ID、图像数量等）。

该脚本对于准备和管理机器学习（尤其是图像分类、人脸识别等任务）所需的数据集非常有用。
它规范了数据列表的生成过程，确保了标签分配的一致性，并提供了数据集的概览信息。
"""
import os
import json

class CreateDataList:
    """
    用于创建图像数据列表和元数据文件的核心类。

    该类通过遍历用户指定的数据集根目录，识别其中的子文件夹作为不同的类别，
    然后为每个类别下的图像文件生成相应的条目，并将其分配到训练列表或测试列表中。
    同时，它还会收集关于数据集的统计信息，并将其保存为一个JSON格式的元数据文件。
    """
    def __init__(self):
        """
        初始化CreateDataList类。

        目前该构造函数不执行特殊操作，仅作为标准类结构的一部分。
        未来可以根据需要扩展，例如接收一些默认配置参数。
        """
        pass # 目前不需要特殊初始化操作

    def create_data_list(self, data_root_path: str, 
                         train_list_name: str = "trainer.list", 
                         test_list_name: str = "test.list", 
                         meta_file_name: str = "readme.json") -> None:
        """
        遍历指定的数据集根目录，为其中的图像生成训练列表、测试列表和元数据JSON文件。

        主要流程：
        1. 清理（通过覆盖写入模式 'w'）或创建指定名称的训练、测试列表文件和元数据文件。
        2. 扫描 `data_root_path` 下的第一级子目录，每个子目录被视为一个独立的类别。
        3. 对识别出的类别进行排序，以确保每次运行时生成的类别标签ID保持一致。
        4. 遍历每个类别目录中的图像文件（支持常见图像格式如 .jpg, .png 等）。
        5. 对每个类别内的图像文件也进行排序，以保证处理顺序的一致性。
        6. 按照预设的简单划分策略（例如，每N张图片取一张作为测试样本），
           将图像条目（`图像相对路径\\t类别标签ID`）写入训练列表或测试列表。
        7. 收集每个类别的统计信息（名称、标签ID、图像总数、训练/测试图像数）。
        8. 将整个数据集的元数据（包括所有类别的详细信息）写入JSON文件。

        Args:
            data_root_path (str): 数据集的根目录路径。
                                  此目录下应包含对应各类别的子文件夹，
                                  每个子文件夹中存放该类别的图像。
                                  例如: `data_root_path = 'data/face'`
                                        `'data/face/person1/img1.jpg'`
                                        `'data/face/person2/img2.jpg'`
            train_list_name (str, optional): 生成的训练数据列表文件名。
                                             默认为 "trainer.list"。
            test_list_name (str, optional): 生成的测试数据列表文件名。
                                            默认为 "test.list"。
            meta_file_name (str, optional): 生成的元数据JSON文件名。
                                            默认为 "readme.json"。

        Returns:
            None: 该方法直接将生成的文件写入磁盘，不返回任何值。

        Raises:
            FileNotFoundError: 如果 `data_root_path` 不存在。
            IOError: 如果在写入列表文件或元数据文件时发生IO错误。
            TypeError: 如果在序列化元数据到JSON时发生类型错误。
        """
        print(f"开始为数据集 '{data_root_path}' 创建数据列表...")
        
        # --- 文件路径准备 ---
        # 训练列表、测试列表和元数据文件的完整路径。
        # 这些文件将直接保存在 data_root_path 目录下。
        train_file_path = os.path.join(data_root_path, train_list_name)
        test_file_path = os.path.join(data_root_path, test_list_name)
        meta_file_path = os.path.join(data_root_path, meta_file_name)

        # --- 清理旧的列表文件 --- 
        # 最佳实践提示：由于后续使用 'w' (写入) 模式打开文件，旧文件（如果存在）会被自动覆盖。
        # 无需用户手动删除或在此处添加显式删除逻辑。
        # print(f"提示: 将使用写入模式 ('w') 创建/覆盖列表文件和元数据文件。")
        # print(f"       如果 '{train_file_path}', '{test_file_path}', 或 '{meta_file_path}' 已存在，它们的内容将被覆盖。")

        # --- 类别发现与统计初始化 ---
        class_details = [] # 存储所有类别详细信息的列表
        
        # 获取根目录下所有的类别子文件夹 (每个子文件夹代表一个类别)
        try:
            # os.listdir 列出目录下所有文件和文件夹名
            # os.path.isdir 检查给定路径是否为目录
            class_dirs_unordered = [d for d in os.listdir(data_root_path) if os.path.isdir(os.path.join(data_root_path, d))]
            if not class_dirs_unordered:
                print(f"错误: 在目录 '{data_root_path}' 下没有找到任何子文件夹作为类别。请检查数据集结构。")
                return
        except FileNotFoundError:
            print(f"错误: 指定的数据集根目录 '{data_root_path}' 不存在。请提供有效的路径。")
            return
        except Exception as e:
            print(f"错误: 访问数据集目录 '{data_root_path}' 时发生意外：{e}")
            return

        # 对类别文件夹进行排序，确保每次生成的标签ID具有一致性。
        # 例如，如果类别是 ['person_c', 'person_a', 'person_b']，排序后会是 ['person_a', 'person_b', 'person_c']，
        # 这样 'person_a' 总是会得到标签0，'person_b' 总是标签1，以此类推。
        class_dirs_sorted = sorted(class_dirs_unordered)

        current_class_label = 0  # 类别标签从0开始递增
        total_images_count = 0   # 数据集中所有图像的总数

        # --- 打开列表文件并处理每个类别 ---
        # 使用 'w' (写入模式) 打开文件，这样如果文件已存在，其内容会被清空并从头开始写入。
        # 使用 `with open(...) as ...:` 结构可以确保文件在使用完毕后自动关闭，即使发生错误。
        # `encoding='utf-8'` 确保能正确处理包含非ASCII字符的文件路径或类别名。
        try:
            with open(train_file_path, 'w', encoding='utf-8') as f_train, \
                 open(test_file_path, 'w', encoding='utf-8') as f_test:
                
                print(f"在 '{data_root_path}' 中找到 {len(class_dirs_sorted)} 个类别 (子文件夹): {class_dirs_sorted}")
                
                # 遍历每个已排序的类别子文件夹
                for class_dir_name in class_dirs_sorted:
                    class_full_path = os.path.join(data_root_path, class_dir_name)
                    
                    # 获取当前类别文件夹下的所有文件
                    try:
                        all_files_in_class = [f for f in os.listdir(class_full_path) if os.path.isfile(os.path.join(class_full_path, f))]
                    except Exception as e:
                        print(f"警告: 无法读取类别文件夹 '{class_dir_name}' (路径: {class_full_path}) 下的文件列表: {e}。已跳过此类别。")
                        continue # 跳到下一个类别

                    # 过滤常见非图片文件，或只选择特定后缀的图片
                    # 确保只处理常见的图像文件扩展名，转换为小写以进行不区分大小写的比较。
                    valid_image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
                    image_files_in_class_unsorted = [f for f in all_files_in_class if f.lower().endswith(valid_image_extensions)]
                    
                    # 对每个类别下的图片文件名排序，保证处理顺序一致，进而使得划分到训练/测试集的结果更稳定。
                    image_files_in_class_sorted = sorted(image_files_in_class_unsorted)

                    if not image_files_in_class_sorted:
                        print(f"警告: 类别文件夹 '{class_dir_name}' 为空或不包含有效格式的图片文件 (支持的格式: {valid_image_extensions})。已跳过此类别。")
                        continue # 跳到下一个类别

                    images_in_current_class_count = 0 # 当前类别的图像计数器
                    test_images_in_current_class = 0    # 当前类别分到测试集的图像数量
                    train_images_in_current_class = 0   # 当前类别分到训练集的图像数量

                    print(f"  正在处理类别: '{class_dir_name}' (将分配标签ID: {current_class_label}), 发现 {len(image_files_in_class_sorted)} 张有效图片...")
                    
                    # 遍历当前类别中的每一张已排序的图片
                    for image_file_name in image_files_in_class_sorted:
                        # 构建图像文件的完整路径 (相对于项目根目录或数据集根目录，取决于 data_root_path 的形式)
                        # 为了使列表文件中的路径具有通用性，通常希望这里的路径是相对于 data_root_path 的。
                        # 例如，如果 data_root_path 是 'data/face'，class_dir_name 是 'person1'，
                        # image_file_name 是 'img1.jpg'，则希望写入列表的是 'person1/img1.jpg'。
                        # 因此，使用 os.path.join(class_dir_name, image_file_name) 而不是完整的 class_full_path。
                        relative_image_path = os.path.join(class_dir_name, image_file_name)
                        # 在Windows上，路径分隔符可能是'\\', 统一转换为'/'以保证跨平台兼容性
                        relative_image_path = relative_image_path.replace('\\\\', '/')
                        
                        # 数据划分策略：例如，每10张图片中取1张作为测试数据，其余作为训练数据。
                        # 这是一个简单的基于计数的划分方法。
                        # images_in_current_class_count 是当前类别已处理图像的序号 (0-indexed)。
                        if images_in_current_class_count % 10 == 0: # 每10张取第1张 (索引0, 10, 20...)
                            f_test.write(f"{relative_image_path}\t{current_class_label}\n")
                            test_images_in_current_class += 1
                        else:
                            f_train.write(f"{relative_image_path}\t{current_class_label}\n")
                            train_images_in_current_class += 1
                        
                        images_in_current_class_count += 1
                        total_images_count += 1 # 累加整个数据集的图像总数
                    
                    # 记录当前类别的详细信息，用于后续生成元数据文件
                    class_info = {
                        'class_name': class_dir_name,            # 类别名称 (即子文件夹名)
                        'class_label': current_class_label,      # 分配给该类别的整数标签ID (0-indexed)
                        'class_images_count': images_in_current_class_count, # 该类别下的图像总数
                        'class_train_images': train_images_in_current_class, # 分配到训练集的数量
                        'class_test_images': test_images_in_current_class    # 分配到测试集的数量
                    }
                    class_details.append(class_info)
                    current_class_label += 1 # 为下一个类别准备标签ID
        except IOError as e:
            print(f"错误: 写入训练/测试列表文件时发生IO错误: {e}")
            return # 发生IO错误，通常无法继续
        except Exception as e:
            print(f"错误: 处理类别或写入列表文件时发生意外: {e}")
            return

        print(f"训练数据列表已成功保存到: {train_file_path}")
        print(f"测试数据列表已成功保存到: {test_file_path}")

        # --- 生成并保存元数据 (readme.json) --- 
        # 获取数据集的顶层目录名作为'数据集名称' (例如 'face')
        # data_root_path.rstrip('/') 移除路径末尾可能存在的斜杠，保证basename行为一致
        dataset_name = os.path.basename(data_root_path.rstrip(os.sep)) # 使用os.sep保证跨平台
        
        meta_data_to_save = {
            'dataset_name': dataset_name,                     # 数据集的名称 (通常是根文件夹名)
            'total_classes': len(class_details),             # 数据集中总的类别数量
            'total_images': total_images_count,              # 数据集中所有图像的总数
            'class_detail': class_details                    # 每个类别的详细信息列表
        }
        
        try:
            with open(meta_file_path, 'w', encoding='utf-8') as f_meta:
                # json.dump 用于将Python字典序列化为JSON格式并写入文件。
                # ensure_ascii=False: 允许JSON文件中直接包含非ASCII字符 (如中文类别名)，而不是转义为 \\uXXXX。
                # indent=4: 对JSON输出进行格式化，使其更易读（使用4个空格缩进）。
                json.dump(meta_data_to_save, f_meta, ensure_ascii=False, indent=4) 
            print(f"元数据文件已成功保存到: {meta_file_path}")
        except IOError as e:
            print(f"错误: 无法写入元数据文件 '{meta_file_path}': {e}")
        except TypeError as e:
            print(f"错误: 序列化元数据到JSON时发生类型错误 (通常是数据结构问题): {e}")
        except Exception as e:
            print(f"错误: 保存元数据文件时发生意外: {e}")
            
        print(f"\n数据列表和元数据创建完成！")
        print(f"总共处理了 {total_images_count} 张图片，这些图片分属于 {len(class_details)} 个类别。")

# --- 脚本主入口 ---
if __name__ == '__main__':
    """
    当该脚本作为主程序直接运行时，执行此处的代码。
    这里提供了一个简单的示例，演示如何实例化 CreateDataList 类并调用其方法。
    """
    print("CreateDataList.py 脚本正在作为主程序运行...")
    
    # 实例化数据列表创建器
    data_lister_instance = CreateDataList()
    
    # --- 用户配置区域 ---
    # 指定你的数据集根目录。
    # 请将此路径替换为你的实际数据集所在的根目录。
    # 例如: target_dataset_root = 'data/face' 
    #      或者 target_dataset_root = 'path/to/your/image_dataset'
    target_dataset_root = 'data/face'  # 默认示例数据集路径
    
    # (可选) 自定义输出文件名
    # custom_train_list_filename = "my_train.txt"
    # custom_test_list_filename = "my_test.txt"
    # custom_meta_filename = "dataset_info.json"
    
    print(f"\n将为数据集位于 '{target_dataset_root}' 的数据创建或更新数据列表和元数据文件。")
    print("重要提示: 如果目标目录下已存在同名列表文件和元数据文件，它们的内容将会被覆盖。")
    
    # 调用核心方法来创建数据列表和元数据文件
    # 如果需要自定义输出文件名，可以传递相应的参数给 create_data_list 方法，例如：
    # data_lister_instance.create_data_list(
    #     target_dataset_root,
    #     train_list_name=custom_train_list_filename,
    #     test_list_name=custom_test_list_filename,
    #     meta_file_name=custom_meta_filename
    # )
    data_lister_instance.create_data_list(target_dataset_root)
    
    print("\n脚本执行完毕。请检查在数据集根目录下生成的列表文件 (trainer.list, test.list) 和元数据文件 (readme.json)。") 