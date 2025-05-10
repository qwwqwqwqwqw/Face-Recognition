# coding=utf-8
import os
import json

class CreateDataList:
    """用于创建图像数据列表和元数据文件的类"""
    def __init__(self):
        """初始化CreateDataList类"""
        pass # 目前不需要特殊初始化操作

    def create_data_list(self, data_root_path, train_list_name="trainer.list", test_list_name="test.list", meta_file_name="readme.json"):
        """
        遍历指定的数据集根目录，为其中的图像生成训练列表、测试列表和元数据JSON文件。

        Args:
            data_root_path (str): 数据集的根目录。此目录下应包含对应各类别的子文件夹，
                                  每个子文件夹中存放该类别的图像。
                                  例如: data_root_path = 'data/face'
                                        'data/face/person1/img1.jpg'
                                        'data/face/person2/img2.jpg'
            train_list_name (str): 生成的训练数据列表文件名。
            test_list_name (str): 生成的测试数据列表文件名。
            meta_file_name (str): 生成的元数据JSON文件名。
        """
        print(f"开始为数据集 {data_root_path} 创建数据列表...")
        
        # 训练列表、测试列表和元数据文件的完整路径
        # 这些文件将直接保存在 data_root_path 目录下
        train_file_path = os.path.join(data_root_path, train_list_name)
        test_file_path = os.path.join(data_root_path, test_list_name)
        meta_file_path = os.path.join(data_root_path, meta_file_name)

        # --- 清理旧的列表文件 --- 
        # 为防止追加写入导致数据重复或混淆，在重新生成列表前，先删除已存在的旧文件。
        # 用户应在调用此脚本前手动删除，或在此处添加自动删除逻辑（需谨慎）。
        # print(f"提示: 如果 {train_file_path}, {test_file_path}, 或 {meta_file_path} 已存在，建议先手动删除它们以避免内容追加导致的问题。")
        # 更安全的做法是在打开文件时使用 'w' (写入模式) 而不是 'a' (追加模式)，这样会自动覆盖旧文件。
        # 以下代码将使用 'w' 模式来创建/覆盖列表文件和json文件。

        # 存储所有类别详细信息的列表
        class_details = []
        # 获取根目录下所有的类别子文件夹 (每个子文件夹代表一个类别)
        try:
            class_dirs = [d for d in os.listdir(data_root_path) if os.path.isdir(os.path.join(data_root_path, d))]
            if not class_dirs:
                print(f"错误: 在目录 {data_root_path} 下没有找到任何子文件夹作为类别。请检查数据集结构。")
                return
        except FileNotFoundError:
            print(f"错误: 指定的数据集根目录 {data_root_path} 不存在。")
            return

        # 对类别文件夹进行排序，确保每次生成的标签ID一致性
        class_dirs.sort()

        current_class_label = 0  # 类别标签从0开始递增
        total_images_count = 0   # 数据集中所有图像的总数

        # --- 打开列表文件 (使用写入模式 'w' 来覆盖旧文件) --- 
        with open(train_file_path, 'w', encoding='utf-8') as f_train, \
             open(test_file_path, 'w', encoding='utf-8') as f_test:
            
            print(f"找到 {len(class_dirs)} 个类别: {class_dirs}")
            # 遍历每个类别子文件夹
            for class_dir_name in class_dirs:
                class_path = os.path.join(data_root_path, class_dir_name)
                image_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
                
                # 过滤常见非图片文件，或只选择特定后缀的图片
                valid_image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
                image_files = [f for f in image_files if f.lower().endswith(valid_image_extensions)]
                image_files.sort() # 对每个类别下的图片文件名排序，保证处理顺序一致

                if not image_files:
                    print(f"警告: 类别文件夹 '{class_dir_name}' 为空或不包含有效图片文件，已跳过。")
                    continue

                images_in_current_class = 0 # 当前类别的图像计数器
                test_images_in_class = 0    # 当前类别分到测试集的图像数量
                train_images_in_class = 0   # 当前类别分到训练集的图像数量

                print(f"  正在处理类别: '{class_dir_name}' (标签ID: {current_class_label}), 包含 {len(image_files)} 张图片...")
                for image_file_name in image_files:
                    full_image_path = os.path.join(class_path, image_file_name)
                    
                    # 数据划分策略：例如，每10张图片中取1张作为测试数据，其余作为训练数据
                    # 为了更均匀的划分，可以考虑对每个类别固定比例划分，或者使用随机数
                    # 当前实现是每10张取第1, 11, 21... 张为测试数据
                    if images_in_current_class % 10 == 0:
                        f_test.write(f"{full_image_path}\t{current_class_label}\n")
                        test_images_in_class += 1
                    else:
                        f_train.write(f"{full_image_path}\t{current_class_label}\n")
                        train_images_in_class += 1
                    
                    images_in_current_class += 1
                    total_images_count += 1
                
                # 记录当前类别的详细信息
                class_info = {
                    'class_name': class_dir_name,
                    'class_label': current_class_label, # 标签ID (0-indexed)
                    'class_images_count': images_in_current_class,
                    'class_train_images': train_images_in_class,
                    'class_test_images': test_images_in_class
                }
                class_details.append(class_info)
                current_class_label += 1 # 更新到下一个类别的标签ID
        
        print(f"训练列表已保存到: {train_file_path}")
        print(f"测试列表已保存到: {test_file_path}")

        # --- 生成并保存元数据 (readme.json) --- 
        # 获取数据集的顶层目录名作为'数据集名称' (例如 'face')
        dataset_name = os.path.basename(data_root_path.rstrip('/'))
        
        meta_data = {
            'dataset_name': dataset_name,             # 数据集名称
            'total_classes': len(class_dirs),        # 总类别数量
            'total_images': total_images_count,       # 数据集中总图像数量
            'class_detail': class_details             # 每个类别的详细信息列表
        }
        
        try:
            with open(meta_file_path, 'w', encoding='utf-8') as f_meta:
                json.dump(meta_data, f_meta, ensure_ascii=False, indent=4) # indent=4 格式化输出, ensure_ascii=False 支持中文
            print(f"元数据文件已保存到: {meta_file_path}")
        except IOError as e:
            print(f"错误: 无法写入元数据文件 {meta_file_path}: {e}")
        except TypeError as e:
            print(f"错误: 序列化元数据到JSON时发生错误: {e}")
            
        print(f"数据列表创建完成！总共处理了 {total_images_count} 张图片，分属于 {len(class_dirs)} 个类别。")


if __name__ == '__main__':
    # 实例化创建器
    data_lister = CreateDataList()
    
    # 指定你的数据集根目录
    # 例如: 'data/face' 或者 'data/my_custom_dataset'
    target_dataset_path = 'data/face' 
    
    print(f"将为数据集 '{target_dataset_path}' 创建或更新数据列表和元数据文件。")
    print("注意: 已存在的同名列表和元数据文件将被覆盖。")
    
    # 调用方法创建数据列表和元数据文件
    data_lister.create_data_list(target_dataset_path)
    
    print("\n脚本执行完毕。请检查生成的文件。") 