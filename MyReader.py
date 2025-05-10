# coding:utf-8
import os
import cv2
import random
import numpy as np
import paddle
from paddle.io import Dataset, DataLoader

class ImageAugmentation:
    """图像数据增强和预处理类。

    该类封装了多种常见的图像增强技术，如随机翻转、颜色抖动、亮度调整等，
    以及图像的缩放和归一化操作。
    它区分训练模式和评估/测试模式，在训练时应用数据增强以提高模型泛化能力，
    在评估/测试时仅进行必要的尺寸调整和归一化。
    """

    def __init__(self, image_size: int | tuple[int, int] = 64, 
                 mean_value: list[float] = [0.5, 0.5, 0.5], 
                 std_value: list[float] = [0.5, 0.5, 0.5], 
                 is_train: bool = True):
        """
        ImageAugmentation 初始化函数。

        Args:
            image_size (int or tuple, optional): 目标图像尺寸。如果是整数，则假设为正方形 (H=W=image_size)。
                                               如果是元组 (H, W)，则分别指定高度和宽度。
                                               默认为 64。
            mean_value (list[float], optional): 图像归一化时使用的均值 (按通道，通常为RGB顺序)。
                                             长度应为3。默认为 [0.5, 0.5, 0.5]。
            std_value (list[float], optional): 图像归一化时使用的标准差 (按通道，通常为RGB顺序)。
                                            长度应为3。默认为 [0.5, 0.5, 0.5]。
            is_train (bool, optional): 指示当前是否为训练模式。如果是训练模式，则会应用数据增强；
                                       否则 (评估/测试模式)，仅进行尺寸调整和归一化。
                                       默认为 True。
        """
        if isinstance(image_size, int):
            self.image_height = image_size
            self.image_width = image_size
        else:
            self.image_height = image_size[0]
            self.image_width = image_size[1]
        
        self.mean = np.array(mean_value).reshape(3, 1, 1).astype('float32')
        self.std = np.array(std_value).reshape(3, 1, 1).astype('float32')
        self.is_train = is_train

    def process_image(self, image_path: str) -> np.ndarray:
        """
        加载、预处理并（如果is_train=True）增强单个图像。

        流程包括：
        1. 使用OpenCV加载图像。
        2. 如果是训练模式，则应用随机数据增强：
            - 随机水平翻转。
            - (可选，可在此处添加) 随机裁剪、旋转、颜色抖动、亮度/对比度调整等。
        3. 将图像缩放到指定的目标尺寸 (self.image_height, self.image_width)。
        4. 转换图像数据类型为 float32。
        5. 转换 HWC (OpenCV) 到 CHW (PaddlePaddle)
        6. 对图像进行归一化 (减均值，除以标准差)。

        Args:
            image_path (str): 要处理的图像文件的完整路径。

        Returns:
            np.ndarray: 处理后的图像数据，格式为 CHW，float32类型，已归一化。
                        可以直接作为模型输入。
        Raises:
            FileNotFoundError: 如果指定的 image_path 不存在。
            Exception: 如果图像加载或处理过程中发生其他OpenCV错误。
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件未找到: {image_path}")
        try:
            # 1. 使用OpenCV加载图像 (BGR格式)
            img = cv2.imread(image_path)
            if img is None:
                raise Exception(f"无法加载图像: {image_path}。文件可能已损坏或格式不支持。")

            # 2. 数据增强 (仅在训练时)
            if self.is_train:
                # 随机水平翻转
                if random.random() > 0.5:
                    img = cv2.flip(img, 1) # 1表示水平翻转，0表示垂直，-1表示水平垂直都翻转
                
                # 示例：随机亮度调整 (可以根据需要扩展)
                # if random.random() > 0.5:
                #     brightness_factor = random.uniform(0.7, 1.3)
                #     img = np.clip(img * brightness_factor, 0, 255).astype(img.dtype)

                # 示例：随机对比度调整
                # if random.random() > 0.5:
                #     contrast_factor = random.uniform(0.7, 1.3)
                #     # 简单的对比度调整: f(x) = alpha*x + beta
                #     # 这里简化为乘以一个因子，然后通过均值调整回原始范围（更复杂的方法会更好）
                #     mean_val = img.mean()
                #     img = np.clip((img - mean_val) * contrast_factor + mean_val, 0, 255).astype(img.dtype)

                # TODO: 根据需要添加更多数据增强方法，例如：
                # - 随机裁剪 (Random Cropping)
                # - 随机旋转 (Random Rotation)
                # - 颜色抖动 (Color Jittering: 亮度、对比度、饱和度、色调)
                # - 高斯模糊 (Gaussian Blur)
                # - 随机擦除 (Random Erasing)
                pass

            # 3. 图像缩放 (resize)
            # 使用 INTER_LINEAR 插值方法，这是一种在效果和速度之间取得较好平衡的方法。
            # 对于缩小图像，INTER_AREA 可能效果更好；对于放大，INTER_CUBIC 可能效果更好但更慢。
            img_resized = cv2.resize(img, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)

            # 4. 转换数据类型为 float32
            img_float = img_resized.astype('float32')

            # 5. 转换 HWC (OpenCV) 到 CHW (PaddlePaddle)
            # OpenCV读取的图像是 H x W x C (BGR顺序)
            # 我们需要将其转换为 C x H x W (通常期望RGB顺序，但这里直接用BGR做后续处理)
            # 如果模型期望RGB，需要在加载后或此处进行 BGR -> RGB 转换：
            # img_float = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)
            img_chw = img_float.transpose((2, 0, 1)) # HWC to CHW

            # 6. 归一化
            #   img_normalized = (img_chw / 255.0 - self.mean) / self.std
            #   为了与许多预训练模型的常见做法对齐，先将像素值从 [0, 255] 缩放到 [0, 1]
            #   然后再进行减均值除以标准差的操作。
            #   注意: self.mean 和 self.std 应该是针对 [0,1] 范围的图像计算得到的。
            #         如果它们是针对 [0,255] 范围计算的，则不应除以255.0。
            #         假设这里的mean/std是针对[0,1]范围。
            img_normalized = (img_chw / 255.0 - self.mean) / self.std
            
            return img_normalized
        
        except FileNotFoundError: # 再次捕获以确保信息传递
            raise
        except Exception as e:
            print(f"处理图像 '{image_path}' 时发生错误: {e}")
            # 可以选择返回一个占位符图像或None，或者重新抛出异常，取决于上层如何处理错误
            # 例如，返回一个全黑的图像：
            # return np.zeros((3, self.image_height, self.image_width), dtype='float32')
            raise Exception(f"图像处理失败: {image_path}, 原因: {str(e)}")


class CustomDataset(paddle.io.Dataset):
    """自定义数据集类，与 PaddlePaddle 的 `paddle.io.Dataset` 兼容。

    该数据集负责从一个包含图像路径和对应标签的列表文件中读取数据，
    并使用传入的 `ImageAugmentation` 对象对每个图像进行加载和预处理。
    """
    def __init__(self, data_list_file: str, image_augmentor: ImageAugmentation, 
                 image_files_base_dir: str, mode: str = 'train'):
        """
        CustomDataset 初始化函数。

        Args:
            data_list_file (str): 数据列表文件的路径。
                                  该文件应每行包含一个样本，格式通常为：`图像相对路径\t类别标签ID`。
                                  例如：`person1/img001.jpg	0`。
            image_augmentor (ImageAugmentation): 用于处理图像的 `ImageAugmentation` 实例。
            image_files_base_dir (str): 列表文件中图像路径的基准目录。
                                        例如，如果列表中的路径是 'person1/img001.jpg'，
                                        且此文件实际位于 'data/face/person1/img001.jpg'，
                                        则此参数应为 'data/face'。
            mode (str, optional): 当前数据集的模式 ('train', 'eval', 'test')。
                                  此参数主要用于日志记录或调试，实际的增强行为由 `image_augmentor.is_train` 控制。
                                  默认为 'train'。
        """
        super(CustomDataset, self).__init__()
        self.image_augmentor = image_augmentor
        self.image_files_base_dir = image_files_base_dir
        self.mode = mode
        self.samples = []

        if not os.path.exists(data_list_file):
            raise FileNotFoundError(f"数据列表文件未找到: {data_list_file}")

        try:
            with open(data_list_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: # 跳过空行
                        continue
                    parts = line.split('\t')
                    if len(parts) == 2:
                        img_path, label_str = parts
                        # 确保标签是整数类型
                        try:
                            label = int(label_str)
                            self.samples.append((img_path, label))
                        except ValueError:
                            print(f"警告: 在文件 '{data_list_file}' 中发现无效的标签 '{label_str}' (行: '{line}')，已跳过。")
                    elif len(parts) == 1 and self.mode == 'infer': # 推理模式可能只有图像路径
                        img_path = parts[0]
                        self.samples.append((img_path, -1)) # 推理时标签可以设为-1或其他占位符
                    else:
                        print(f"警告: 在文件 '{data_list_file}' 中发现格式不正确的行: '{line}' (分割后部分数: {len(parts)})，已跳过。")
        except Exception as e:
            raise RuntimeError(f"读取数据列表文件 '{data_list_file}' 失败: {e}")
        
        if not self.samples:
            print(f"警告: 从 '{data_list_file}' 加载的样本为空。请检查文件内容和格式。")
        else:
            print(f"成功从 '{data_list_file}' 加载 {len(self.samples)} 个样本用于模式 '{self.mode}'。")

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, str]:
        """
        根据索引获取单个数据样本 (处理后的图像和标签)。

        Args:
            idx (int): 要获取的数据样本的索引。

        Returns:
            tuple:
                - processed_image (np.ndarray): 经过 `ImageAugmentation` 处理后的图像数据 (CHW, float32)。
                - label (np.ndarray or str): 
                    如果标签是整数类别ID，则返回一个包含该ID的 `np.ndarray` (int64类型，形状为[1])。
                    如果原始样本中是图像路径（例如用于某些类型的推理或对比），则可能返回图像路径字符串。
                    当前实现中，标签总是被转换为整数，所以这里总是返回 `np.ndarray`。
                    对于推理模式且标签为-1的情况，仍返回 `np.array([-1], dtype='int64')`。
        """
        img_relative_path, label_id = self.samples[idx]
        
        # 构建图像的完整路径
        full_img_path = os.path.join(self.image_files_base_dir, img_relative_path)
        
        try:
            # 调用图像增强器处理图像 (使用完整路径)
            processed_image = self.image_augmentor.process_image(full_img_path)
            
            # 将标签转换为PaddlePaddle期望的格式 (通常是int64类型的Tensor)
            label_tensor_like = np.array([label_id], dtype='int64') 
            
            return processed_image, label_tensor_like
        except Exception as e:
            print(f"错误: 获取索引 {idx} (图像路径: '{full_img_path}') 的数据时失败: {e}")
            # 当一个样本处理失败时，可以选择：
            # 1. 重新抛出异常，这可能会中断整个dataloader的迭代。
            # 2. 返回一个特殊的占位符样本（需要dataloader的collate_fn能处理）。
            # 3. 跳过这个样本（这在Dataset API中不易直接实现，通常在创建samples列表时过滤）。
            # 这里选择重新抛出，以便上层捕获并决定如何处理。
            raise

    def __len__(self) -> int:
        """返回数据集中样本的总数。"""
        return len(self.samples)


def create_data_loader(config, mode: str, custom_dataset_class=CustomDataset) -> paddle.io.DataLoader:
    """
    创建并返回一个 PaddlePaddle DataLoader 实例。

    该函数根据提供的配置和模式（'train', 'eval', 'test', 'infer'）来设置数据加载器。
    它会实例化 `ImageAugmentation` 和 `CustomDataset` (或指定的 `custom_dataset_class`)。

    Args:
        config (ConfigObject): 项目的全局配置对象，应包含以下相关参数：
            - `image_size` (int or tuple): 目标图像尺寸。
            - `dataset_params.mean` (list[float]): 归一化均值。
            - `dataset_params.std` (list[float]): 归一化标准差。
            - `dataset_params.train_list` (str): 训练数据列表文件路径。
            - `dataset_params.eval_list` (str, optional): 评估数据列表文件路径。
            - `dataset_params.test_list` (str, optional): 测试数据列表文件路径。
            - `dataset_params.infer_list` (str, optional): 推理数据列表文件路径 (可能格式不同)。
            - `batch_size` (int): 每个批次中的样本数量。
            - `dataset_params.num_workers` (int, optional): 用于数据加载的子进程数量。默认为0 (在主进程中加载)。
        mode (str): 指定数据加载的模式。可选值：'train', 'eval', 'test', 'infer'。
                    根据模式选择不同的数据列表文件和增强策略 (is_train)。
        custom_dataset_class (type[paddle.io.Dataset], optional): 
            允许传入自定义的数据集类 (必须继承自 `paddle.io.Dataset`)。
            默认为本文件中定义的 `CustomDataset`。

    Returns:
        paddle.io.DataLoader: 配置好的 PaddlePaddle 数据加载器实例。

    Raises:
        ValueError: 如果 `mode` 无效，或者所需的配置参数缺失。
        FileNotFoundError: 如果对应模式的数据列表文件未在配置中指定或文件不存在。
    """
    is_train = (mode == 'train')
    
    # 从配置中获取参数
    image_size = config.get('image_size', 64)
    dataset_params = config.get('dataset_params', {})
    mean_val = dataset_params.get('mean', [0.5, 0.5, 0.5])
    std_val = dataset_params.get('std', [0.5, 0.5, 0.5])
    num_workers = dataset_params.get('num_workers', 0)
    batch_size = config.get('batch_size', 32)
    
    # 根据模式选择数据列表文件
    # 首先获取列表文件名 (例如 "trainer.list")
    list_filename_only = None
    if mode == 'train':
        list_filename_only = dataset_params.get('train_list')
        shuffle = True # 训练时通常需要打乱数据
    elif mode == 'eval':
        list_filename_only = dataset_params.get('eval_list')
        shuffle = False
    elif mode == 'test':
        list_filename_only = dataset_params.get('test_list')
        shuffle = False
    elif mode == 'infer': # 推理模式特定处理
        list_filename_only = dataset_params.get('infer_list')
        shuffle = False
        # 推理时，batch_size 通常设为1或根据模型和硬件能力调整
        # batch_size = config.get('infer_batch_size', 1) # 可以单独配置推理的batch_size
    else:
        raise ValueError(f"无效的数据加载模式: '{mode}'. 支持的模式为: 'train', 'eval', 'test', 'infer'。")

    if not list_filename_only:
        raise FileNotFoundError(f"模式 '{mode}' 所需的数据列表文件名未在配置的 dataset_params 中指定 (例如 config.dataset_params.{mode}_list)。")

    # 构建数据列表文件的完整路径
    # 列表文件位于 config.data_dir / config.class_name / list_filename_only
    dataset_list_file_dir = os.path.join(config.data_dir, config.class_name)
    actual_data_list_file_path = os.path.join(dataset_list_file_dir, list_filename_only)

    if not os.path.exists(actual_data_list_file_path):
         raise FileNotFoundError(f"模式 '{mode}' 的数据列表文件 '{actual_data_list_file_path}' 未找到。请确保它存在于 '{dataset_list_file_dir}' 目录下，或先运行 CreateDataList.py 生成。")

    # 图像文件路径的基准目录 (与列表文件所在的目录相同)
    image_files_base_dir = dataset_list_file_dir

    # 实例化图像增强/预处理对象
    augmentor = ImageAugmentation(
        image_size=image_size,
        mean_value=mean_val,
        std_value=std_val,
        is_train=is_train
    )

    # 实例化自定义数据集
    dataset = custom_dataset_class(
        data_list_file=actual_data_list_file_path, # 使用构建好的完整路径
        image_augmentor=augmentor,
        image_files_base_dir=image_files_base_dir, # 传递基准目录
        mode=mode
    )

    # 创建 DataLoader
    # drop_last: 如果最后一个批次的样本数小于batch_size，是否丢弃它。
    #            训练时通常设为True，以保证每个batch的形状一致，避免BN层等出现问题。
    #            评估/测试时通常设为False，以确保所有样本都被评估。
    drop_last_setting = True if mode == 'train' else False
    
    data_loader = paddle.io.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last_setting, 
        use_shared_memory=False # 根据系统和数据量调整，Windows下多进程可能与共享内存存在问题
    )
    
    print(f"为模式 '{mode}' 创建 DataLoader 成功。Batch size: {batch_size}, Shuffle: {shuffle}, Num workers: {num_workers}, Drop last: {drop_last_setting}")
    return data_loader


if __name__ == '__main__':
    # MyReader.py 的单元测试和使用示例
    print("开始测试 MyReader.py ...")

    # 准备一个假的配置文件对象 (模拟ConfigObject)
    class MockConfig:
        def __init__(self):
            self.image_size = 64
            self.batch_size = 2 # 测试时用小批量
            self.dataset_params = {
                'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5],
                'num_workers': 0,
                # 需要确保这些文件存在且内容符合格式
                'train_list': 'temp_train_list.txt', 
                'eval_list': 'temp_eval_list.txt',
                'infer_list': 'temp_infer_list.txt'
            }
        def get(self, key, default=None):
            return getattr(self, key, default)

    mock_config = MockConfig()

    # 准备假的列表文件和图像数据
    base_data_dir = 'temp_test_data'
    if not os.path.exists(base_data_dir):
        os.makedirs(base_data_dir)
    
    sample_images_info = {
        'class_a': ['img1.png', 'img2.jpg'],
        'class_b': ['img3.png']
    }

    # 创建假的图像文件 (全黑的PNG)
    def create_dummy_image(path, size=(64,64)):
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        cv2.imwrite(path, img)

    for class_name, img_names in sample_images_info.items():
        class_dir = os.path.join(base_data_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        for img_name in img_names:
            create_dummy_image(os.path.join(class_dir, img_name))

    # 创建假的列表文件
    with open(mock_config.dataset_params['train_list'], 'w') as f:
        f.write(f"{os.path.join(base_data_dir, 'class_a', 'img1.png')}\t0\n")
        f.write(f"{os.path.join(base_data_dir, 'class_a', 'img2.jpg')}\t0\n")
        f.write(f"{os.path.join(base_data_dir, 'class_b', 'img3.png')}\t1\n")
    
    with open(mock_config.dataset_params['eval_list'], 'w') as f:
        f.write(f"{os.path.join(base_data_dir, 'class_a', 'img1.png')}\t0\n")

    with open(mock_config.dataset_params['infer_list'], 'w') as f:
        f.write(f"{os.path.join(base_data_dir, 'class_b', 'img3.png')}\n") # 推理列表可能只有路径

    print("\n--- 测试 ImageAugmentation ---")
    augmentor_train = ImageAugmentation(image_size=mock_config.image_size, is_train=True)
    augmentor_eval = ImageAugmentation(image_size=mock_config.image_size, is_train=False)
    
    sample_img_path = os.path.join(base_data_dir, 'class_a', 'img1.png')
    try:
        processed_train_img = augmentor_train.process_image(sample_img_path)
        processed_eval_img = augmentor_eval.process_image(sample_img_path)
        print(f"ImageAugmentation (train) 输出形状: {processed_train_img.shape}, 类型: {processed_train_img.dtype}")
        print(f"ImageAugmentation (eval) 输出形状: {processed_eval_img.shape}, 类型: {processed_eval_img.dtype}")
        assert processed_train_img.shape == (3, mock_config.image_size, mock_config.image_size)
        assert processed_eval_img.dtype == np.float32
        print("ImageAugmentation 初步测试通过。")
    except Exception as e:
        print(f"ImageAugmentation 测试失败: {e}")

    print("\n--- 测试 CustomDataset 和 create_data_loader (train模式) ---")
    try:
        train_loader = create_data_loader(mock_config, mode='train')
        for i, (images, labels) in enumerate(train_loader):
            print(f"Train Batch {i+1}: Images shape {images.shape}, Labels shape {labels.shape}, Labels dtype {labels.dtype}")
            print(f"  Sample labels: {labels.numpy().flatten()[:5]}") # 打印部分标签
            assert images.shape == (mock_config.batch_size if i < len(train_loader)-1 or len(train_loader.dataset) % mock_config.batch_size == 0 else len(train_loader.dataset) % mock_config.batch_size, 
                                     3, mock_config.image_size, mock_config.image_size) if not train_loader.drop_last or len(train_loader.dataset) >= mock_config.batch_size else True 
            # 更简单的校验，如果drop_last=True且样本数不足一个batch，则loader为空
            if train_loader.drop_last and len(train_loader.dataset) < mock_config.batch_size:
                 assert False, "Drop_last is True but dataset is smaller than batch_size, loader should be empty or not enter loop."
            elif not train_loader.drop_last and len(train_loader.dataset) > 0 : # 确保至少有一个batch被迭代
                pass 
            elif len(train_loader.dataset) == 0: # 如果数据集为空
                pass
            else: # 正常情况
                 assert images.shape[0] <= mock_config.batch_size

            assert labels.dtype == paddle.int64
            if i >= 1: # 测试少量批次即可
                break
        if len(train_loader.dataset) > 0 and i == 0 and len(train_loader) == 0 : # 处理drop_last=True且样本不足一个batch的情况
            print("训练数据加载器: drop_last=True, 样本数小于batch_size，加载器为空或迭代未执行，符合预期。")
        elif len(train_loader.dataset) == 0:
            print("训练数据加载器: 数据集为空，迭代未执行，符合预期。")
        else:
            print("Train DataLoader 初步迭代测试通过。")
    except Exception as e:
        print(f"Train DataLoader 测试失败: {e}")

    print("\n--- 测试 CustomDataset 和 create_data_loader (eval模式) ---")
    try:
        eval_loader = create_data_loader(mock_config, mode='eval')
        for i, (images, labels) in enumerate(eval_loader):
            print(f"Eval Batch {i+1}: Images shape {images.shape}, Labels shape {labels.shape}")
            # 对于eval，drop_last=False，所以最后一个batch可能不等于batch_size
            assert images.shape[1:] == (3, mock_config.image_size, mock_config.image_size)
            assert images.shape[0] <= mock_config.batch_size 
            if i >= 1: break
        print("Eval DataLoader 初步迭代测试通过。")
    except Exception as e:
        print(f"Eval DataLoader 测试失败: {e}")

    print("\n--- 测试 CustomDataset 和 create_data_loader (infer模式) ---")
    try:
        infer_loader = create_data_loader(mock_config, mode='infer')
        for i, (images, labels) in enumerate(infer_loader):
            print(f"Infer Batch {i+1}: Images shape {images.shape}, Labels shape {labels.shape}, Labels: {labels.numpy().flatten()}")
            assert images.shape[1:] == (3, mock_config.image_size, mock_config.image_size)
            assert labels.numpy()[0][0] == -1 # 推理模式下标签应为-1
            if i >= 1: break
        print("Infer DataLoader 初步迭代测试通过。")
    except Exception as e:
        print(f"Infer DataLoader 测试失败: {e}")

    # 清理临时文件和目录
    print("\n清理临时测试文件...")
    os.remove(mock_config.dataset_params['train_list'])
    os.remove(mock_config.dataset_params['eval_list'])
    os.remove(mock_config.dataset_params['infer_list'])
    for class_name, img_names in sample_images_info.items():
        for img_name in img_names:
            try:
                os.remove(os.path.join(base_data_dir, class_name, img_name))
            except OSError:
                pass # 可能文件已被其他方式删除
        try:
            os.rmdir(os.path.join(base_data_dir, class_name))
        except OSError:
            pass
    try:
        os.rmdir(base_data_dir)
    except OSError:
        pass
    print("MyReader.py 测试完成。") 