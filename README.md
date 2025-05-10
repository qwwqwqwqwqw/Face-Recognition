# 基于PaddlePaddle的人脸对比和人脸识别系统

> **重要提示**: 训练和测试时的图像大小**必须保持一致**！模型加载时会优先使用模型文件中保存的图像大小。

本项目基于PaddlePaddle框架实现了人脸对比和人脸识别功能，支持VGG和ResNet（使用新版PaddlePaddle API）作为特征提取器，并集成了ArcFace Loss以提升识别性能。项目参数通过YAML文件集中管理，同时支持通过命令行进行关键参数的指定和覆盖。

## 📑 目录

- [项目结构](#项目结构)
- [功能特点](#功能特点)
- [配置管理](#配置管理)
  - [YAML配置文件概览 (`configs/default_config.yaml`)](#yaml配置文件概览-configsdefault_configyaml)
  - [命令行参数与覆盖规则](#命令行参数与覆盖规则)
  - [配置加载工具 (`config_utils.py`)](#配置加载工具-config_utilspy)
- [详细使用指南](#详细使用指南)
  - [准备数据](#1-准备数据)
  - [创建数据列表](#2-创建数据列表)
  - [模型训练流程](#3-模型训练流程)
    - [初始训练](#初始训练)
    - [继续训练（模型微调）](#继续训练模型微调)
    - [训练输出示例](#训练输出示例)
  - [创建人脸特征库 (针对ArcFace模型)](#4-创建人脸特征库-针对arcface模型)
    - [ArcFace特征库原理](#arcface特征库原理)
    - [命令与输出示例](#命令与输出示例)
  - [人脸识别测试](#5-人脸识别测试)
    - [命令与输出示例](#命令与输出示例-1)
    - [可视化结果说明](#可视化结果说明)
  - [人脸对比测试](#6-人脸对比测试)
    - [命令与输出示例](#命令与输出示例-2)
    - [可视化结果说明](#可视化结果说明-1)
  - [更换数据集流程](#7-更换数据集流程)
- [模型调优指南](#模型调优指南)
- [常见问题解决](#常见问题解决)
- [参数调整效果分析](#参数调整效果分析)
- [技术实现](#技术实现)
- [注意事项](#注意事项)
- [项目提升和优化目标](#项目提升和优化目标)

## ⚠️ 重要环境准备

在开始之前，请确保您已完成以下环境设置：

1.  **进入项目根目录**: 所有后续命令都应在克隆本仓库后的 `Face-Recognition` 目录下执行。
    ```bash
    cd path/to/Face-Recognition
    ```
2.  **激活Python虚拟环境**: 本项目推荐使用名为 `paddle_env` 的虚拟环境。如果尚未创建，请先创建它。
    ```bash
    # 如果是第一次，创建虚拟环境 (python3 -m venv paddle_env)
    source paddle_env/bin/activate
    ```
    **提示**: 你可以将以下代码块添加到你的 `~/.bashrc` (或 `~/.zshrc` 等) 文件末尾，以实现当你 `cd` 进入 `Face-Recognition` 目录时自动激活 `paddle_env` 虚拟环境，并在离开时自动停用：
    ```bash
    # Auto-activate/deactivate paddle_env for Face-Recognition project
    auto_activate_paddle_env() {
        if [ -d "paddle_env" ] && [ -f "paddle_env/bin/activate" ]; then
            if [[ "$PWD" == *"/Face-Recognition"* ]] && [[ "$VIRTUAL_ENV" != "$PWD/paddle_env" ]]; then
                echo "Activating paddle_env in $PWD..."
                source "paddle_env/bin/activate"
            elif [[ "$VIRTUAL_ENV" == "$PWD/paddle_env" ]] && [[ "$PWD" != *"/Face-Recognition"* ]]; then
                echo "Deactivating paddle_env..."
                deactivate
            fi
        # If navigating out of a subdir of Face-Recognition but still within project, and env is active, keep it.
        elif [[ "$VIRTUAL_ENV" == */Face-Recognition/paddle_env* ]] && [[ "$PWD" != *"/Face-Recognition"* ]]; then
             # Check if we are in a parent directory that is NOT Face-Recognition
            if [[ "$VIRTUAL_ENV" != "$PWD/paddle_env" ]]; then
                 # Check if we are not in a subdirectory of where the virtual env is defined
                if [[ "$PWD"* != "$(dirname "$VIRTUAL_ENV")"*  ]]; then
                    echo "Deactivating paddle_env as we left the project root..."
                    deactivate
                fi
            fi
        fi
    }
    # Run on every prompt
    export PROMPT_COMMAND="auto_activate_paddle_env;$PROMPT_COMMAND"
    # Initial check in case .bashrc is sourced while already in the directory
    auto_activate_paddle_env
    ```
    保存 `~/.bashrc` 后，运行 `source ~/.bashrc` 使其生效。

3.  **GPU用户环境变量设置 (重要)**: 如果您计划使用GPU进行训练或推理，请在您的终端会话中设置以下环境变量。将这些命令添加到您的 `~/.bashrc` 或 `~/.zshrc` 文件中可以使其永久生效。
    ```bash
    # 根据您的CUDA实际安装路径调整
    export CUDA_HOME=/usr/local/cuda
    # 对于WSL用户，可能需要包含 /usr/lib/wsl/lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:/usr/lib/wsl/lib
    ```
    运行 `source ~/.bashrc` (或对应的shell配置文件) 使更改生效。

4.  **安装依赖**:\
    本项目使用 `tqdm` 库显示进度条，请确保安装。
    对于GPU版本 (推荐，示例为CUDA 11.8，请根据你的CUDA版本查找对应安装命令):\
    ```bash
    # 确保 paddle_env 虚拟环境已激活
    python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
    # 安装固定版本的 numpy (非常重要，防止 ABI 冲突)
    pip uninstall numpy -y
    pip install numpy==1.26.4
    # 安装其他依赖
    pip install PyYAML opencv-python==4.5.5.64 matplotlib==3.5.3 scikit-learn tqdm
    ```
    对于CPU版本:\
    ```bash
    # 确保 paddle_env 虚拟环境已激活
    pip install paddlepaddle==2.4.2 # CPU 版本可能也需要调整 numpy
    pip uninstall numpy -y
    pip install numpy==1.26.4
    pip install PyYAML opencv-python==4.5.5.64 matplotlib==3.5.3 scikit-learn tqdm
    ```
    > **兼容性提示**: 如果遇到 `numpy` ABI 错误或与PaddlePaddle版本不兼容，请查阅PaddlePaddle官方文档获取推荐的 `numpy` 版本，或尝试让 `pip` 自动解析依赖后，若仍有问题再指定版本。

## 项目结构

```
Face-Recognition/
├── configs/                  # 配置目录
│   └── default_config.yaml   # 默认YAML配置文件
├── data/                     # 数据目录
│   └── face/                 # 人脸数据集 (示例)
├── model/                    # 模型保存目录 (默认，可在配置文件中修改 model_save_dir)
├── results/                  # 推理和对比结果图片保存目录
├── CreateDataList.py         # 创建数据列表脚本
├── MyReader.py               # 图像读取和预处理模块
├── config_utils.py           # 配置加载与合并工具模块
├── vgg.py                    # VGG网络定义
├── resnet_new.py             # ResNet骨干网络及ArcFaceHead定义
├── train.py                  # 模型训练脚本
├── create_face_library.py    # 创建人脸特征库脚本 (用于ArcFace模型推理)
├── infer.py                  # 人脸识别预测脚本
├── face_compare.py           # 人脸对比工具脚本
└── README.md                 # 项目说明
```
*注：旧版 `resnet.py` 已被 `resnet_new.py` (新版API) 完全替代。ArcFace Loss 相关逻辑主要在 `resnet_new.py` (ArcFaceHead定义) 和 `train.py` (训练时调用) 中实现。*

## 功能特点

1.  **人脸对比**：比较两张人脸图片是否是同一个人，返回相似度得分。
2.  **人脸识别**：识别一张人脸图片属于哪个类别/人物。
    *   支持基于传统分类损失 (CrossEntropy) 的模型进行闭集识别。
    *   支持基于 ArcFace Loss 训练的模型，通过与预先计算的特征库进行比对，实现开集或闭集识别。
3.  **模型训练**：提供完整的模型训练、继续训练和微调功能。支持VGG+CrossEntropy, ResNet+CrossEntropy, ResNet+ArcFaceLoss。
4.  **参数配置管理**：使用YAML文件 (`configs/default_config.yaml`) 集中管理大部分参数，同时支持通过命令行指定关键参数和覆盖YAML配置。
5.  **数据增强**：支持多种数据增强方法提高模型泛化能力。
6.  **灵活的模型选择与配置**: 方便地切换骨干网络和损失函数。

## 配置管理

本项目采用YAML文件结合命令行参数的方式进行配置管理，以 `config_utils.py` 模块中的工具函数实现加载和合并。

### YAML配置文件概览 (`configs/default_config.yaml`)
主要的配置文件位于 `configs/default_config.yaml`。它包含了项目运行所需的绝大多数参数。以下是其结构概览和一个简化的示例：

```yaml
# configs/default_config.yaml (结构概览与部分示例)

# 通用设置
use_gpu: false        # 是否使用GPU (命令行可覆盖)
seed: 42              # 随机种子
image_size: 64        # 图像预处理统一尺寸

# 路径设置
data_dir: 'data'              # 数据集根目录
class_name: 'face'            # 当前使用的数据集子目录名
model_save_dir: 'model'       # 模型保存目录
# face_library_path: 'model/face_library.pkl' # (用于infer.py) 特征库路径
# label_file: 'data/face/readme.json'        # (用于infer.py) 标签文件路径

# 模型类型和结构
model_type: 'vgg'     # 可选: 'vgg', 'resnet'
num_classes: 5        # 数据集中的总类别数 (必须根据实际数据集修改!)

# ResNet 特定参数 (仅当 model_type='resnet' 时有效)
feature_dim: 512
nf: 32
n_resnet_blocks: 3

# ArcFace Loss 相关参数 (仅当 model_type='resnet' 且 use_arcface=true 时有效)
use_arcface: false
arcface_m1: 1.0
arcface_m2: 0.5
arcface_m3: 0.0
arcface_s: 64.0

# 训练超参数
batch_size: 32
epochs: 50
learning_rate: 0.001
log_interval: 10      # 每多少个batch打印一次日志
resume: false         # 是否从检查点恢复训练 (命令行可覆盖)

# 推理与对比相关参数
# infer.py
# recognition_threshold: 0.5 # ArcFace识别阈值
# infer_visualize: false     # 推理时是否可视化结果

# face_compare.py
# compare_threshold: 0.8     # 人脸对比相似度阈值
# compare_visualize: false   # 对比时是否可视化结果

# create_face_library.py
# output_library_path: 'model/face_library_default.pkl' # 特征库输出路径
# model_path: '' # 此脚本通常需要命令行指定模型路径
# data_list_file: '' # 此脚本通常需要命令行指定数据列表

# ... 其他脚本可能需要的参数，建议在使用时查阅或添加到各自的配置中 ...
```
**强烈建议**：用户根据自己的需求复制 `default_config.yaml` 并重命名（例如 `my_experiment.yaml`），然后修改其中的参数值，再通过 `--config_path my_experiment.yaml` 来加载。

### 命令行参数与覆盖规则
-   **核心参数**:
    -   `--config_path <路径>`: 指定要加载的YAML配置文件路径。若不提供，各脚本会尝试加载其内部定义的默认路径（通常是 `configs/default_config.yaml`）。
    -   `--use_gpu`: (布尔型开关) 是否使用GPU。
    -   `--resume`: (布尔型开关, 仅 `train.py`) 是否从检查点恢复训练。
    -   特定脚本的**必需输入**：如 `train.py` 通常不需要额外的必需命令行参数（因为它依赖YAML），而 `infer.py` 需要 `--model_path` 和 `--image_path`；`face_compare.py` 需要 `--model_path`, `--img1`, `--img2`；`create_face_library.py` 需要 `--model_path` 和 `--data_list_file`。这些通常通过命令行提供，因为它们是单次运行的特定输入。
-   **覆盖参数**: YAML中定义的大多数参数可以通过命令行提供同名参数来进行覆盖。
    -   **命名对应**: 命令行参数的名称通常直接对应于YAML文件中的键名。例如，YAML中的 `learning_rate: 0.001` 可以通过命令行 `--learning_rate 0.0005` 来覆盖。
    -   **布尔型参数**: 对于YAML中的布尔值（如 `use_arcface: false`），命令行可以使用 `action='store_true'` 或 `action='store_false'` (对于 `BooleanOptionalAction` 则是 `--use_arcface` / `--no-use_arcface`)。本项目中，`--use_gpu` 和 `--resume` 是简单的 `action='store_true'` 开关，如果出现则为True，否则依赖YAML。对于 `use_arcface` (在`train.py`中) 和 可视化选项 (如 `infer_visualize`) 使用了 `action=argparse.BooleanOptionalAction` (Python 3.9+), 这意味着命令行使用 `--use_arcface` 会设为True, `--no-use_arcface` 会设为False，若不提供则遵循YAML。
-   **优先级**: 命令行参数的优先级高于YAML文件中的参数。

### 配置加载工具 (`config_utils.py`)
该模块包含一个核心函数 `load_config`，其主要职责是：
1.  **加载默认或指定的YAML配置文件**：首先尝试从用户通过 `--config_path` 指定的路径加载YAML文件。如果未指定，则加载一个预设的默认路径。
2.  **解析命令行参数**：使用 `argparse` 解析用户在命令行提供的参数。
3.  **合并配置**：将从YAML文件加载的配置与命令行参数进行合并。对于在两处都定义的参数，命令行中的值将覆盖YAML中的值。
4.  **返回配置对象**：返回一个类似字典的配置对象 (`ConfigObject`)，该对象支持通过属性点号 (e.g., `config.learning_rate`) 和字典键 (e.g., `config['learning_rate']`) 两种方式访问配置项。

所有使用配置的脚本（`train.py`, `infer.py`, `create_face_library.py`, `face_compare.py`）都会在开始时调用 `load_config` 来获取最终生效的配置。

## 详细使用指南

### 1. 准备数据
将人脸数据放在YAML配置中 `data_dir` 和 `class_name` 指定的目录下（默认为 `data/face`），每个人的照片放在单独的子文件夹中：

```
data/face/  (或 config.data_dir / config.class_name)
├── person1/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── person2/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── ...
```
**注意事项**：
- 每个人的照片建议不少于20张，角度和光照多样化。
- 图片格式支持jpg、jpeg、png。
- 建议预先对人脸进行检测和裁剪，保证人脸居中。

### 2. 创建数据列表
此脚本会遍历指定的数据集根目录，为其中的图片生成训练和测试列表文件 (`trainer.list`, `test.list`) 以及一个元数据文件 (`readme.json`)。

    ```bash
# 为默认数据集 (data/face) 生成列表
# (如果存在旧文件，先手动删除)
    # rm -f data/face/trainer.list data/face/test.list data/face/readme.json
    python CreateDataList.py data/face

# 为自定义数据集 (例如 data/my_faces) 生成列表
# python CreateDataList.py data/my_faces
    ```
    **重要**:\
*   在重新生成列表前，请**手动删除**目标目录下（如 `data/face` 或 `data/my_faces`）已存在的 `trainer.list`, `test.list`, 和 `readme.json` 文件。
*   `CreateDataList.py` 脚本的参数是**包含所有人物子文件夹的数据集根目录**。

### 3. 模型训练流程
通过 `train.py` 训练模型。参数主要由YAML文件控制，但可以通过命令行覆盖。

**模型文件说明**:
-   **检查点 (Checkpoint)**: 训练过程中，每个epoch结束后会保存一个检查点文件到配置文件中 `model_save_dir` 指定的目录，命名为 `checkpoint_<model_type>.pdparams` (例如 `checkpoint_resnet.pdparams`)。此文件包含模型权重、优化器状态、学习率调度器状态、当前epoch、最佳准确率以及训练时的完整配置。
-   **最佳模型 (Best Model)**: 如果当前epoch在测试集上的准确率优于之前所有epoch，则会将当前模型的状态（模型权重和训练配置）保存为最佳模型，文件名为 `face_model_<model_type>.pdparams` (例如 `face_model_resnet.pdparams`)，也位于 `model_save_dir`。

#### 初始训练
**通用命令结构**:
```bash
python train.py --config_path <您的配置文件.yaml> [--use_gpu]
```
-   确保 `<您的配置文件.yaml>` 中已正确设置 `model_type`, `num_classes`, `data_dir`, `class_name`, `image_size` 以及其他相关训练参数。
-   `--use_gpu` (可选): 如果希望使用GPU并已正确配置环境。

**示例：使用默认配置训练 (假设 default_config.yaml 已设好)**
```bash
# 确保 Face-Recognition 目录下，虚拟环境已激活，GPU环境变量已设置 (如需)
# 检查并修改 configs/default_config.yaml 中的 num_classes, model_type 等参数
python train.py --config_path configs/default_config.yaml --use_gpu
```

#### 继续训练（模型微调）
使用 `--resume` 参数从检查点 (`checkpoint_<model_type>.pdparams`) 继续训练。
```bash
python train.py --config_path <您的配置文件.yaml> --resume [--use_gpu] [--learning_rate <新学习率>] [--epochs <新总轮数>]
```
-   脚本会自动加载上次训练的检查点。
-   确保配置文件中的 `model_type` 和 `model_save_dir` 与要恢复的检查点一致。

#### 训练输出示例
训练过程中，终端会定期打印日志，类似如下 (内容和格式可能略有不同):
```
使用 GPU 进行训练
开始训练前的最终配置确认 (来自YAML并由命令行更新后):
模型类型: resnet, GPU使用: True, 恢复训练: False
学习率: 0.001, Epochs: 50, 批大小: 32
ResNet参数: 特征维度=512, nf=32, n_blocks=3
  使用 ArcFace Loss, 参数: m1=1.0, m2=0.5, m3=0.0, s=64.0
开始训练，总共 50 个 epochs...
--- Epoch 1/50 ---
  Batch 0/15, Loss: 3.4512, Train Acc: 0.0938, LR: 0.001000
  Batch 10/15, Loss: 2.8765, Train Acc: 0.1875, LR: 0.000998
...
Epoch 1 Training Summary: Avg Loss: 2.9876, Avg Acc: 0.2150
Epoch 1 Test Summary: Accuracy: 0.2500
检查点已保存到: model/checkpoint_resnet.pdparams
最佳模型已更新并保存到: model/face_model_resnet.pdparams (Epoch 1, Accuracy: 0.2500)
--- Epoch 2/50 ---
...
```

### 4. 创建人脸特征库 (针对ArcFace模型)
如果模型使用 ArcFace Loss 训练 (即配置文件中 `model_type: 'resnet'` 且 `use_arcface: true`)，你需要为已知的身份创建一个特征库。后续 `infer.py` 在进行识别时会使用这个库进行1:N比对。

#### ArcFace特征库原理
ArcFace Loss旨在学习具有"角度区分性"的人脸特征，即同一身份的人脸特征在角度空间中更聚集，不同身份的人脸特征在角度空间中更分散。
1.  **特征提取**：对于每张已知身份的图像，使用训练好的ResNet骨干网络提取其高维特征向量。
2.  **特征聚合**：对于每个身份（标签），将其所有图像的特征向量进行平均（或其他聚合方式），得到该身份的一个代表性特征向量。
3.  **特征库构建**：将所有身份的代表性特征向量及其对应的标签存储起来，形成人脸特征库。这个库通常保存为一个pickle文件 (`.pkl`)。
在推理时，对待识别人脸提取特征后，将其与库中所有身份的特征向量计算余弦相似度。相似度最高的那个身份，如果超过设定的阈值，则认为是识别结果。

#### 命令与输出示例
```bash
python create_face_library.py \
    --config_path <您的配置文件.yaml> \
    --model_path <训练好的ArcFace模型.pdparams路径> \
    --data_list_file <数据集列表文件路径, 如data/face/trainer.list> \
    [--use_gpu]
```
-   `--model_path`: **必需**，指向训练好的、包含ArcFace的模型文件 (例如 `model/face_model_resnet.pdparams`)。
-   `--data_list_file`: **必需**，包含图像路径和标签的数据列表文件。
-   `output_library_path` (特征库保存路径) 和 `image_size` 等参数会从配置文件加载或命令行覆盖。脚本优先使用模型文件中保存的 `image_size`。

**终端输出示例**:
```
使用 GPU 进行特征提取
从 model/face_model_resnet.pdparams 加载模型...
ResNet 模型骨干加载成功。
开始从 480 张图像中提取特征...
提取特征: 100%|██████████| 480/480 [00:10<00:00, 47.50it/s]
计算每个类别的平均特征向量...
计算平均特征: 100%|██████████| 10/10 [00:00<00:00, 1520.23it/s]
创建输出目录: model_libs
人脸特征库已成功保存到: model_libs/face_library_arcface_custom.pkl
库中包含 10 个身份的特征。
```

### 5. 人脸识别测试
使用 `infer.py` 进行人脸识别。
```bash
python infer.py \
    --config_path <您的配置文件.yaml> \
    --model_path <训练好的模型.pdparams路径> \
    --image_path <您的测试图片路径.jpg> \
    [--use_gpu]
```
-   `--model_path`: **必需**。
-   `--image_path`: **必需**。
-   其他参数如 `label_file` (用于非ArcFace模型), `face_library_path` (用于ArcFace模型), `recognition_threshold`, `infer_visualize` 会从配置文件加载或命令行覆盖。

#### 命令与输出示例
**对于CrossEntropy模型**:
```
使用 CPU 进行推理
从 model/face_model_vgg.pdparams 加载模型...
使用 VGG 模型进行推理，分类数量: 10
标签文件 data/face/readme.json 加载成功。
开始推理图像: data/face/person_A/test_img.jpg
预测的人脸类别 (基于分类): Person_A, 置信度: 0.9876
结果图像已保存至: results/recognition_vgg_test_img.jpg
```

**对于ArcFace模型**:
```
使用 GPU 进行推理
从 model/face_model_resnet.pdparams 加载模型...
使用 ResNet 模型骨干进行推理，特征维度: 512
  L--> 加载 ArcFaceHead，分类数量: 10
人脸特征库 model_libs/face_library_arcface_custom.pkl 加载成功，包含 10 个已知身份。
开始推理图像: data/face/person_B/another_img.jpg
ArcFace 特征比对完成。
预测的人脸类别 (基于特征库): Person_B, 最高相似度: 0.8567
结果图像已保存至: results/recognition_resnet_another_img.jpg
```
如果相似度低于阈值，可能会输出 "未知人物"。

#### 可视化结果说明
如果配置文件中的 `infer_visualize` 或命令行的 `--infer_visualize` (或 `--no-infer_visualize`) 被设置为启用可视化，脚本会将识别结果（预测的类别名和置信度/相似度）标注在输入图像上，并保存到 `results/` 目录下。文件名通常包含模型类型和原图文件名。

### 6. 人脸对比测试
使用 `face_compare.py` 对比两张人脸图像。
```bash
python face_compare.py \
    --config_path <您的配置文件.yaml> \
    --model_path <训练好的模型.pdparams路径> \
    --img1 <第一张人脸图像路径.jpg> \
    --img2 <第二张人脸图像路径.jpg> \
    [--use_gpu]
```
-   `--model_path`: **必需**。
-   `--img1`, `--img2`: **必需**。
-   参数如 `compare_threshold`, `compare_visualize` 会从配置文件加载或命令行覆盖。

#### 命令与输出示例
```
使用 CPU 进行人脸对比
从 model/face_model_resnet.pdparams 加载模型...
使用 ResNet 模型骨干进行对比 (feat_dim=512)
开始对比图像:
1. data/face/person_A/img1.jpg
2. data/face/person_A/img2.jpg
图像1: data/face/person_A/img1.jpg
图像2: data/face/person_A/img2.jpg
使用模型类型: resnet
计算得到的相似度: 0.9123
判断结果 (阈值 0.8): 是同一个人
对比结果图像已保存至: results/compare_resnet_img1_vs_img2.png
```

#### 可视化结果说明
如果配置文件中的 `compare_visualize` 或命令行的 `--compare_visualize` (或 `--no-compare_visualize`) 被设置为启用可视化，脚本会并排显示两张输入图像，并在标题处标注计算出的相似度、判断结果（是否为同一个人）和使用的阈值。图像会保存到 `results/` 目录。

### 7. 更换数据集流程
1.  **准备新数据集**: 同原有说明，创建新数据目录并按结构组织图像。
2.  **创建数据列表**: `python CreateDataList.py data/new_dataset_name`
3.  **配置YAML文件**:
    *   复制 `configs/default_config.yaml` 为例如 `configs/new_dataset_config.yaml`。
    *   修改 `configs/new_dataset_config.yaml` 中的参数以适应新数据集：
        *   `data_dir`: 指向新数据集的父目录 (例如 `data`)
        *   `class_name`: 新数据集的名称 (例如 `new_dataset_name`)
        *   `num_classes`: **必须更新为新数据集的实际类别数**。
        *   `model_save_dir`: 为新模型指定一个新的保存目录 (例如 `model_new_dataset`)。
        *   根据需要调整其他训练参数。
4.  **开始新数据集训练**:
    ```bash
    python train.py --config_path configs/new_dataset_config.yaml [--use_gpu]
    ```
    如果需要迁移学习 (加载旧模型的骨干权重)，目前脚本未直接支持此功能，需要手动修改`train.py`的权重加载逻辑，或者确保预训练模型的骨干网络与新任务兼容。

## 模型调优指南
为了提高模型准确率，您可以主要通过修改YAML配置文件中的参数，或通过命令行临时覆盖来进行实验。

**示例：调整学习率**
1.  修改您的配置文件 (例如 `configs/my_experiment_config.yaml`) 中的 `learning_rate` 值。
2.  运行训练: `python train.py --config_path configs/my_experiment_config.yaml`
3.  或者，临时覆盖: `python train.py --config_path <基础配置文件> --learning_rate 0.0005`

其他可调参数（如批大小、轮数、ArcFace参数、ResNet结构参数等）也遵循类似逻辑。请参考 `configs/default_config.yaml` 中各参数的说明。

## 常见问题解决
(基本与原版一致，但注意参数调整现在主要通过YAML或特定命令行覆盖)

### 1. 识别准确率低
**可能原因**：
- 训练数据不足或质量低。
- 模型参数不合适（学习率、模型类型、ArcFace超参数等选择不当）。
- 过拟合或欠拟合。
- 对于ArcFace模型：特征库未正确生成，或识别阈值不当。
**解决方案**：
- 增加训练数据和数据多样性。
- 调整学习率、ArcFace超参数（`s`, `m2`），ResNet结构参数。
- 对于ArcFace模型：
    - 确保使用 `create_face_library.py` 生成了覆盖所有已知身份的特征库。
    - 在 `infer.py` 中调整 `--recognition_threshold`。
- 对于分类模型：检查类别是否平衡，损失函数是否合适。

### 2. 训练不收敛
**可能原因**：
- 学习率设置不当（过大或过小）。
- 数据预处理问题。
- 模型结构复杂度与数据集不匹配。
- ArcFace参数设置极端导致梯度消失或爆炸。
**解决方案**：
- 尝试更小的学习率。
- 检查数据集质量和预处理步骤。
- 对于ArcFace，尝试更保守的 `s` 和 `m` 值。
- 简化模型（例如，减少ResNet的深度或通道数）或增加正则化。

### 3. 模型过拟合
**可能原因**：
- 训练数据太少。
- 模型太复杂。
- 正则化不足。
**解决方案**：
- 增加训练数据或使用更多数据增强。
- 增加权重衰减 (`weight_decay` في `train.py` 的优化器设置中)。
- 提前停止训练。
- 对于ResNet，可以尝试更浅的网络。

### 4. 对比模式不准确 (`face_compare.py`)
**可能原因**：
- 阈值设置不当 (`--threshold`)。
- 模型特征提取能力不足（模型未充分训练或不适合当前任务）。
- 测试图像质量问题。
**解决方案**：
- 调整相似度阈值：`python face_compare.py --model_load_path <模型路径> --threshold=0.75 ...`
- 继续训练模型提高特征提取能力，或尝试更适合的骨干网络/损失函数组合。
- 提高测试图像质量，确保人脸对齐、光照均匀。

### 5. 训练时内存不足
**可能原因**：
- 批大小 (`--batch_size`) 设置过大。
- 图像尺寸 (`--image_size`) 过大。
- 模型过于复杂（例如，ResNet的 `nf` 或 `n_resnet_blocks` 过大）。
**解决方案**：
- 减小批大小。
- 减小图像尺寸。
- 减小ResNet模型的 `nf` 或 `n_resnet_blocks`。
- 使用CPU训练（`--use_gpu False`），但会非常慢。

### 6. `infer.py` 报 "unknown" 或错误识别 (ArcFace模型)
**可能原因**：
- `--face_library_path` 未指定或路径错误。
- 特征库 (`.pkl` 文件) 与当前测试的模型不兼容（例如，用不同 `feature_dim` 训练的模型生成的库）。
- `--recognition_threshold` 太高，导致所有真实匹配都被拒绝；或太低，导致错误接受。
- 输入图像的人脸与库中人脸差异过大。
**解决方案**：
- 确保 `--face_library_path` 指向正确的、与当前模型兼容的特征库。
- 重新生成特征库：`python create_face_library.py --model_load_path <当前模型路径> ...`
- 调整 `--recognition_threshold`。
- 检查输入图像质量。

### 7. 图像大小不一致错误
**重要说明**：训练时使用的图像大小 (在YAML中配置的 `image_size`) 必须与后续使用该模型进行特征库生成、推理、对比时内部处理图像的大小一致！
**好消息**：当前脚本在加载模型文件 (`.pdparams`) 时，会优先使用模型文件中保存的训练时配置（包括 `image_size`）。这意味着如果您使用同一个训练好的模型进行后续操作，图像大小通常会自动保持一致。
**潜在问题**：
-   如果您在不同脚本的配置文件中为 `image_size` 设置了不同的值，并且某个脚本没有成功从模型文件中读取到 `image_size`（例如，模型文件非常旧，没有保存配置），则可能出现不一致。
-   手动修改了 `MyReader.py` 中的图像处理逻辑，或使用外部工具预处理图像，尺寸与模型期望不一致。

**最佳实践**：
-   主要在训练时的配置文件中设定好 `image_size`。
-   后续操作（推理、对比等）加载此模型时，脚本会尝试使用模型自带的 `image_size`。您可以在这些脚本的配置文件中也指定 `image_size`，它将作为模型文件未提供此信息时的备用值。

## 参数调整效果分析
(内容保持不变，但读者应理解参数调整主要通过修改YAML实现)
...

#### ArcFace 超参数 (`arcface_s`, `arcface_m1`, `arcface_m2`, `arcface_m3`)

| 参数 (YAML中/命令行) | 值范围建议 | 优点 | 缺点 | 适用场景 |
|-------------------|------------|------|------|----------|
| `arcface_s` (scale) | 16-64      | 较大的s使类间距离更大，决策边界更清晰 | 过大可能导致训练不稳定，梯度爆炸 | 需要强判别力的场景 |
| `arcface_m1` (mode) | 通常为1.0  | 控制ArcFace margin计算模式 | - | 标准ArcFace设置 |
| `arcface_m2` (margin) | 0.2-0.5    | 增加类内紧凑性和类间可分性 | 过大可能使模型难以收敛 | 大多数人脸识别任务 |
| `arcface_m3` (additive margin) | 通常为0.0  | 对余弦值进行偏移调整 | - | 标准ArcFace设置 |

## 技术实现
-   使用VGG和ResNet (基于PaddlePaddle新版API `paddle.nn`) 神经网络进行特征提取。
-   支持CrossEntropy Loss 和 ArcFace Loss (`paddle.nn.functional.margin_cross_entropy`)。
-   **参数化配置**：通过 `config_utils.py` 实现YAML文件和命令行参数的灵活管理实验配置。
-   使用PaddlePaddle 动态图模式。
-   支持多类别人脸识别和基于特征相似度的人脸对比。
-   实现了包含模型结构参数、权重、优化器状态的统一模型文件保存和加载机制，模型文件中也保存了训练时的配置。
-   提供了工具脚本 `create_face_library.py` 用于生成ArcFace模型所需的人脸特征库。

## 注意事项
1.  确保已正确安装PaddlePaddle及 `PyYAML`, `tqdm` 等相关依赖。
2.  训练前，请务必在配置文件中正确设置 `num_classes` 以匹配数据集。
3.  ... (其余内容不变)

## 项目提升和优化目标
(内容保持不变)
...

### 核心目标 (满足课程基础要求并显著提升) - 部分已完成

1.  **实现多种骨干网络**:
    *   ✅ 在新版 PaddlePaddle 中实现 VGG (作为基线，可运行)。
    *   ✅ 实现至少一种 ResNet 变体 (如 ResNetFace，基于 `resnet_new.py`)。
2.  **灵活选择骨干网络**:
    *   ✅ 修改训练 (`train.py`)、推理 (`infer.py`)、对比 (`face_compare.py`) 脚本，使其可以灵活选择 VGG 或 ResNet 变体作为骨干网络。
3.  **集成先进损失函数**:
    *   ✅ 集成 ArcFace Loss。
    *   ✅ 将 ArcFace Loss 应用于 ResNet 变体模型上进行训练。
4.  **充分的实验对比**:
    *   ⏳ *进行中*: 对比 VGG + CrossEntropyLoss 和 ResNet 变体 + ArcFace Loss 这两种组合的性能。
    *   ⏳ *进行中*: 通过调参，对这两种组合分别进行优化训练，并记录和分析结果。

### 可选进阶目标 (如果时间允许且精力充沛)

1.  **探索不同 ResNet 变体**:
    *   尝试不同的 ResNet 变体 (如 ResNet18 vs ResNet34 vs ResNet50) 与 ArcFace Loss 结合的性能。
2.  **尝试其他先进损失函数**:
    *   尝试另一种先进损失函数 (如 CosFace Loss) 与 ResNet 变体结合，并与 ArcFace Loss 进行对比。
3.  **高级数据增强**:
    *   探索更高级的数据增强策略。
4.  **更完善的评估**:
    *   在标准人脸数据集（如LFW）上评估模型性能，并计算标准指标（如Verification Accuracy, TAR@FAR）。

### 第三阶段：集成 ArcFace Loss 并进行联合调参 - 大部分完成

**目标：**

1.  ✅ 在项目中实现 ArcFace Loss (`resnet_new.py` 中的 `ArcFaceHead`)。
2.  ✅ 修改模型结构 (特别是 ResNet) 和训练脚本，使其能够使用 ArcFace Loss 进行训练 (`train.py`)。
3.  ✅ 修改推理 (`infer.py`) 和对比 (`face_compare.py`) 脚本，使其能够有效利用 ArcFace Loss 训练出的模型。
4.  ⏳ *进行中*: 结合 ResNet 模型结构和 ArcFace Loss 的特性，进行系统的调参优化，以获得最佳性能。

**主要实践步骤概述：**

1.  ✅ **理解与实现 ArcFace Loss**：已在 `resnet_new.py` 中使用 `paddle.nn.functional.margin_cross_entropy` 实现 `ArcFaceHead`。
2.  ✅ **调整 ResNetFace 模型**：`ResNetFace` 输出特征向量。
3.  ✅ **集成 ArcFace Loss 到训练流程 (`train.py`)**：已支持 `--use_arcface`，优化器包含 `head_module` 参数，损失计算和评估逻辑已更新。检查点保存加载已更新。
4.  ✅ **创建人脸库生成脚本 (`create_face_library.py`)**：已创建并实现。
5.  ✅ **更新推理与对比脚本**：
    *   `infer.py`：已修改为支持从模型文件加载配置，并可使用特征库进行 ArcFace 模型识别。
    *   `face_compare.py`：已修改为支持从模型文件加载配置，并使用骨干网络进行特征提取对比。
6.  ⏳ **进行中**: **进行联合调参和实验对比**：系统调整学习率、批大小、ArcFace Loss 的超参数等。对比 ResNet + ArcFace Loss 与 VGG + CrossEntropyLoss 的性能。记录并分析实验结果。 