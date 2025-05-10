# 基于PaddlePaddle的人脸对比和人脸识别系统

> **重要提示**: 训练和测试时的图像大小**必须保持一致**！[查看详细说明](#6-图像大小不一致错误)

本项目基于PaddlePaddle框架实现了人脸对比和人脸识别功能，使用VGG网络作为特征提取器。

## 📑 目录

- [项目结构](#项目结构)
- [功能特点](#功能特点)
- [详细使用指南](#详细使用指南)
  - [准备数据](#1-准备数据)
  - [创建数据列表](#2-创建数据列表)
  - [模型训练流程](#3-模型训练流程)
  - [人脸识别测试](#4-人脸识别测试)
  - [人脸对比测试](#5-人脸对比测试)
  - [更换数据集流程](#6-更换数据集流程)
- [模型调优指南](#模型调优指南)
- [常见问题解决](#常见问题解决)
- [参数调整效果分析](#参数调整效果分析)
- [技术实现](#技术实现)
- [注意事项](#注意事项)

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

4.  **安装依赖**:
    对于GPU版本 (推荐，示例为CUDA 11.8，请根据你的CUDA版本查找对应安装命令):
    ```bash
    # 确保 paddle_env 虚拟环境已激活
    python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
    # 安装固定版本的 numpy (非常重要，防止 ABI 冲突)
    pip uninstall numpy -y
    pip install numpy==1.26.4
    # 安装其他依赖
    pip install opencv-python==4.5.5.64 matplotlib==3.5.3
    ```
    对于CPU版本:
    ```bash
    # 确保 paddle_env 虚拟环境已激活
    pip install paddlepaddle==2.4.2 # CPU 版本可能也需要调整 numpy
    pip uninstall numpy -y
    pip install numpy==1.26.4 
    pip install opencv-python==4.5.5.64 matplotlib==3.5.3
    ```
    > **注意**: CPU 版本的 `paddlepaddle` 和 `numpy` 的兼容性也可能需要关注，上述组合是一个建议。如果遇到 `numpy` ABI 错误，请尝试调整 `numpy` 版本。

## 项目结构

```
Face-Recognition/
├── data/                  # 数据目录
│   └── face/              # 人脸数据集
├── model/                 # 模型保存目录
├── CreateDataList.py      # 创建数据列表
├── MyReader.py            # 图像读取和预处理
├── vgg.py                 # VGG网络定义
├── resnet.py              # ResNet网络定义
├── train.py               # 模型训练
├── continue_train.py      # 继续训练模型
├── infer.py               # 人脸识别预测
├── face_compare.py        # 人脸对比工具
└── README.md              # 项目说明
```

## 功能特点

1. **人脸对比**：比较两张人脸图片是否是同一个人，返回相似度得分。
2. **人脸识别**：识别一张人脸图片属于哪个类别/人物。
3. **模型训练**：提供完整的模型训练、继续训练和微调功能。
4. **数据增强**：支持多种数据增强方法提高模型泛化能力。

## 详细使用指南

### 1. 准备数据

将人脸数据放在`data/face`目录下，每个人的照片放在单独的子文件夹中：

```
data/face/
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
- 每个人的照片建议不少于20张，角度和光照多样化
- 图片格式支持jpg、jpeg、png
- 建议预先对人脸进行检测和裁剪，保证人脸居中

### 2. 创建数据列表

此脚本会遍历指定的数据集目录，为其中的图片生成训练和测试列表文件 (`trainer.list`, `test.list`) 以及一个元数据文件 (`readme.json`)。

*   **为默认数据集生成列表**:
    ```bash
    # （手动操作）如果存在旧的列表文件，请先删除
    # sudo rm -f data/face/trainer.list data/face/test.list data/face/readme.json
    python CreateDataList.py data/face 
    ```
*   **为自定义数据集生成列表** (假设自定义数据在 `data/my_faces`):
    ```bash
    # （手动操作）如果存在旧的列表文件，请先删除
    # sudo rm -f data/my_faces/trainer.list data/my_faces/test.list data/my_faces/readme.json
    python CreateDataList.py data/my_faces
    ```
    **重要**:
    *   在重新生成列表前，请**手动删除**目标目录下（如 `data/face` 或 `data/my_faces`）已存在的 `trainer.list`, `test.list`, 和 `readme.json` 文件。如果这些文件由`root`用户创建，你可能需要使用 `sudo rm`。
    *   `CreateDataList.py` 脚本的参数是**包含所有人物子文件夹的数据集根目录** (例如 `data/face`，而不是 `data/face/dilireba`)。

### 3. 模型训练流程

#### 3.1 初始训练 [🔗](#模型调优指南)

运行以下命令开始训练 (默认使用 `data/face` 目录下的数据):

```bash
# 确保你在 Face-Recognition 目录下，虚拟环境已激活，GPU环境变量已设置
python train.py
```

可以通过参数调整训练配置：

```bash
# 指定类别数量, epochs, 和学习率
python train.py --num_classes=5 --epochs=10 --learning_rate=0.001

# 使用GPU训练 (脚本内默认尝试使用GPU，此参数可显式控制)
python train.py --use_gpu=True

# 调整批大小和图像大小
python train.py --batch_size=32 --image_size=64


python train.py --use_gpu=True --num_classes=5 --epochs=10 --learning_rate=0.001 --batch_size=32 --image_size=64
```
**训练过程说明**：
- 每个epoch会显示训练损失和准确率
- 每个epoch会在测试集上评估模型性能
- 训练会自动保存以下文件：
  - `model/face_model.pdparams`：最佳模型权重（用于推理，此为默认路径）
  - `model/checkpoint.pdparams`：完整检查点（用于继续训练）

#### 3.2 继续训练（模型微调） [🔗](#模型调优指南)

当模型需要继续训练或调整参数时，使用以下命令：

```bash
# 基本继续训练命令
python continue_train.py --epochs=80

# 调整学习率继续训练
python continue_train.py --learning_rate=0.0001 --epochs=80

# 使用GPU继续训练
python continue_train.py --use_gpu=True
```

**继续训练机制**：
- 自动加载上次训练的检查点（包括模型权重、优化器状态等）
- 从上次训练结束的epoch继续训练
- 保持训练状态连续性（如动量、学习率等）
- 自动更新检查点和最佳模型

**注意事项**：
- 继续训练前必须已有由`train.py`创建的检查点文件
- 调整学习率通常有助于模型突破性能瓶颈

### 4. 人脸识别测试

运行以下命令进行人脸识别：

```bash
# 基本测试命令
python infer.py --image_path=<图像路径>

# 指定模型和标签文件
python infer.py --image_path=<图像路径> \
                                 --model_path=model/face_model.pdparams \
                                 --label_file=data/face/readme.json

# 使用GPU推理
python infer.py --image_path=<图像路径> --use_gpu=True

# 可视化结果
python infer.py --image_path=<图像路径> --visualize=True
```

**输出说明**：
- 预测的人脸类别和置信度
- 可视化结果（如启用）会保存在`results`目录

示例输出：
```
预测的人脸类别: 张三, 置信度: 0.9876
```

### 5. 人脸对比测试

运行以下命令进行人脸对比：

```bash
# 基本对比命令
python face_compare.py --img1=<第一张人脸图像路径> --img2=<第二张人脸图像路径>

# 指定阈值和模型
python face_compare.py --img1=<图片1路径> --img2=<图片2路径> \
                                       --threshold=0.75 \
                                       --model_path=model/face_model.pdparams

# 使用GPU进行对比
python face_compare.py --img1=<图片1路径> --img2=<图片2路径> --use_gpu=True
```

示例：
```bash
python face_compare.py --img1=data/face/dilireba/159116043650.jpg --img2=data/face/dilireba/159116043610.jpg
```

**参数说明**：
- `--threshold`：设置判断为同一人的阈值，默认0.8
- `--visualize`：是否可视化对比结果，默认True
- `--use_gpu`：是否使用GPU，默认False

**输出说明**：
- 两张图片的相似度得分
- 是否为同一人的判断结果
- 可视化对比结果保存在`results`目录

### 6. 更换数据集流程

当需要训练新的人脸数据集时，请按以下步骤操作：

#### 6.1 备份当前模型

```bash
# 创建备份目录
mkdir -p model/backup

# 备份模型文件
cp model/face_model.pdparams model/backup/face_model_backup.pdparams
cp model/checkpoint.pdparams model/backup/checkpoint_backup.pdparams
```

#### 6.2 准备新数据集

```bash
# 按同样结构组织新数据集
# data/new_face/
# ├── person1/
# │   ├── 1.jpg...
# └── ...

# 创建数据列表
python CreateDataList.py data/new_face
```

#### 6.3 开始新数据集训练

```bash
# 从头训练新数据集
python train.py --data_dir=data \
                                --class_name=new_face \
                                --num_classes=<新类别数量>

# 或使用迁移学习（加载旧模型权重开始训练）
python train.py --data_dir=data \
                                --class_name=new_face \
                                --num_classes=<新类别数量> \
                                --resume=True
```

## 模型调优指南 [🔗](#31-初始训练)

为了提高人脸识别模型的准确率，您可以尝试以下参数微调策略：

### 1. 学习率调整

学习率是最重要的超参数之一，直接影响模型的收敛速度和最终性能：

```bash
# 降低学习率以获得更精确的结果
python train.py --learning_rate=0.0005

# 使用更大的学习率加速初期训练
python train.py --learning_rate=0.003

# 继续训练时使用较小学习率进行精调
python continue_train.py --learning_rate=0.0001
```

您还可以尝试使用学习率衰减策略，在train.py中我们使用了余弦退火策略，您可以调整T_max参数：

```python
# 建议修改train.py中的学习率调度器
lr_scheduler = optimizer.lr.CosineAnnealingDecay(
    learning_rate=args.learning_rate, 
    T_max=args.epochs,  # 可以尝试调整这个值
    eta_min=1e-6  # 最小学习率
)
```

### 2. 批大小调整

批大小会影响训练稳定性和优化效果：

```bash
# 小批量大小，更新更频繁，可能更好地捕捉细节特征
python train.py --batch_size=16

# 大批量大小，训练更稳定，需要相应调整学习率
python train.py --batch_size=64 --learning_rate=0.004
```

### 3. 训练轮数增加

增加训练轮数可以让模型更充分学习：

```bash
# 增加训练轮数
python train.py --epochs=100

# 继续训练时延长训练轮数
python continue_train.py --epochs=150
```

### 4. 正则化调整

可以调整模型中的dropout率来改变正则化强度：

```bash
# 在vgg.py中可以修改dropout_rate参数
model = VGGFace(num_classes=args.num_classes, dropout_rate=0.3)  # 降低dropout率
model = VGGFace(num_classes=args.num_classes, dropout_rate=0.7)  # 增加dropout率
```

### 5. 数据增强策略

在MyReader.py中已经实现了基本的数据增强，您可以尝试添加更多增强方法，如：

```python
# 在MyReader.py的__getitem__方法中添加更多数据增强
# 添加高斯噪声
if random.random() > 0.5:
    noise = np.random.normal(0, 0.05, img.shape).astype('float32')
    img = img + noise
    img = np.clip(img, 0, 1)

# 添加随机裁剪和缩放
if random.random() > 0.5:
    # 实现随机裁剪和缩放逻辑
    pass
```

### 6. 优化器调整

尝试不同的优化器：

```python
# 在train.py中可以尝试不同的优化器
# Adam优化器
opt = optimizer.Adam(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=1e-4
)

# AdamW优化器
opt = optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=1e-2
)
```

### 7. 网络结构调整

可以尝试调整VGG网络结构：

```python
# 在VGGFace类中添加批归一化层
self.bn1_1 = nn.BatchNorm2D(64)
self.bn1_2 = nn.BatchNorm2D(64)
# 在forward方法中应用
x = F.relu(self.bn1_1(self.conv1_1(x)))
x = F.relu(self.bn1_2(self.conv1_2(x)))
```

或尝试不同的特征提取网络，如ResNet:

```bash
# 如果实现了ResNet，可以使用这种方式切换网络
python train.py --backbone=resnet
```

### 8. 使用预训练权重

如果可用，使用在大规模数据集上预训练的权重：

```python
# 加载预训练权重
model = VGGFace(num_classes=args.num_classes)
state_dict = paddle.load('pretrained_vgg.pdparams')
model.set_state_dict(state_dict)
```

### 9. 集成多个模型

训练多个模型并集成预测结果：

```python
# 训练多个模型
python train.py --seed=42 --model_path=model/face_model1.pdparams
python train.py --seed=43 --model_path=model/face_model2.pdparams
python train.py --seed=44 --model_path=model/face_model3.pdparams

# 集成预测时加载多个模型并平均预测结果
```

### 10. 评估和超参数搜索

使用交叉验证或网格搜索寻找最佳参数组合：

```bash
# 可以编写脚本尝试不同的参数组合
for lr in 0.0001 0.0005 0.001 0.005; do
    for bs in 16 32 64; do
        python train.py --learning_rate=$lr --batch_size=$bs
    done
done
```

## 常见问题解决

### 1. 识别准确率低

**可能原因**：
- 训练数据不足或质量低
- 模型参数不合适
- 过拟合或欠拟合

**解决方案**：
- 增加训练数据和数据多样性
- 降低学习率继续训练：`python continue_train.py --learning_rate=0.0001`
- 增加数据增强策略
- 调整dropout率：尝试0.3-0.7之间的不同值

### 2. 训练不收敛

**可能原因**：
- 学习率设置不当
- 数据预处理问题
- 模型结构复杂度与数据集不匹配

**解决方案**：
- 尝试更小的学习率：`python train.py --learning_rate=0.0001`
- 检查数据集质量和预处理步骤
- 简化模型或增加正则化

### 3. 模型过拟合

**可能原因**：
- 训练数据太少
- 模型太复杂
- 正则化不足

**解决方案**：
- 增加训练数据或使用更多数据增强
- 增加dropout率：`model = VGGFace(num_classes=args.num_classes, dropout_rate=0.7)`
- 增加权重衰减：修改优化器中的`weight_decay`参数
- 提前停止训练

### 4. 对比模式不准确

**可能原因**：
- 阈值设置不当
- 模型特征提取能力不足
- 测试图像质量问题

**解决方案**：
- 调整相似度阈值：`python face_compare.py --threshold=0.75`
- 继续训练模型提高特征提取能力
- 提高测试图像质量，确保人脸对齐、光照均匀

### 5. 训练时内存不足

**可能原因**：
- 批大小设置过大
- 图像尺寸过大
- GPU内存不足

**解决方案**：
- 减小批大小：`python train.py --batch_size=8`
- 减小图像尺寸：`python train.py --image_size=48`
- 使用CPU训练：`python train.py --use_gpu=False`

### 6. 图像大小不一致错误

**重要说明**：训练和测试时的图像大小**必须保持一致**！

**问题描述**：
- 若训练时使用的图像大小与测试/推理时不同，将导致模型无法正常工作
- 常见错误信息包括形状不匹配、维度错误等

**原因**：
- 模型架构中的输入层、卷积层和全连接层都依赖固定的图像尺寸
- 特征提取和特征匹配需要在相同维度空间才有意义

**如何确保一致性**：
1. **记录训练参数**：
   ```bash
   # 训练时明确指定并记录图像大小
   python train.py --image_size=64
   ```

2. **推理时使用相同参数**：
   ```bash
   # 使用相同的图像大小进行推理/测试
   python infer.py --image_path=<图片路径> --image_size=64
   python face_compare.py --img1=<路径1> --img2=<路径2> --image_size=64
   ```

3. **检查点加载机制**：
   - 继续训练时，程序会自动从检查点读取原始图像大小参数
   - 但在`infer.py`和`face_compare.py`中需要手动确保一致

**最佳实践**：
- 项目默认值是64×64，建议除非有特殊需求，否则保持此默认值
- 在切换不同模型时，先检查训练时使用的图像大小
- 在项目文档中记录每个模型的图像大小参数

## 参数调整效果分析

不同参数调整对模型准确率有不同影响，下面详细记录了各种参数调整的效果分析：

#### 学习率 (Learning Rate)

| 参数值 | 优点 | 缺点 | 适用场景 |
|--------|------|------|----------|
| 较大学习率 (0.003) | 训练初期收敛快 | 可能错过最优解，模型不稳定 | 训练初期，数据质量好的情况 |
| 较小学习率 (0.0005) | 能找到更精确的最优解 | 训练速度慢 | 模型精调阶段 |

#### 批大小 (Batch Size)

| 参数值 | 优点 | 缺点 | 适用场景 |
|--------|------|------|----------|
| 小批量 (16) | 更新更频繁，泛化性能可能更好 | 训练不稳定，耗时更长 | 数据集较小，显存受限 |
| 大批量 (64) | 训练更稳定，计算效率高 | 可能陷入局部最优 | 大数据集，有足够显存 |

#### Dropout率

| 参数值 | 优点 | 缺点 | 适用场景 |
|--------|------|------|----------|
| 高dropout率 (0.7) | 减少过拟合 | 训练时间更长，可能欠拟合 | 数据集较小，模型复杂 |
| 低dropout率 (0.3) | 保留更多特征信息 | 容易过拟合 | 数据集大，特征不足 |

#### 数据增强

| 增强强度 | 优点 | 缺点 | 适用场景 |
|----------|------|------|----------|
| 强数据增强 | 提高模型泛化能力 | 可能引入噪声 | 数据集小，场景单一 |
| 弱数据增强 | 保持原始特征 | 泛化能力可能不足 | 数据集大，场景丰富 |

#### 优化器选择

| 优化器 | 优点 | 缺点 | 适用场景 |
|--------|------|------|----------|
| Adam | 收敛快，适应性好 | 可能泛化性能不如SGD | 大多数情况 |
| AdamW | 权重衰减更合理 | 需要更细致的调参 | 大模型训练 |

#### 正则化强度

| 正则化强度 | 优点 | 缺点 | 适用场景 |
|------------|------|------|----------|
| 强正则化 | 减少过拟合 | 可能导致欠拟合 | 小数据集，简单任务 |
| 弱正则化 | 保留更多特征 | 容易过拟合 | 大数据集，复杂任务 |

#### 训练轮数

| 训练轮数 | 优点 | 缺点 | 适用场景 |
|----------|------|------|----------|
| 增加轮数 | 模型学习更充分 | 耗时长，可能过拟合 | 复杂任务，充足计算资源 |
| 减少轮数 | 训练快，节省资源 | 模型可能欠拟合 | 简单任务，资源受限 |

根据实验和数据集特点，建议先尝试较大的学习率和较小的批大小进行初期训练，随后降低学习率进行精调；同时根据数据集大小调整dropout率和正则化强度。如果数据集较小，可以增加数据增强和提高dropout率来防止过拟合。

## 技术实现

- 使用VGG神经网络进行特征提取
- 使用PaddlePaddle 2.x版本的动态图模式
- 支持多类别人脸识别和人脸对比
- 实现了完整的检查点保存和加载机制

## 注意事项

1. 确保已正确安装PaddlePaddle及相关依赖：
   ```bash
   pip install paddlepaddle==2.4.2 opencv-python==4.5.5.64 matplotlib==3.5.3
   ```

2. 训练时根据实际数据集类别数量修改`num_classes`参数

3. 人脸对比的阈值默认为0.8，可根据需要调整

4. 模型训练和预测支持CPU和GPU，但GPU可大幅提升性能

5. 在进行继续训练前，确保已有`checkpoint.pdparams`文件 