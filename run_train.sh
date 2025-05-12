#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- 配置 ---
# !!! 请将此路径替换为您的实际项目根目录 !!!
PROJECT_DIR="/root/Face-Recognition"
VENV_PATH="paddle_env/bin/activate" # 虚拟环境激活脚本的相对路径
CONFIG_FILE="configs/default_config.yaml" # 主配置文件
# DATA_LIST_FILE_FOR_LIB="data/face/trainer.list" # ArcFace相关，已移除

# 需要训练的配置块名称列表 (与YAML文件中的键名对应)
# 由于 margin_cross_entropy (ArcFace核心) 在PaddlePaddle CPU上不支持，
# 我们暂时只训练 CrossEntropy (ce) 模型。
CONFIG_NAMES_TO_TRAIN=(
    "vgg_ce_config"
    "resnet_ce_config"
)
# 原来的ArcFace配置:
# "vgg_arcface_config"
# "resnet_arcface_config"

# --- 脚本开始 ---
echo "自动化训练脚本启动 (CPU模式)..."

# 1. 进入项目目录
echo "--> 正在进入项目目录: ${PROJECT_DIR}"
cd "${PROJECT_DIR}" || { echo "错误: 无法进入项目目录 ${PROJECT_DIR}"; exit 1; }
echo "当前工作目录: $(pwd)"

# 2. 激活Python虚拟环境
if [ -f "${VENV_PATH}" ]; then
    echo "--> 正在激活Python虚拟环境: ${VENV_PATH}"
    source "${VENV_PATH}" || { echo "错误: 无法激活虚拟环境 ${VENV_PATH}"; exit 1; }
else
    echo "警告: 虚拟环境激活脚本未找到于 ${VENV_PATH}。将尝试在无虚拟环境的情况下继续。"
    echo "     请确保您的环境中已安装必要的依赖 (如 PaddlePaddle)。"
fi

# 3. 从Git拉取最新的代码和配置
echo "--> 正在从Git仓库拉取最新代码 (origin main)..."
git pull origin main || { echo "警告: git pull 失败。将使用本地现有代码继续。"; }

echo "--> 最新代码拉取完成（或已跳过）。"

# 4. 循环训练不同的模型组合
echo "--> 开始循环训练模型组合..."

for config_name in "${CONFIG_NAMES_TO_TRAIN[@]}"; do
    echo ""
    echo "======================================================================"
    echo "准备训练配置: ${config_name}"
    echo "======================================================================"

    LOG_FILE_TRAIN="train_${config_name}.log"
    echo "训练日志将保存到: ${LOG_FILE_TRAIN}"

    echo "启动训练 (train.py) for ${config_name}..."
    # 现在 train.py 可以通过 --active_config 参数选择配置块
    python train.py \
        --config_path "${CONFIG_FILE}" \
        --active_config "${config_name}" \
        --no-use_gpu \
        --resume \
        > "${LOG_FILE_TRAIN}" 2>&1

    echo "训练配置 ${config_name} 完成。检查日志文件 ${LOG_FILE_TRAIN} 获取详情。"

    # ArcFace相关创建特征库的逻辑已移除
done

echo ""
echo "======================================================================"
echo "所有指定的模型训练配置均已执行完毕。"
echo "======================================================================"
echo "自动化训练脚本结束。" 