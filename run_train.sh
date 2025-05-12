#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- 配置 ---
# !!! 请将此路径替换为您的实际项目根目录 !!!
PROJECT_DIR="/root/Face-Recognition"
VENV_PATH="paddle_env/bin/activate" # 虚拟环境激活脚本的相对路径
CONFIG_FILE="configs/default_config.yaml" # 主配置文件
DATA_LIST_FILE_FOR_LIB="data/face/trainer.list" # 用于创建特征库的数据列表文件

# 需要训练的配置块名称列表 (与YAML文件中的键名对应)
CONFIG_NAMES_TO_TRAIN=(
    "vgg_ce_config"
    "vgg_arcface_config"
    "resnet_ce_config"
    "resnet_arcface_config"
)

# --- 脚本开始 ---
echo "自动化训练脚本启动..."

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

    # 运行训练脚本
    # 注意: 下面的 --active_config 参数需要 train.py 和 config_utils.py 支持
    #       通过命令行参数来实际选择和加载 YAML 中的配置块。
    #       如果不支持，训练将始终使用 YAML 文件中 'active_config:' 指定的块。
    echo "启动训练 (train.py) for ${config_name}..."
    python train.py \
        --config_path "${CONFIG_FILE}" \
        --active_config "${config_name}" \
        --no-use_gpu \
        --resume \
        > "${LOG_FILE_TRAIN}" 2>&1

    echo "训练配置 ${config_name} 完成。检查日志文件 ${LOG_FILE_TRAIN} 获取详情。"

    # (可选) 如果是ArcFace模型，训练完成后创建特征库
    if [[ "${config_name}" == *"arcface"* ]]; then
        echo ""
        echo "----------------------------------------------------------------------"
        echo "检测到ArcFace配置 (${config_name})，准备为其创建人脸特征库..."
        echo "----------------------------------------------------------------------"

        # 从配置名推断模型文件名后缀 (e.g., vgg_arcface_config -> vgg_arcface)
        model_suffix_for_lib="${config_name%_config}"
        MODEL_PATH_FOR_LIB="model/best_model_${model_suffix_for_lib}.pdparams"
        LOG_FILE_CREATE_LIB="create_library_${config_name}.log"

        echo "将使用模型: ${MODEL_PATH_FOR_LIB}"
        echo "将使用数据列表: ${DATA_LIST_FILE_FOR_LIB}"
        echo "特征库创建日志将保存到: ${LOG_FILE_CREATE_LIB}"

        if [ ! -f "${MODEL_PATH_FOR_LIB}" ]; then
            echo "警告: 预期的最佳模型文件 ${MODEL_PATH_FOR_LIB} 未找到。可能训练未成功保存模型，或者路径/命名规则不匹配。将跳过为此配置创建特征库。"
        else
            echo "启动创建特征库 (create_face_library.py) for ${config_name}..."
            python create_face_library.py \
                --config_path "${CONFIG_FILE}" \
                --active_config "${config_name}" \
                --model_path "${MODEL_PATH_FOR_LIB}" \
                --data_list_file "${DATA_LIST_FILE_FOR_LIB}" \
                --no-use_gpu \
                > "${LOG_FILE_CREATE_LIB}" 2>&1
            echo "为 ${config_name} 创建特征库完成。检查日志文件 ${LOG_FILE_CREATE_LIB} 获取详情。"
        fi
    fi
done

echo ""
echo "======================================================================"
echo "所有指定的模型训练配置均已执行完毕。"
echo "======================================================================"
echo "自动化训练脚本结束。" 