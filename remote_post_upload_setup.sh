#!/bin/bash
set -e # 任何命令失败立即退出

#在服务器上由 upload_to_server.sh 触发，用于运行 CreateDataList.py 生成最新的数据列表和类别数文件，并设置数据更新标志。

# --- 用户配置区域 (服务器环境) ---
# !!! 请根据您的实际情况修改以下变量 !!!
# 服务器上的项目根目录 (通常由调用者确定，但最好能从环境或固定路径获取)
# REMOTE_PROJECT_DIR=$(cd "$(dirname "$0")" && pwd) # 假设脚本放在项目根目录
REMOTE_PROJECT_DIR="/root/Face-Recognition" # 或者硬编码，如果确定
VENV_PATH="${REMOTE_PROJECT_DIR}/paddle/bin/activate" # !!! 修改为您的虚拟环境路径 !!! 例如 paddle_env/bin/activate

# 数据集根目录，由第一个命令行参数传入
DATA_ROOT_ON_SERVER="$1" 

# 用于通知主训练脚本数据已更新的旗标文件路径
DATA_UPDATED_FLAG_FILE="${REMOTE_PROJECT_DIR}/data_updated.flag"
# 存储最新类别数的临时文件路径
NUM_CLASSES_FILE="${REMOTE_PROJECT_DIR}/latest_num_classes.txt"

# --- 脚本开始 ---
echo "===================================================="
echo "远程设置脚本启动 (在服务器上执行)..."
echo "脚本执行目录: $(pwd)"
echo "项目根目录 (预期): ${REMOTE_PROJECT_DIR}"
echo "接收到的数据根目录: ${DATA_ROOT_ON_SERVER}"
echo "虚拟环境路径: ${VENV_PATH}"
echo "===================================================="

# 检查参数
if [ -z "$DATA_ROOT_ON_SERVER" ]; then
    echo "错误: 未指定数据集根目录。脚本应接收数据根目录作为第一个参数。"
    exit 1
fi
if [ ! -d "$DATA_ROOT_ON_SERVER" ]; then
    echo "错误: 提供的数据集根目录不存在: ${DATA_ROOT_ON_SERVER}"
    exit 1
fi

# 1. 进入项目目录并激活虚拟环境
echo "--> 进入项目目录: ${REMOTE_PROJECT_DIR}"
cd "${REMOTE_PROJECT_DIR}" || { echo "错误: 无法进入项目目录 ${REMOTE_PROJECT_DIR}"; exit 1; }
echo "当前工作目录: $(pwd)"

echo "--> 激活虚拟环境: ${VENV_PATH}"
if [ -f "${VENV_PATH}" ]; then
    source "${VENV_PATH}" || { echo "错误: 无法激活虚拟环境 ${VENV_PATH}"; exit 1; }
else
    echo "错误: 虚拟环境激活脚本未找到: ${VENV_PATH}"
    exit 1
fi
echo "Python环境: $(which python)"

# 2. 生成数据列表
echo "--> 正在为 ${DATA_ROOT_ON_SERVER} 下的所有子目录生成数据列表..."
# 查找 data_root 下的所有第一级子目录 (这些是包含图像的类别目录)
DATASET_SUBDIRS=($(find "${DATA_ROOT_ON_SERVER}" -maxdepth 1 -mindepth 1 -type d))

if [ ${#DATASET_SUBDIRS[@]} -eq 0 ]; then
    echo "警告: 在 ${DATA_ROOT_ON_SERVER} 下没有找到任何子目录作为数据集。跳过 CreateDataList.py 执行。"
else
    echo "找到以下数据集子目录将进行处理:"
    printf " - %s\n" "${DATASET_SUBDIRS[@]}"
    
    SUCCESS_COUNT=0
    FAILURE_COUNT=0
    LATEST_CLASSES="" # 用于存储最后成功读取的类别数

    for dataset_path in "${DATASET_SUBDIRS[@]}"; do
        echo "  -> 处理数据集: ${dataset_path}"
        # CreateDataList.py 的参数是包含人物子文件夹的根目录
        python CreateDataList.py "${dataset_path}" 
        if [ $? -ne 0 ]; then
            echo "  错误: 为数据集 ${dataset_path} 生成数据列表失败。"
            ((FAILURE_COUNT++))
        else
            echo "  数据列表生成成功: ${dataset_path}"
            ((SUCCESS_COUNT++))
            # 读取生成的 readme.json 获取类别数
            README_JSON_PATH="${dataset_path}/readme.json"
            if [ -f "${README_JSON_PATH}" ]; then
                 # 使用 Python 或 jq 解析 JSON 更可靠，但 grep/awk 也能工作
                CURRENT_CLASSES=$(grep '"total_classes":' "${README_JSON_PATH}" | head -n 1 | awk '{print $2}' | sed 's/,//')
                if [[ "$CURRENT_CLASSES" =~ ^[0-9]+$ ]]; then # 确保是数字
                    echo "    从 ${README_JSON_PATH} 读取到类别数: ${CURRENT_CLASSES}"
                    LATEST_CLASSES=$CURRENT_CLASSES # 更新为最新读取到的有效类别数
                else
                    echo "    警告: 未能从 ${README_JSON_PATH} 中解析出有效的 total_classes 值。"
                fi
            else
                echo "    警告: 未找到 ${README_JSON_PATH} 文件，无法读取类别数。"
            fi
        fi
    done
    
    echo "数据列表生成完成。成功: ${SUCCESS_COUNT}, 失败: ${FAILURE_COUNT}."

    # 如果至少有一个数据集成功处理并且读取到了类别数，则保存
    if [ -n "$LATEST_CLASSES" ]; then
        echo "将最后成功读取的类别数 (${LATEST_CLASSES}) 保存到 ${NUM_CLASSES_FILE}"
        echo "${LATEST_CLASSES}" > "${NUM_CLASSES_FILE}"
    elif [ $SUCCESS_COUNT -gt 0 ]; then
         echo "警告: 数据列表已生成，但未能从任何 readme.json 文件中成功读取类别数。"
         # 可以选择删除旧的类别数文件，迫使训练脚本依赖 YAML
         # rm -f "${NUM_CLASSES_FILE}" 
    else
         echo "错误: 所有数据集的数据列表生成均失败。"
         # 可以选择退出脚本，防止创建更新标志
         # exit 1 
    fi
fi

# 3. 创建数据更新旗标文件，通知主训练脚本
echo "--> 创建数据更新旗标文件: ${DATA_UPDATED_FLAG_FILE}"
date > "${DATA_UPDATED_FLAG_FILE}" # 写入时间戳
echo "数据更新旗标已创建。"

echo "===================================================="
echo "远程设置脚本执行完毕。"
echo "===================================================="

# 停用虚拟环境 (可选)
echo "--> 停用虚拟环境..."
deactivate 2>/dev/null || true 