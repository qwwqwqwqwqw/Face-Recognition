#!/bin/bash
# auto_train_server.sh - 自动化人脸识别模型训练脚本 (CPU, CrossEntropy)
# 推荐使用 systemd 或类似的工具管理此脚本的运行。


#用于在服务器上自动化训练 CrossEntropy 模型（CPU），支持暂停/恢复，并在数据更新时自动重新加载配置和模型。

# --- 用户配置区域 (服务器环境) ---
# !!! 请根据您的实际情况修改以下变量 !!!
REMOTE_PROJECT_DIR="/root/Face-Recognition" # 服务器上的项目根目录
VENV_PATH="${REMOTE_PROJECT_DIR}/paddle_env/bin/activate" # 虚拟环境激活脚本路径 例如 paddle_env/bin/activate
CONFIG_FILE="configs/default_config.yaml" # 主配置文件 (相对于 PROJECT_DIR)

# 需要自动训练的 CrossEntropy 配置块名称列表
CONFIG_NAMES_TO_TRAIN=(
    "vgg_ce_config"
    "resnet_ce_config"
)

# --- 控制与状态文件路径 ---
# 暂停旗标文件路径 (管理员手动创建/删除以暂停/恢复)
PAUSE_FLAG_FILE="/tmp/pause_face_training.flag" 
# 数据更新旗标文件路径 (由 remote_post_upload_setup.sh 创建)
DATA_UPDATED_FLAG_FILE="${REMOTE_PROJECT_DIR}/data_updated.flag"
# 存储最新类别数的临时文件路径
NUM_CLASSES_FILE="${REMOTE_PROJECT_DIR}/latest_num_classes.txt"
# 存储最新Git Commit Hash的文件路径
LAST_GIT_HASH_FILE="${REMOTE_PROJECT_DIR}/.last_git_hash"

# --- 日志与输出 ---
LOG_DIR="${REMOTE_PROJECT_DIR}/logs_auto_train"
SERVICE_LOG_FILE="${LOG_DIR}/auto_train_service.log" # 主服务日志

# --- 训练循环控制 ---
# 完成一轮所有配置的训练后，暂停多长时间（秒）
POST_TRAIN_SLEEP_DURATION=3600 # 1小时
# 检查暂停标志的频率（秒）
PAUSE_CHECK_INTERVAL=60 
# 拉取代码失败后，等待多长时间重试（秒）
GIT_PULL_RETRY_INTERVAL=300 # 5分钟

# --- 函数定义 ---
log_message() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "${timestamp} - ${message}" | tee -a "${SERVICE_LOG_FILE}"
}

activate_venv() {
    log_message "激活虚拟环境: ${VENV_PATH}"
    if [ -f "${VENV_PATH}" ]; then
        source "${VENV_PATH}"
        if [ $? -ne 0 ]; then
            log_message "错误: 激活虚拟环境失败: ${VENV_PATH}"
            return 1
        fi
        log_message "Python环境: $(which python)"
    else
        log_message "错误: 虚拟环境激活脚本未找到: ${VENV_PATH}"
        return 1
    fi
    return 0
}

deactivate_venv() {
    log_message "尝试停用虚拟环境..."
    # deactivate 命令可能不存在或报错，忽略错误
    deactivate &>/dev/null || true 
}

# 优雅退出处理
trap 'log_message "接收到退出信号，正在清理..."; deactivate_venv; exit 0' SIGINT SIGTERM

# --- 脚本主逻辑 ---
log_message "===================================================="
log_message "自动化训练脚本启动..."
log_message "项目目录: ${REMOTE_PROJECT_DIR}"
log_message "日志目录: ${LOG_DIR}"
log_message "暂停旗标: ${PAUSE_FLAG_FILE}"
log_message "数据更新旗标: ${DATA_UPDATED_FLAG_FILE}"
log_message "===================================================="

# 进入项目目录
cd "${REMOTE_PROJECT_DIR}" || { log_message "错误: 无法进入项目目录 ${REMOTE_PROJECT_DIR}"; exit 1; }

# 创建日志目录
mkdir -p "${LOG_DIR}" || { log_message "错误: 无法创建日志目录 ${LOG_DIR}"; exit 1; }

# 激活虚拟环境
activate_venv || exit 1

# 读取上次的Git Hash（如果有）
previous_git_hash=""
if [ -f "${LAST_GIT_HASH_FILE}" ]; then
    previous_git_hash=$(cat "${LAST_GIT_HASH_FILE}")
fi

# 主循环
while true; do
    log_message "----------------- 新训练周期检查 -----------------"
    
    # 1. 检查暂停旗标
    if [ -f "${PAUSE_FLAG_FILE}" ]; then
        log_message "检测到暂停旗标 (${PAUSE_FLAG_FILE})。暂停 ${PAUSE_CHECK_INTERVAL} 秒..."
        sleep ${PAUSE_CHECK_INTERVAL}
        continue # 继续外层循环检查旗标
    fi
    
    # 2. 拉取最新代码和配置
    log_message "正在拉取最新代码 (git pull origin main)..."
    git fetch origin # 先获取远程更新信息
    current_local_hash=$(git rev-parse HEAD)
    current_remote_hash=$(git rev-parse origin/main)

    if [ "$current_local_hash" != "$current_remote_hash" ]; then
        log_message "检测到远程代码更新。正在执行 git pull..."
        git pull origin main
        pull_status=$?
        if [ $pull_status -ne 0 ]; then
            log_message "警告: Git 拉取失败 (状态码: ${pull_status})。将在 ${GIT_PULL_RETRY_INTERVAL} 秒后重试。"
            sleep ${GIT_PULL_RETRY_INTERVAL}
            continue # 重新开始循环，尝试再次拉取
        else
            log_message "代码拉取成功。"
            # 更新 Git Hash 记录文件
            current_git_hash=$(git rev-parse HEAD)
            echo "${current_git_hash}" > "${LAST_GIT_HASH_FILE}"
            previous_git_hash=$current_git_hash # 更新内存中的hash
        fi
    else
        log_message "本地代码已是最新 (${current_local_hash:0:7})。"
        # 确保记录文件存在且内容正确
        if [ ! -f "${LAST_GIT_HASH_FILE}" ] || [ "$(cat "${LAST_GIT_HASH_FILE}")" != "$current_local_hash" ]; then
             echo "${current_local_hash}" > "${LAST_GIT_HASH_FILE}"
        fi
    fi

    # 3. 检查数据更新旗标并处理
    current_num_classes="" # 重置当前轮次的类别数
    data_needs_processing=false
    if [ -f "${DATA_UPDATED_FLAG_FILE}" ]; then
        log_message "检测到数据更新旗标 (${DATA_UPDATED_FLAG_FILE})。将触发数据处理。"
        data_needs_processing=true
        # 不需要在这里运行 CreateDataList，假设 remote_setup 脚本已完成
        # 读取由 remote_setup 生成的类别数文件
        if [ -f "${NUM_CLASSES_FILE}" ]; then
            current_num_classes=$(cat "${NUM_CLASSES_FILE}")
            if [[ ! "$current_num_classes" =~ ^[0-9]+$ ]]; then
                 log_message "警告: 从 ${NUM_CLASSES_FILE} 读取的类别数无效 ('${current_num_classes}')。将不使用。"
                 current_num_classes=""
            else
                log_message "从 ${NUM_CLASSES_FILE} 读取到新的类别数: ${current_num_classes}"
            fi
        else
            log_message "警告: 检测到数据更新旗标，但未找到类别数文件 (${NUM_CLASSES_FILE})。可能是 remote_setup 脚本未能读取到类别数。"
        fi
        # 移除数据更新旗标
        log_message "移除数据更新旗标: ${DATA_UPDATED_FLAG_FILE}"
        rm "${DATA_UPDATED_FLAG_FILE}" || log_message "警告: 移除数据更新旗标失败。"
    else
        # log_message "未检测到数据更新旗标。"
        # 如果没有更新，尝试使用上次保存的类别数
        if [ -f "${NUM_CLASSES_FILE}" ]; then
            current_num_classes=$(cat "${NUM_CLASSES_FILE}")
             if [[ ! "$current_num_classes" =~ ^[0-9]+$ ]]; then
                 log_message "警告: 从 ${NUM_CLASSES_FILE} 读取的上次类别数无效 ('${current_num_classes}')。将不使用。"
                 current_num_classes=""
             # else
                 # log_message "使用上次读取的类别数: ${current_num_classes}"
             fi
        fi
    fi
    
    # 4. 循环训练配置
    log_message "开始自动化训练循环..."
    all_configs_trained_successfully_this_round=true # 标记本轮是否所有配置都成功

    for config_name in "${CONFIG_NAMES_TO_TRAIN[@]}"; do
        # 在每个训练任务前再次检查暂停旗标
        if [ -f "${PAUSE_FLAG_FILE}" ]; then
             log_message "检测到暂停旗标 (${PAUSE_FLAG_FILE})。中断当前训练轮次。"
             all_configs_trained_successfully_this_round=false # 标记本轮未完成
             break # 跳出 for 循环
        fi

        log_message "----------------------------------------------------"
        log_message "准备训练配置: ${config_name}"
        
        # 构建日志文件名
        train_log_file="${LOG_DIR}/train_${config_name}_cpu_auto_$(date +%Y%m%d_%H%M%S).log"
        log_message "训练日志将保存到: ${train_log_file}"

        # 构建 train.py 命令
        # --no-use_gpu: 强制使用CPU
        # --resume: 尝试从检查点恢复
        # --source auto: 告知 train.py 这是自动化训练，用于模型命名和元数据
        # --num_classes (可选): 如果读取到有效值，则传递以覆盖YAML
        train_cmd_base="python train.py --config_path \"${CONFIG_FILE}\" --active_config \"${config_name}\" --no-use_gpu --resume --source auto"
        
        train_cmd="${train_cmd_base}"
        if [ -n "$current_num_classes" ]; then
             train_cmd="${train_cmd} --num_classes ${current_num_classes}"
             log_message "将使用类别数: ${current_num_classes} (来自文件或数据更新)"
        else
             log_message "未指定类别数，将使用配置文件 (${config_name}) 中的默认值。"
        fi

        log_message "执行命令: ${train_cmd}"
        # 使用 eval 执行命令，并将标准输出和错误都重定向到日志文件
        eval ${train_cmd} > "${train_log_file}" 2>&1 
        train_exit_code=$?

        if [ $train_exit_code -ne 0 ]; then
            log_message "警告: 配置 '${config_name}' 的训练过程失败 (退出码: ${train_exit_code})。详情请查看日志: '${train_log_file}'"
            all_configs_trained_successfully_this_round=false # 标记本轮训练有失败
            # 考虑是否需要通知管理员
        else
             log_message "配置 '${config_name}' 的训练过程已完成。日志: '${train_log_file}'"
             # 可以在这里添加成功后的操作，例如更新一个状态文件
        fi
    done # 结束 for config_name 循环
    
    # 检查是否因为暂停而中断
    if [ -f "${PAUSE_FLAG_FILE}" ]; then
        log_message "训练循环因暂停旗标而中断。"
        sleep ${PAUSE_CHECK_INTERVAL} # 短暂暂停后继续检查
        continue # 回到 while 循环顶部检查暂停
    fi

    log_message "本轮自动化训练循环结束。"
    
    # 在完成一整轮训练后暂停指定时间
    log_message "暂停 ${POST_TRAIN_SLEEP_DURATION} 秒..."
    sleep ${POST_TRAIN_SLEEP_DURATION}

done # 结束 while true 循环

# 理论上不会执行到这里，除非手动停止循环或发生未捕获的错误
log_message "自动化训练脚本意外退出。"
deactivate_venv
exit 1 # 非正常退出 