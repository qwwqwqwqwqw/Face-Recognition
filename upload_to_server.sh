#!/bin/bash
set -e # 任何命令失败立即退出

#用于上传数据集和本地 GPU 模型到服务器，并触发服务器端的设置脚本。

# --- 用户配置区域 (WSL 本地环境) ---
# !!! 请根据您的实际情况修改以下变量 !!!
SERVER_USER="root"
SERVER_IP="139.9.133.46"
REMOTE_PROJECT_DIR="/root/Face-Recognition" # 例如 /root/Face-Recognition
# 使用你的私钥路径进行 SSH 无密码登录，如果需要的话
# SSH_KEY_PATH="~/.ssh/your_private_key" 
# SSH_OPTIONS="-i ${SSH_KEY_PATH}" # 如果使用密钥，取消此行注释并在 scp/ssh 命令中添加 ${SSH_OPTIONS}

LOCAL_PROJECT_DIR="$(pwd)" # 假设在项目根目录下运行此脚本
LOCAL_DATA_DIR="${LOCAL_PROJECT_DIR}/data" # 本地数据集根目录
LOCAL_MODEL_DIR="${LOCAL_PROJECT_DIR}/model" # 本地模型保存目录

# 需要上传到服务器的数据集子目录名称 (位于 data/ 下)
# 请确保列出所有需要同步的数据子目录
DATASET_SUBDIRS_TO_UPLOAD=(
    "face"      # 你的主数据集目录
    "my_photos" # 可能新增的个人照片目录
    # "acceptance_test" # 如果验收测试集也在本地维护
    # 添加其他需要同步的子目录...
)

# --- 脚本开始 ---
echo "===================================================="
echo "数据和模型上传脚本启动 (本地 WSL -> 服务器)..."
echo "本地项目目录: ${LOCAL_PROJECT_DIR}"
echo "目标服务器: ${SERVER_USER}@${SERVER_IP}:${REMOTE_PROJECT_DIR}"
echo "===================================================="

# 检查 SSH 连接是否正常 (可选，但推荐)
echo "--> 检查SSH连接..."
ssh "${SERVER_USER}@${SERVER_IP}" "exit" # 使用 ${SSH_OPTIONS} 如果需要密钥
if [ $? -ne 0 ]; then
    echo "错误: 无法建立SSH连接到 ${SERVER_USER}@${SERVER_IP}。请检查SSH配置、网络和密钥（如果使用）。"
    exit 1
fi
echo "SSH连接成功。"

# 1. 上传数据集
echo "--> 正在上传数据集目录..."
REMOTE_DATA_ROOT="${REMOTE_PROJECT_DIR}/data"
# 确保服务器上的 data 根目录存在
ssh "${SERVER_USER}@${SERVER_IP}" "mkdir -p ${REMOTE_DATA_ROOT}" # 使用 ${SSH_OPTIONS} 如果需要密钥
for subdir in "${DATASET_SUBDIRS_TO_UPLOAD[@]}"; do
    LOCAL_SUBDIR_PATH="${LOCAL_DATA_DIR}/${subdir}"
    if [ -d "${LOCAL_SUBDIR_PATH}" ]; then
        echo "上传目录: ${LOCAL_SUBDIR_PATH} -> ${REMOTE_DATA_ROOT}/"
        # 使用 rsync 通常比 scp -r -u 更高效，特别是对于大量小文件或需要删除服务器上多余文件时
        # rsync -avz -e "ssh ${SSH_OPTIONS}" --delete --update "${LOCAL_SUBDIR_PATH}/" "${SERVER_USER}@${SERVER_IP}:${REMOTE_DATA_ROOT}/${subdir}/"
        # 这里继续使用 scp -r -u 以保持与原始方案一致
        scp -r -u "${LOCAL_SUBDIR_PATH}" "${SERVER_USER}@${SERVER_IP}:${REMOTE_DATA_ROOT}/" # 使用 ${SSH_OPTIONS} 如果需要密钥
        if [ $? -ne 0 ]; then
            echo "警告: 上传数据集目录失败: ${LOCAL_SUBDIR_PATH}。将继续上传其他文件。"
        else
            echo "数据集目录上传/更新成功: ${LOCAL_SUBDIR_PATH}"
        fi
    else
        echo "警告: 本地数据集目录未找到，跳过: ${LOCAL_SUBDIR_PATH}"
    fi
done
echo "--> 数据集上传完毕。"

# 2. 上传本地 GPU 训练的模型文件和元数据
echo "--> 正在上传本地 GPU 训练的模型文件 (*_gpu*)..."
REMOTE_MODEL_DIR="${REMOTE_PROJECT_DIR}/model"
# 确保服务器上的 model 目录存在
ssh "${SERVER_USER}@${SERVER_IP}" "mkdir -p ${REMOTE_MODEL_DIR}" # 使用 ${SSH_OPTIONS} 如果需要密钥

# 查找所有包含 _gpu 的 .pdparams 或 .json 文件 (更灵活的命名)
# 注意：这可能包含非手动训练的文件，如果命名不严格区分 manual/auto
# 如果严格要求只上传手动GPU模型，命名应为 *_gpu_manual.*
LOCAL_GPU_MODEL_FILES=()
while IFS= read -r -d $'\0'; do
    LOCAL_GPU_MODEL_FILES+=("$REPLY")
done < <(find "${LOCAL_MODEL_DIR}" -maxdepth 1 -type f \( -name "*_gpu*.pdparams" -o -name "*_gpu*.json" \) -print0)


if [ ${#LOCAL_GPU_MODEL_FILES[@]} -eq 0 ]; then
    echo "没有找到本地 GPU 训练的模型文件需要上传 (${LOCAL_MODEL_DIR}/*_gpu*.{pdparams,json})。"
else
    echo "找到 ${#LOCAL_GPU_MODEL_FILES[@]} 个本地 GPU 相关文件。目标服务器模型目录: ${SERVER_USER}@${SERVER_IP}:${REMOTE_MODEL_DIR}"
    for model_file in "${LOCAL_GPU_MODEL_FILES[@]}"; do
         # 使用 -u 参数仅上传更新的文件
        echo "上传文件: ${model_file} -> ${REMOTE_MODEL_DIR}/"
        scp -u "${model_file}" "${SERVER_USER}@${SERVER_IP}:${REMOTE_MODEL_DIR}/" # 使用 ${SSH_OPTIONS} 如果需要密钥
        if [ $? -ne 0 ]; then
            echo "警告: 上传模型文件失败: ${model_file}"
        else
            echo "模型文件上传/更新成功: ${model_file}"
        fi
    done
fi
echo "--> 本地 GPU 模型上传完毕。"

# 3. 远程执行服务器上的数据列表生成和设置脚本
echo "--> 正在远程触发服务器端设置脚本..."
REMOTE_SETUP_SCRIPT="${REMOTE_PROJECT_DIR}/remote_post_upload_setup.sh" # 服务器上的辅助脚本路径
REMOTE_SETUP_LOG_DIR="${REMOTE_PROJECT_DIR}/logs_auto_train" # 远程日志目录
REMOTE_SETUP_LOG_FILE="${REMOTE_SETUP_LOG_DIR}/remote_setup_log_$(date +%Y%m%d_%H%M%S).txt" # 带时间戳的日志

# 使用 heredoc 来构建远程执行的命令，更清晰
ssh "${SERVER_USER}@${SERVER_IP}" bash -s -- "${REMOTE_DATA_ROOT}" << EOF # 使用 ${SSH_OPTIONS} 如果需要密钥
set -e # 远程脚本内部也设置立即退出
echo "--- 开始远程执行设置脚本 ($(date)) ---"
REMOTE_PROJECT_DIR_INNER="${REMOTE_PROJECT_DIR}" # 在heredoc内部重新定义，避免混淆
REMOTE_SETUP_SCRIPT_INNER="${REMOTE_SETUP_SCRIPT}"
REMOTE_SETUP_LOG_DIR_INNER="${REMOTE_SETUP_LOG_DIR}"
REMOTE_SETUP_LOG_FILE_INNER="${REMOTE_SETUP_LOG_FILE}"
DATA_DIR_ARG="\$1" # 接收传递给heredoc的第一个参数

echo "项目目录: \${REMOTE_PROJECT_DIR_INNER}"
echo "日志文件: \${REMOTE_SETUP_LOG_FILE_INNER}"
echo "数据目录参数: \${DATA_DIR_ARG}"

# 确保日志目录存在
mkdir -p "\${REMOTE_SETUP_LOG_DIR_INNER}"

# 确保远程脚本存在且可执行
if [ ! -x "\${REMOTE_SETUP_SCRIPT_INNER}" ]; then
    echo "错误: 服务器上的远程设置脚本不存在或无执行权限: \${REMOTE_SETUP_SCRIPT_INNER}"
    exit 1
fi

echo "正在服务器上执行远程设置脚本: \${REMOTE_SETUP_SCRIPT_INNER}"
# 将标准输出和错误都重定向到日志文件
bash "\${REMOTE_SETUP_SCRIPT_INNER}" "\${DATA_DIR_ARG}" > "\${REMOTE_SETUP_LOG_FILE_INNER}" 2>&1

if [ \$? -ne 0 ]; then
    echo "警告: 服务器上的远程设置脚本执行失败。详情请检查日志: \${REMOTE_SETUP_LOG_FILE_INNER}"
    # 这里可以选择 exit 1 来让本地脚本知道远程执行失败
    # exit 1
else
    echo "服务器上的远程设置脚本执行成功。日志: \${REMOTE_SETUP_LOG_FILE_INNER}"
fi
echo "--- 远程执行设置脚本结束 ($(date)) ---"
EOF # 结束 heredoc

if [ $? -ne 0 ]; then
    echo "警告: SSH远程执行远程设置脚本失败。请SSH登录服务器查看详细日志。"
else
     echo "SSH远程执行远程设置脚本成功完成。"
fi

echo "===================================================="
echo "上传脚本执行完毕。"
echo "===================================================="
echo "!!! 重要提示 !!!"
echo "1. 请SSH登录服务器，检查最新生成的 ${REMOTE_DATA_ROOT}/<数据集子目录>/readme.json 文件获取类别数。"
echo "2. 如果类别数有变化，请务必更新本地项目的 configs/default_config.yaml 文件中所有相关配置块的 num_classes 参数。"
echo "3. 将更新后的 default_config.yaml 通过 git commit 和 git push 推送到远程仓库，以便服务器上的自动化脚本能够拉取到最新的配置。"
echo "4. 远程设置脚本的详细日志位于服务器的: ${REMOTE_SETUP_LOG_FILE}" 