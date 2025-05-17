#!/bin/bash
set -e # 任何命令失败立即退出

# --- 用户配置区域 (WSL 本地环境) ---
PROJECT_DIR="$(pwd)" # 使用当前目录作为项目目录，或者您可以硬编码一个绝对路径
VENV_PATH="paddle/bin/activate" # 虚拟环境激活脚本的相对路径 (相对于PROJECT_DIR)
CONFIG_FILE="configs/432_config.yaml"
LOG_DIR="${PROJECT_DIR}/logs_Terminal" # 日志文件保存目录

# 确保日志目录存在
mkdir -p "${LOG_DIR}"

# 激活虚拟环境 (如果使用)
if [ -f "${PROJECT_DIR}/${VENV_PATH}" ]; then
    echo "激活虚拟环境..."
    source "${PROJECT_DIR}/${VENV_PATH}"
else
    echo "警告: 虚拟环境激活脚本 '${VENV_PATH}' 未找到。请手动确保您的Python环境正确。"
    # 如果没有虚拟环境，依赖当前系统的 Python 环境
fi

# --- 定义参数集合用于组合生成 ---
# 这些集合必须与 default_config.yaml 中特定配置块名称的构成逻辑和值完全一致。
model_types=("vgg" "resnet")

# 使用代码简写和对应的全称，方便命名和配置修改
loss_codes=("ce" "arcface")
loss_types=("cross_entropy" "arcface") # Full names for YAML override

optimizer_codes=("adamw" "momentum")
optimizer_types=("AdamW" "Momentum") # Full names for YAML override

scheduler_codes=("stepdecay" "multistepdecay" "cosineannealingdecay" "reduceonplateau" "cosineannealingwarmrestarts" "polynomialdecay")
scheduler_types=("StepDecay" "MultiStepDecay" "CosineAnnealingDecay" "ReduceOnPlateau" "CosineAnnealingWarmRestarts" "PolynomialDecay") # Full names for YAML override

# 学习率 (LR) 和权重衰减 (WD) 的代码与值
lr_codes=("lr1" "lr2" "lr3")
lr_values=("0.1" "0.01" "0.001") # Values for YAML override

wd_codes=("wd1" "wd2" "wd3")
wd_values=("0.0001" "0.001" "0.01") # Values for YAML override

# 固定随机种子
SEED=42

# 检查参数集合数量是否对应 (简单验证)
if [ ${#loss_codes[@]} -ne ${#loss_types[@]} ] || \
   [ ${#optimizer_codes[@]} -ne ${#optimizer_types[@]} ] || \
   [ ${#scheduler_codes[@]} -ne ${#scheduler_types[@]} ] || \
   [ ${#lr_codes[@]} -ne ${#lr_values[@]} ] || \
   [ ${#wd_codes[@]} -ne ${#wd_values[@]} ]; then
   echo "错误: 参数代码和值数组长度不匹配，请检查脚本中的参数定义！"
   exit 1
fi

# --- 遍历所有组合并运行训练 ---
echo "----------------- 开始所有配置组合的训练 -----------------"

# 记录开始时间
START_TIME=$(date +%s)
CONFIG_COUNT=0
SUCCESS_COUNT=0
FAILURE_COUNT=0

# 使用嵌套循环遍历所有参数组合
for model_type in "${model_types[@]}"; do
    for loss_idx in "${!loss_codes[@]}"; do
        loss_code="${loss_codes[$loss_idx]}"
        loss_type="${loss_types[$loss_idx]}"

        for optimizer_idx in "${!optimizer_codes[@]}"; do
            optimizer_code="${optimizer_codes[$optimizer_idx]}"
            optimizer_type="${optimizer_types[$optimizer_idx]}"

            for scheduler_idx in "${!scheduler_codes[@]}"; do
                scheduler_code="${scheduler_codes[$scheduler_idx]}"
                scheduler_type="${scheduler_types[$scheduler_idx]}"

                for lr_idx in "${!lr_codes[@]}"; do
                    lr_code="${lr_codes[$lr_idx]}"
                    lr_value="${lr_values[$lr_idx]}"

                    for wd_idx in "${!wd_codes[@]}"; do
                        wd_code="${wd_codes[$wd_idx]}"
                        wd_value="${wd_values[$wd_idx]}"

                        # 构造当前的配置名称
                        config_name="${model_type}_${loss_code}_${optimizer_code}_${scheduler_code}_${lr_code}_${wd_code}_config"
                        CONFIG_COUNT=$((CONFIG_COUNT + 1))

                        echo "--- (${CONFIG_COUNT}/432) 准备训练配置: ${config_name} ---"
                        CURRENT_LOG_FILE="${LOG_DIR}/${config_name}_$(date +%Y%m%d-%H%M%S).log"

                        # --- 修改 default_config.yaml 文件中的 active_config ---
                        # 使用 sed 安全地替换 active_config 的值
                        # 假定 active_config 行格式固定且在文件开头部分
                        # 备份原文件
                        cp "${PROJECT_DIR}/${CONFIG_FILE}" "${PROJECT_DIR}/${CONFIG_FILE}.bak"

                        # 替换 active_config 行
                        # 注意: sed -i 在不同系统(如macOS)可能有差异，-i "" 表示不创建备份文件
                        # 如果您的系统需要，请调整 sed 命令
                        sed -i "s/^active_config: .*/active_config: ${config_name}/" "${PROJECT_DIR}/${CONFIG_FILE}"

                        # 检查文件是否成功修改 (可选)
                        if grep -q "^active_config: ${config_name}" "${PROJECT_DIR}/${CONFIG_FILE}"; then
                            echo "配置文件 active_config 已成功修改为: ${config_name}"
                        else
                            echo "错误: 无法修改配置文件 active_config。" | tee -a "${CURRENT_LOG_FILE}"
                            FAILURE_COUNT=$((FAILURE_COUNT + 1))
                            # 恢复原文件
                            mv "${PROJECT_DIR}/${CONFIG_FILE}.bak" "${PROJECT_DIR}/${CONFIG_FILE}"
                            continue # 跳过本次训练，进行下一个组合
                        fi

                        # --- 执行训练脚本 ---
                        echo "执行命令: python train.py --config_path ${CONFIG_FILE} --seed ${SEED} --use_gpu" | tee -a "${CURRENT_LOG_FILE}"
                        # 将 train.py 的输出同时输出到控制台和日志文件
                        python "${PROJECT_DIR}/train.py" --config_path "${PROJECT_DIR}/${CONFIG_FILE}" --seed ${SEED} --use_gpu --resume 2>&1 | tee -a "${CURRENT_LOG_FILE}"

                        # 检查 train.py 的退出状态
                        TRAIN_EXIT_STATUS=${PIPESTATUS[0]} # 获取 python 命令的退出状态
                        if [ ${TRAIN_EXIT_STATUS} -eq 0 ]; then
                            echo "--- 配置 ${config_name} 训练成功 ---" | tee -a "${CURRENT_LOG_FILE}"
                            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                        else
                            echo "--- 配置 ${config_name} 训练失败，退出状态: ${TRAIN_EXIT_STATUS} ---" | tee -a "${CURRENT_LOG_FILE}"
                            FAILURE_COUNT=$((FAILURE_COUNT + 1))
                            # 可以选择在这里添加重试逻辑或跳过
                        fi

                        # 恢复原文件 (无论是成功还是失败，都恢复备份)
                        mv "${PROJECT_DIR}/${CONFIG_FILE}.bak" "${PROJECT_DIR}/${CONFIG_FILE}"

                        echo "--- 配置 ${config_name} 检查/执行完毕 ---" | tee -a "${CURRENT_LOG_FILE}"
                        echo "" | tee -a "${CURRENT_LOG_FILE}" # 添加空行分隔日志

                    done # End wd_code loop
                done # End lr_code loop
            done # End scheduler_code loop
        done # End optimizer_code loop
    done # End loss_code loop
done # End model_type loop

# --- 训练检查/执行完毕 ---
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "----------------- 本轮所有配置训练检查/执行完毕 -----------------"
echo "总配置数: ${CONFIG_COUNT}"
echo "成功数: ${SUCCESS_COUNT}"
echo "失败数: ${FAILURE_COUNT}"
echo "总耗时: ${DURATION} 秒"