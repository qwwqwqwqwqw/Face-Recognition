#!/bin/bash
set -e

# Project 根目录 & 环境
PROJECT_DIR="$(pwd)"
VENV_PATH="paddle/bin/activate"
CONFIG_FILE="configs/432_config.yaml"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# 激活虚拟环境
if [ -f "${PROJECT_DIR}/${VENV_PATH}" ]; then
  echo "Activating virtualenv..."
  source "${PROJECT_DIR}/${VENV_PATH}"
else
  echo "Warning: virtualenv not found at ${VENV_PATH}"
fi

# Top N 配置名称列表 (当前包含所有评估准确率>0.5的配置)
# 请根据您的数据分析结果从以下列表中选择最终的 Top 50 配置
configs=(
  # ResNet 配置 (评估准确率 > 0.5)
  # resnet_ce_adamw_stepdecay_lr1_wd3_config
  # resnet_ce_adamw_stepdecay_lr2_wd2_config
  # resnet_ce_adamw_stepdecay_lr2_wd3_config
  # resnet_ce_adamw_stepdecay_lr3_wd1_config
  # resnet_ce_adamw_stepdecay_lr3_wd2_config
  
  # resnet_ce_adamw_multistepdecay_lr2_wd3_config
  # resnet_ce_adamw_multistepdecay_lr3_wd1_config
  # resnet_ce_adamw_multistepdecay_lr3_wd2_config
  
  # resnet_ce_adamw_cosineannealingdecay_lr2_wd3_config
  # resnet_ce_adamw_cosineannealingdecay_lr3_wd1_config
  # resnet_ce_adamw_cosineannealingdecay_lr3_wd2_config

  # resnet_ce_adamw_reduceonplateau_lr2_wd3_config
  # resnet_ce_adamw_reduceonplateau_lr3_wd1_config
  # resnet_ce_adamw_reduceonplateau_lr3_wd2_config

  # resnet_ce_adamw_cosineannealingwarmrestarts_lr2_wd2_config
  # resnet_ce_adamw_cosineannealingwarmrestarts_lr2_wd3_config
  # resnet_ce_adamw_cosineannealingwarmrestarts_lr3_wd1_config
  
  # resnet_ce_adamw_polynomialdecay_lr2_wd3_config
  # resnet_ce_adamw_polynomialdecay_lr3_wd1_config
  # resnet_ce_adamw_polynomialdecay_lr3_wd2_config
  # resnet_ce_adamw_polynomialdecay_lr3_wd3_config
  
  # resnet_ce_momentum_stepdecay_lr1_wd3_config
  # resnet_ce_momentum_stepdecay_lr2_wd1_config
  # resnet_ce_momentum_stepdecay_lr2_wd2_config
  # resnet_ce_momentum_stepdecay_lr2_wd3_config
  # resnet_ce_momentum_stepdecay_lr3_wd1_config
  # resnet_ce_momentum_stepdecay_lr3_wd2_config
  
  # resnet_ce_momentum_multistepdecay_lr1_wd3_config
  # resnet_ce_momentum_multistepdecay_lr2_wd1_config
  # resnet_ce_momentum_multistepdecay_lr2_wd2_config
  # resnet_ce_momentum_multistepdecay_lr2_wd3_config
  # resnet_ce_momentum_multistepdecay_lr3_wd1_config
  # resnet_ce_momentum_multistepdecay_lr3_wd2_config
  
  # resnet_ce_momentum_cosineannealingdecay_lr1_wd3_config
  # resnet_ce_momentum_cosineannealingdecay_lr2_wd1_config
  # resnet_ce_momentum_cosineannealingdecay_lr2_wd2_config
  # resnet_ce_momentum_cosineannealingdecay_lr2_wd3_config
  # resnet_ce_momentum_cosineannealingdecay_lr3_wd1_config
  # resnet_ce_momentum_cosineannealingdecay_lr3_wd2_config
  
  # resnet_ce_momentum_reduceonplateau_lr1_wd3_config
  # resnet_ce_momentum_reduceonplateau_lr2_wd1_config
  # resnet_ce_momentum_reduceonplateau_lr2_wd2_config
  # resnet_ce_momentum_reduceonplateau_lr2_wd3_config
  # resnet_ce_momentum_reduceonplateau_lr3_wd1_config
  # resnet_ce_momentum_reduceonplateau_lr3_wd2_config
  
  # resnet_ce_momentum_cosineannealingwarmrestarts_lr1_wd3_config
  # resnet_ce_momentum_cosineannealingwarmrestarts_lr2_wd1_config
  # resnet_ce_momentum_cosineannealingwarmrestarts_lr2_wd2_config
  # resnet_ce_momentum_cosineannealingwarmrestarts_lr2_wd3_config
  # resnet_ce_momentum_cosineannealingwarmrestarts_lr3_wd1_config
  # resnet_ce_momentum_cosineannealingwarmrestarts_lr3_wd2_config
  
  # resnet_ce_momentum_polynomialdecay_lr1_wd3_config
  # resnet_ce_momentum_polynomialdecay_lr2_wd1_config
  # resnet_ce_momentum_polynomialdecay_lr2_wd2_config
  # resnet_ce_momentum_polynomialdecay_lr2_wd3_config
  # resnet_ce_momentum_polynomialdecay_lr3_wd1_config
  # resnet_ce_momentum_polynomialdecay_lr3_wd2_config
  
  # # VGG 配置 (评估准确率 > 0.5)
  # vgg_ce_adamw_stepdecay_lr3_wd3_config

  # vgg_ce_adamw_multistepdecay_lr2_wd3_config
  # vgg_ce_adamw_multistepdecay_lr3_wd1_config
  # vgg_ce_adamw_multistepdecay_lr3_wd2_config

  # vgg_ce_adamw_cosineannealingdecay_lr2_wd3_config
  # vgg_ce_adamw_cosineannealingdecay_lr3_wd1_config
  # vgg_ce_adamw_cosineannealingdecay_lr3_wd2_config
  
  # vgg_ce_adamw_reduceonplateau_lr2_wd3_config
  # vgg_ce_adamw_reduceonplateau_lr3_wd1_config
  # vgg_ce_adamw_reduceonplateau_lr3_wd2_config

  # vgg_ce_adamw_cosineannealingwarmrestarts_lr3_wd1_config
  # vgg_ce_adamw_cosineannealingwarmrestarts_lr3_wd2_config

  # vgg_ce_adamw_polynomialdecay_lr2_wd3_config
  # vgg_ce_adamw_polynomialdecay_lr3_wd2_config
  
  # vgg_ce_momentum_stepdecay_lr1_wd3_config
  # vgg_ce_momentum_stepdecay_lr2_wd1_config
  # vgg_ce_momentum_stepdecay_lr2_wd2_config
  # vgg_ce_momentum_stepdecay_lr2_wd3_config
  # vgg_ce_momentum_stepdecay_lr3_wd1_config
  # vgg_ce_momentum_stepdecay_lr3_wd2_config
  
  # vgg_ce_momentum_multistepdecay_lr1_wd3_config
  # vgg_ce_momentum_multistepdecay_lr2_wd1_config
  # vgg_ce_momentum_multistepdecay_lr2_wd2_config
  # vgg_ce_momentum_multistepdecay_lr2_wd3_config
  # vgg_ce_momentum_multistepdecay_lr3_wd1_config
  # vgg_ce_momentum_multistepdecay_lr3_wd2_config
  
  # vgg_ce_momentum_cosineannealingdecay_lr1_wd3_config
  # vgg_ce_momentum_cosineannealingdecay_lr2_wd1_config
  # vgg_ce_momentum_cosineannealingdecay_lr2_wd2_config
  # vgg_ce_momentum_cosineannealingdecay_lr2_wd3_config
  # vgg_ce_momentum_cosineannealingdecay_lr3_wd1_config
  # vgg_ce_momentum_cosineannealingdecay_lr3_wd2_config

  # vgg_ce_momentum_reduceonplateau_lr1_wd3_config
  # vgg_ce_momentum_reduceonplateau_lr2_wd1_config
  # vgg_ce_momentum_reduceonplateau_lr2_wd2_config
  # vgg_ce_momentum_reduceonplateau_lr2_wd3_config
  # vgg_ce_momentum_reduceonplateau_lr3_wd1_config
  # vgg_ce_momentum_reduceonplateau_lr3_wd2_config

  # vgg_ce_momentum_cosineannealingwarmrestarts_lr1_wd3_config
  # vgg_ce_momentum_cosineannealingwarmrestarts_lr2_wd1_config
  # vgg_ce_momentum_cosineannealingwarmrestarts_lr2_wd2_config
  # vgg_ce_momentum_cosineannealingwarmrestarts_lr2_wd3_config
  # vgg_ce_momentum_cosineannealingwarmrestarts_lr3_wd1_config
  # vgg_ce_momentum_cosineannealingwarmrestarts_lr3_wd2_config
  
  # vgg_ce_momentum_polynomialdecay_lr1_wd3_config
  # vgg_ce_momentum_polynomialdecay_lr2_wd1_config
  # vgg_ce_momentum_polynomialdecay_lr2_wd2_config
  # vgg_ce_momentum_polynomialdecay_lr2_wd3_config
  # vgg_ce_momentum_polynomialdecay_lr3_wd1_config
  # vgg_ce_momentum_polynomialdecay_lr3_wd2_config
  # vgg_ce_momentum_polynomialdecay_lr3_wd3_config

  # #评估准确率为1的
#   vgg_arcface_adamw_stepdecay_lr1_wd1_config
#   vgg_arcface_adamw_stepdecay_lr1_wd2_config
#   vgg_arcface_adamw_stepdecay_lr1_wd3_config
#   vgg_arcface_adamw_stepdecay_lr2_wd1_config
#   vgg_arcface_adamw_stepdecay_lr2_wd2_config
#   vgg_arcface_adamw_stepdecay_lr2_wd3_config
#   vgg_arcface_adamw_stepdecay_lr3_wd1_config
#   vgg_arcface_adamw_stepdecay_lr3_wd2_config
#   vgg_arcface_adamw_stepdecay_lr3_wd3_config
#   vgg_arcface_adamw_multistepdecay_lr1_wd1_config
#   vgg_arcface_adamw_multistepdecay_lr1_wd2_config
#   vgg_arcface_adamw_multistepdecay_lr1_wd3_config
#   vgg_arcface_adamw_multistepdecay_lr2_wd1_config
#   vgg_arcface_adamw_multistepdecay_lr2_wd2_config
#   vgg_arcface_adamw_multistepdecay_lr2_wd3_config
#   vgg_arcface_adamw_multistepdecay_lr3_wd1_config
#   vgg_arcface_adamw_multistepdecay_lr3_wd2_config
#   vgg_arcface_adamw_multistepdecay_lr3_wd3_config
#   vgg_arcface_adamw_cosineannealingdecay_lr1_wd1_config
#   vgg_arcface_adamw_cosineannealingdecay_lr1_wd2_config
#   vgg_arcface_adamw_cosineannealingdecay_lr1_wd3_config
#   vgg_arcface_adamw_cosineannealingdecay_lr2_wd1_config
#   vgg_arcface_adamw_cosineannealingdecay_lr2_wd2_config
#   vgg_arcface_adamw_cosineannealingdecay_lr2_wd3_config
#   vgg_arcface_adamw_cosineannealingdecay_lr3_wd1_config
#   vgg_arcface_adamw_cosineannealingdecay_lr3_wd2_config
#   vgg_arcface_adamw_cosineannealingdecay_lr3_wd3_config
#   vgg_arcface_adamw_reduceonplateau_lr1_wd1_config
#   vgg_arcface_adamw_reduceonplateau_lr1_wd2_config
#   vgg_arcface_adamw_reduceonplateau_lr1_wd3_config
#   vgg_arcface_adamw_reduceonplateau_lr2_wd1_config
#   vgg_arcface_adamw_reduceonplateau_lr2_wd2_config
#   vgg_arcface_adamw_reduceonplateau_lr2_wd3_config
#   vgg_arcface_adamw_reduceonplateau_lr3_wd1_config
#   vgg_arcface_adamw_reduceonplateau_lr3_wd2_config
#   vgg_arcface_adamw_reduceonplateau_lr3_wd3_config
#   vgg_arcface_adamw_cosineannealingwarmrestarts_lr1_wd1_config
#   vgg_arcface_adamw_cosineannealingwarmrestarts_lr1_wd2_config
#   vgg_arcface_adamw_cosineannealingwarmrestarts_lr1_wd3_config
#   vgg_arcface_adamw_cosineannealingwarmrestarts_lr2_wd1_config
#   vgg_arcface_adamw_cosineannealingwarmrestarts_lr2_wd2_config
#   vgg_arcface_adamw_cosineannealingwarmrestarts_lr2_wd3_config
#   vgg_arcface_adamw_cosineannealingwarmrestarts_lr3_wd1_config
#   vgg_arcface_adamw_cosineannealingwarmrestarts_lr3_wd2_config
#   vgg_arcface_adamw_cosineannealingwarmrestarts_lr3_wd3_config
#   vgg_arcface_adamw_polynomialdecay_lr1_wd1_config
#   vgg_arcface_adamw_polynomialdecay_lr1_wd2_config
#   vgg_arcface_adamw_polynomialdecay_lr1_wd3_config
#   vgg_arcface_adamw_polynomialdecay_lr2_wd1_config
#   vgg_arcface_adamw_polynomialdecay_lr2_wd2_config
#   vgg_arcface_adamw_polynomialdecay_lr2_wd3_config
#   vgg_arcface_adamw_polynomialdecay_lr3_wd1_config
#   vgg_arcface_adamw_polynomialdecay_lr3_wd2_config
#   vgg_arcface_adamw_polynomialdecay_lr3_wd3_config
#   vgg_arcface_momentum_stepdecay_lr1_wd1_config
#   vgg_arcface_momentum_stepdecay_lr1_wd2_config
#   vgg_arcface_momentum_stepdecay_lr1_wd3_config
#   vgg_arcface_momentum_stepdecay_lr2_wd1_config
#   vgg_arcface_momentum_stepdecay_lr2_wd2_config
#   vgg_arcface_momentum_stepdecay_lr2_wd3_config
#   vgg_arcface_momentum_stepdecay_lr3_wd1_config
#   vgg_arcface_momentum_stepdecay_lr3_wd2_config
#   vgg_arcface_momentum_stepdecay_lr3_wd3_config
#   vgg_arcface_momentum_multistepdecay_lr1_wd1_config
#   vgg_arcface_momentum_multistepdecay_lr1_wd2_config
#   vgg_arcface_momentum_multistepdecay_lr1_wd3_config
#   vgg_arcface_momentum_multistepdecay_lr2_wd1_config
#   vgg_arcface_momentum_multistepdecay_lr2_wd2_config
#   vgg_arcface_momentum_multistepdecay_lr2_wd3_config
#   vgg_arcface_momentum_multistepdecay_lr3_wd1_config
#   vgg_arcface_momentum_multistepdecay_lr3_wd2_config
#   vgg_arcface_momentum_multistepdecay_lr3_wd3_config
#   vgg_arcface_momentum_cosineannealingdecay_lr1_wd1_config
#   vgg_arcface_momentum_cosineannealingdecay_lr1_wd2_config
#   vgg_arcface_momentum_cosineannealingdecay_lr1_wd3_config
#   vgg_arcface_momentum_cosineannealingdecay_lr2_wd1_config
#   vgg_arcface_momentum_cosineannealingdecay_lr2_wd2_config
#   vgg_arcface_momentum_cosineannealingdecay_lr2_wd3_config
#   vgg_arcface_momentum_cosineannealingdecay_lr3_wd1_config
#   vgg_arcface_momentum_cosineannealingdecay_lr3_wd2_config
#   vgg_arcface_momentum_cosineannealingdecay_lr3_wd3_config
#   vgg_arcface_momentum_reduceonplateau_lr1_wd1_config
#   vgg_arcface_momentum_reduceonplateau_lr1_wd2_config
#   vgg_arcface_momentum_reduceonplateau_lr1_wd3_config
#   vgg_arcface_momentum_reduceonplateau_lr2_wd1_config
#   vgg_arcface_momentum_reduceonplateau_lr2_wd2_config
#   vgg_arcface_momentum_reduceonplateau_lr2_wd3_config
#   vgg_arcface_momentum_reduceonplateau_lr3_wd1_config
#   vgg_arcface_momentum_reduceonplateau_lr3_wd2_config
#   vgg_arcface_momentum_reduceonplateau_lr3_wd3_config
#   vgg_arcface_momentum_cosineannealingwarmrestarts_lr1_wd1_config
#   vgg_arcface_momentum_cosineannealingwarmrestarts_lr1_wd2_config
#   vgg_arcface_momentum_cosineannealingwarmrestarts_lr1_wd3_config
#   vgg_arcface_momentum_cosineannealingwarmrestarts_lr2_wd1_config
#   vgg_arcface_momentum_cosineannealingwarmrestarts_lr2_wd2_config
#   vgg_arcface_momentum_cosineannealingwarmrestarts_lr2_wd3_config
#   vgg_arcface_momentum_cosineannealingwarmrestarts_lr3_wd1_config
#   vgg_arcface_momentum_cosineannealingwarmrestarts_lr3_wd2_config
#   vgg_arcface_momentum_cosineannealingwarmrestarts_lr3_wd3_config
#   vgg_arcface_momentum_polynomialdecay_lr1_wd1_config
#   vgg_arcface_momentum_polynomialdecay_lr1_wd2_config
#   vgg_arcface_momentum_polynomialdecay_lr1_wd3_config
#   vgg_arcface_momentum_polynomialdecay_lr2_wd1_config
#   vgg_arcface_momentum_polynomialdecay_lr2_wd2_config
#   vgg_arcface_momentum_polynomialdecay_lr2_wd3_config
#   vgg_arcface_momentum_polynomialdecay_lr3_wd1_config
#   vgg_arcface_momentum_polynomialdecay_lr3_wd2_config
#   vgg_arcface_momentum_polynomialdecay_lr3_wd3_config
#   resnet_arcface_adamw_stepdecay_lr1_wd1_config
#   resnet_arcface_adamw_stepdecay_lr1_wd2_config
#   resnet_arcface_adamw_stepdecay_lr1_wd3_config
#   resnet_arcface_adamw_stepdecay_lr2_wd1_config
#   resnet_arcface_adamw_stepdecay_lr2_wd2_config
#   resnet_arcface_adamw_stepdecay_lr2_wd3_config
#   resnet_arcface_adamw_stepdecay_lr3_wd1_config
#   resnet_arcface_adamw_stepdecay_lr3_wd2_config
#   resnet_arcface_adamw_stepdecay_lr3_wd3_config
#   resnet_arcface_adamw_multistepdecay_lr1_wd1_config
#   resnet_arcface_adamw_multistepdecay_lr1_wd2_config
#   resnet_arcface_adamw_multistepdecay_lr1_wd3_config
#   resnet_arcface_adamw_multistepdecay_lr2_wd1_config
#   resnet_arcface_adamw_multistepdecay_lr2_wd2_config
#   resnet_arcface_adamw_multistepdecay_lr2_wd3_config
#   resnet_arcface_adamw_multistepdecay_lr3_wd1_config
#   resnet_arcface_adamw_multistepdecay_lr3_wd2_config
#   resnet_arcface_adamw_multistepdecay_lr3_wd3_config
  resnet_arcface_adamw_cosineannealingdecay_lr1_wd1_config
  # resnet_arcface_adamw_cosineannealingdecay_lr1_wd2_config
  # resnet_arcface_adamw_cosineannealingdecay_lr1_wd3_config
  # resnet_arcface_adamw_cosineannealingdecay_lr2_wd1_config
  # resnet_arcface_adamw_cosineannealingdecay_lr2_wd2_config
  # resnet_arcface_adamw_cosineannealingdecay_lr2_wd3_config
  # resnet_arcface_adamw_cosineannealingdecay_lr3_wd1_config
  # resnet_arcface_adamw_cosineannealingdecay_lr3_wd2_config
  # resnet_arcface_adamw_cosineannealingdecay_lr3_wd3_config

#   resnet_arcface_adamw_reduceonplateau_lr1_wd1_config
#   resnet_arcface_adamw_reduceonplateau_lr1_wd2_config
#   resnet_arcface_adamw_reduceonplateau_lr1_wd3_config
#   resnet_arcface_adamw_reduceonplateau_lr2_wd1_config
#   resnet_arcface_adamw_reduceonplateau_lr2_wd2_config
#   resnet_arcface_adamw_reduceonplateau_lr2_wd3_config
#   resnet_arcface_adamw_reduceonplateau_lr3_wd1_config
#   resnet_arcface_adamw_reduceonplateau_lr3_wd2_config
#   resnet_arcface_adamw_reduceonplateau_lr3_wd3_config
#   resnet_arcface_adamw_cosineannealingwarmrestarts_lr1_wd1_config
#   resnet_arcface_adamw_cosineannealingwarmrestarts_lr1_wd2_config
#   resnet_arcface_adamw_cosineannealingwarmrestarts_lr1_wd3_config
#   resnet_arcface_adamw_cosineannealingwarmrestarts_lr2_wd1_config
#   resnet_arcface_adamw_cosineannealingwarmrestarts_lr2_wd2_config
#   resnet_arcface_adamw_cosineannealingwarmrestarts_lr2_wd3_config
#   resnet_arcface_adamw_cosineannealingwarmrestarts_lr3_wd1_config
#   resnet_arcface_adamw_cosineannealingwarmrestarts_lr3_wd2_config
#   resnet_arcface_adamw_cosineannealingwarmrestarts_lr3_wd3_config
#   resnet_arcface_adamw_polynomialdecay_lr1_wd1_config
#   resnet_arcface_adamw_polynomialdecay_lr1_wd2_config
#   resnet_arcface_adamw_polynomialdecay_lr1_wd3_config
#   resnet_arcface_adamw_polynomialdecay_lr2_wd1_config
#   resnet_arcface_adamw_polynomialdecay_lr2_wd2_config
#   resnet_arcface_adamw_polynomialdecay_lr2_wd3_config
#   resnet_arcface_adamw_polynomialdecay_lr3_wd1_config
#   resnet_arcface_adamw_polynomialdecay_lr3_wd2_config
#   resnet_arcface_adamw_polynomialdecay_lr3_wd3_config
#   resnet_arcface_momentum_stepdecay_lr1_wd1_config
#   resnet_arcface_momentum_stepdecay_lr1_wd2_config
#   resnet_arcface_momentum_stepdecay_lr1_wd3_config
#   resnet_arcface_momentum_stepdecay_lr2_wd1_config
#   resnet_arcface_momentum_stepdecay_lr2_wd2_config
#   resnet_arcface_momentum_stepdecay_lr2_wd3_config
#   resnet_arcface_momentum_stepdecay_lr3_wd1_config
#   resnet_arcface_momentum_stepdecay_lr3_wd2_config
#   resnet_arcface_momentum_stepdecay_lr3_wd3_config
#   resnet_arcface_momentum_multistepdecay_lr1_wd1_config
#   resnet_arcface_momentum_multistepdecay_lr1_wd2_config
#   resnet_arcface_momentum_multistepdecay_lr1_wd3_config
#   resnet_arcface_momentum_multistepdecay_lr2_wd1_config
#   resnet_arcface_momentum_multistepdecay_lr2_wd2_config
#   resnet_arcface_momentum_multistepdecay_lr2_wd3_config
#   resnet_arcface_momentum_multistepdecay_lr3_wd1_config
#   resnet_arcface_momentum_multistepdecay_lr3_wd2_config
#   resnet_arcface_momentum_multistepdecay_lr3_wd3_config
#   resnet_arcface_momentum_cosineannealingdecay_lr1_wd1_config
#   resnet_arcface_momentum_cosineannealingdecay_lr1_wd2_config
#   resnet_arcface_momentum_cosineannealingdecay_lr1_wd3_config
#   resnet_arcface_momentum_cosineannealingdecay_lr2_wd1_config
#   resnet_arcface_momentum_cosineannealingdecay_lr2_wd2_config
#   resnet_arcface_momentum_cosineannealingdecay_lr2_wd3_config
#   resnet_arcface_momentum_cosineannealingdecay_lr3_wd1_config
#   resnet_arcface_momentum_cosineannealingdecay_lr3_wd2_config
#   resnet_arcface_momentum_cosineannealingdecay_lr3_wd3_config
#   resnet_arcface_momentum_reduceonplateau_lr1_wd1_config
#   resnet_arcface_momentum_reduceonplateau_lr1_wd2_config
#   resnet_arcface_momentum_reduceonplateau_lr1_wd3_config
#   resnet_arcface_momentum_reduceonplateau_lr2_wd1_config
#   resnet_arcface_momentum_reduceonplateau_lr2_wd2_config
#   resnet_arcface_momentum_reduceonplateau_lr2_wd3_config
#   resnet_arcface_momentum_reduceonplateau_lr3_wd1_config
#   resnet_arcface_momentum_reduceonplateau_lr3_wd2_config
#   resnet_arcface_momentum_reduceonplateau_lr3_wd3_config
#   resnet_arcface_momentum_cosineannealingwarmrestarts_lr1_wd1_config
#   resnet_arcface_momentum_cosineannealingwarmrestarts_lr1_wd2_config
#   resnet_arcface_momentum_cosineannealingwarmrestarts_lr1_wd3_config
#   resnet_arcface_momentum_cosineannealingwarmrestarts_lr2_wd1_config
#   resnet_arcface_momentum_cosineannealingwarmrestarts_lr2_wd2_config
#   resnet_arcface_momentum_cosineannealingwarmrestarts_lr2_wd3_config
#   resnet_arcface_momentum_cosineannealingwarmrestarts_lr3_wd1_config
#   resnet_arcface_momentum_cosineannealingwarmrestarts_lr3_wd2_config
#   resnet_arcface_momentum_cosineannealingwarmrestarts_lr3_wd3_config
#   resnet_arcface_momentum_polynomialdecay_lr1_wd1_config
#   resnet_arcface_momentum_polynomialdecay_lr1_wd2_config
#   resnet_arcface_momentum_polynomialdecay_lr1_wd3_config
#   resnet_arcface_momentum_polynomialdecay_lr2_wd1_config
#   resnet_arcface_momentum_polynomialdecay_lr2_wd2_config
#   resnet_arcface_momentum_polynomialdecay_lr2_wd3_config
#   resnet_arcface_momentum_polynomialdecay_lr3_wd1_config
#   resnet_arcface_momentum_polynomialdecay_lr3_wd2_config
#   resnet_arcface_momentum_polynomialdecay_lr3_wd3_config
  
)

# 循环训练
# 注意：此循环会覆盖 configs/top50_config.yaml 中的 active_config 并运行 train.py
# train.py 中的 CheckpointManager 会将模型和日志保存在以 active_config 和 timestamp 命名的子目录下
for cfg in "${configs[@]}"; do
  echo "=== Training ${cfg} ==="
  # 备份原配置文件
  cp "${PROJECT_DIR}/${CONFIG_FILE}" "${PROJECT_DIR}/${CONFIG_FILE}.bak"

  # 使用 sed 修改 active_config
  # 确保 sed 命令在您的系统上工作正常
  # sed -i "" for macOS, sed -i for Linux
  SED_INPLACE="-i"
  if [[ "$OSTYPE" == "darwin"* ]]; then
    SED_INPLACE="-i \"\""
  fi
  # shellcheck disable=SC2086
  sed $SED_INPLACE "s/^active_config: .*/active_config: ${cfg}/" "${PROJECT_DIR}/${CONFIG_FILE}"

  # 构造日志文件名
  # 日志将保存在 logs/active_config_name/timestamp/ 结构中，此处仅用于tee到终端日志
  TERMINAL_LOGFILE="${LOG_DIR}/terminal_${cfg}_$(date +%Y%m%d-%H%M%S).log"

  # 执行训练脚本，输出同时到控制台和终端日志文件
  # train.py 内部会使用 VisualDL LogWriter 保存详细日志到 logs/active_config_name/timestamp/visualdl/
  python "${PROJECT_DIR}/train.py" \
    --config_path "${PROJECT_DIR}/${CONFIG_FILE}" \
    --use_gpu \
    --resume \
    2>&1 | tee "${TERMINAL_LOGFILE}"

  # 检查 train.py 的退出状态
  TRAIN_EXIT_STATUS=${PIPESTATUS[0]}
  if [ ${TRAIN_EXIT_STATUS} -eq 0 ]; then
      echo "--- 配置 ${cfg} 训练成功 ---" | tee -a "${TERMINAL_LOGFILE}"
  else
      echo "--- 配置 ${cfg} 训练失败，退出状态: ${TRAIN_EXIT_STATUS} ---" | tee -a "${TERMINAL_LOGFILE}"
      # 如果有失败，您可以选择在这里添加处理逻辑，例如跳过或重试
  fi

  # 恢复原配置文件
  mv "${PROJECT_DIR}/${CONFIG_FILE}.bak" "${PROJECT_DIR}/${CONFIG_FILE}"

  echo "--- 配置 ${cfg} 检查/执行完毕 ---" | tee -a "${TERMINAL_LOGFILE}"
  echo "" | tee -a "${TERMINAL_LOGFILE}" # 添加空行分隔日志

done

echo "All TopN experiments finished." # 将 Top50 改为 TopN
