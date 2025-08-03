#!/bin/bash

# 手部抓握分类 - 快速测试脚本
# 用于验证数据集和模型是否正常工作

export CUDA_VISIBLE_DEVICES=0

echo "=== 手部抓握分类快速测试 ==="

# 测试健康状态分类任务（训练1个epoch快速验证）
echo "测试健康状态分类数据集..."

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/HandGrip_Enhanced/ \
  --data_path HandGrip_health_status.ts \
  --model_id HandGrip_test \
  --model TimesNet \
  --data UEA \
  --e_layers 2 \
  --batch_size 8 \
  --d_model 32 \
  --d_ff 64 \
  --top_k 2 \
  --num_kernels 3 \
  --des 'QuickTest' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 1 \
  --patience 5

if [ $? -eq 0 ]; then
    echo "✅ 测试成功！数据集和模型配置正确"
    echo "现在可以运行完整训练脚本："
    echo "bash ./scripts/classification/HandGrip_Classification.sh"
else
    echo "❌ 测试失败，请检查数据集格式或模型配置"
fi