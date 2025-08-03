#!/bin/bash

# 快速测试8:2数据划分的分类任务

export CUDA_VISIBLE_DEVICES=0

echo "=== 8:2数据划分快速测试 ==="

# 首先创建数据集划分
echo "步骤1: 创建8:2数据划分..."
if [ ! -d "./dataset/HandGrip_Split" ]; then
    echo "正在创建数据集划分..."
    bash ./scripts/data_processing/create_handgrip_splits.sh
else
    echo "数据集划分已存在，跳过创建步骤"
fi

# 快速测试健康状态分类
echo ""
echo "步骤2: 快速测试健康状态分类 (1个epoch)..."

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/HandGrip_Split/HandGrip_health_status/ \
  --data_path HandGrip_health_status_TRAIN.ts \
  --model_id HandGrip_test_8_2 \
  --model TimesNet \
  --data UEA \
  --e_layers 2 \
  --batch_size 8 \
  --d_model 32 \
  --d_ff 64 \
  --top_k 2 \
  --num_kernels 3 \
  --des 'QuickTest_8_2_Split' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 1 \
  --patience 5

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 8:2划分测试成功！"
    echo ""
    echo "数据集信息:"
    echo "- 训练集: 80%"
    echo "- 测试集: 20% (同时用作验证集)"
    echo ""
    echo "现在可以运行完整训练:"
    echo "bash ./scripts/classification/HandGrip_Split_Training.sh"
    echo ""
    echo "或者进行多模型对比:"
    echo "bash ./scripts/classification/HandGrip_MultiModel_Comparison_8_2.sh"
else
    echo ""
    echo "❌ 测试失败，请检查数据集格式或配置"
fi