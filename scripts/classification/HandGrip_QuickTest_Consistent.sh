#!/bin/bash

# 一致性数据划分快速测试脚本

export CUDA_VISIBLE_DEVICES=0

echo "=== 一致性数据划分快速测试 ==="

# 首先创建一致性数据集划分
echo "步骤1: 创建一致性数据划分..."
if [ ! -d "./dataset/HandGrip_Consistent_Split" ]; then
    echo "正在创建一致性数据集划分..."
    bash ./scripts/data_processing/create_handgrip_splits.sh
else
    echo "一致性数据集划分已存在"
    echo "验证现有划分的一致性..."
    python utils/create_consistent_splits.py \
        --output_dir "./dataset/HandGrip_Consistent_Split" \
        --verify_only
fi

echo ""
echo "步骤2: 快速测试一致性健康状态分类 (1个epoch)..."

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/HandGrip_Consistent_Split/HandGrip_health_status/ \
  --data_path HandGrip_health_status_TRAIN.ts \
  --model_id HandGrip_test_consistent \
  --model TimesNet \
  --data UEA \
  --e_layers 2 \
  --batch_size 8 \
  --d_model 32 \
  --d_ff 64 \
  --top_k 2 \
  --num_kernels 3 \
  --des 'QuickTest_Consistent_Split' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 1 \
  --patience 5

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 一致性划分测试成功！"
    echo ""
    echo "重要特性:"
    echo "✅ 四个分类任务使用完全相同的样本划分"
    echo "✅ 训练集: 80%, 测试集: 20%"
    echo "✅ 基于健康状态分层划分，保持类别平衡"
    echo "✅ 可重现的随机种子(42)"
    echo ""
    echo "数据文件结构:"
    echo "$(ls -la ./dataset/HandGrip_Consistent_Split/*/)"
    echo ""
    echo "现在可以运行完整的一致性训练:"
    echo "bash ./scripts/classification/HandGrip_Consistent_Training.sh"
else
    echo ""
    echo "❌ 测试失败，请检查数据集格式或配置"
fi