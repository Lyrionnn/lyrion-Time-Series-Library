#!/bin/bash

# HandGrip数据集UEA格式修复和快速测试脚本
# 解决 TypeError: 'NoneType' object cannot be interpreted as an integer

export CUDA_VISIBLE_DEVICES=0

echo "=== HandGrip UEA格式修复和测试 ==="

# 步骤1: 修复UEA格式
echo "步骤1: 修复UEA时间序列格式..."
python utils/fix_uea_format.py \
    --input_dir "./dataset/HandGrip_Enhanced" \
    --output_dir "./dataset/HandGrip_UEA_Fixed"

if [ $? -ne 0 ]; then
    echo "❌ UEA格式修复失败"
    exit 1
fi

echo ""
echo "步骤2: 快速测试修复后的数据 (1个epoch)..."

# 使用修复后的UEA格式数据进行快速测试
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/HandGrip_UEA_Fixed/ \
  --data_path HandGrip_health_status.ts \
  --model_id HandGrip_uea_test \
  --model TimesNet \
  --data UEA \
  --e_layers 2 \
  --batch_size 8 \
  --d_model 32 \
  --d_ff 64 \
  --top_k 2 \
  --num_kernels 3 \
  --des 'UEA_Format_Fixed_Test' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 1 \
  --patience 5

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ UEA格式修复测试成功！"
    echo ""
    echo "步骤3: 创建一致性8:2数据划分..."
    
    # 创建一致性划分
    python utils/create_uea_consistent_splits.py \
        --input_dir "./dataset/HandGrip_UEA_Fixed" \
        --output_dir "./dataset/HandGrip_Consistent_Split" \
        --train_ratio 0.8 \
        --random_state 42
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 一致性划分创建成功！"
        echo ""
        echo "重要特性:"
        echo "✅ 修复了UEA时间序列格式头部信息"
        echo "✅ 解决了sktime解析错误"
        echo "✅ 四个分类任务使用完全相同的样本划分"
        echo "✅ 训练集: 80%, 测试集: 20%"
        echo "✅ 基于健康状态分层划分，保持类别平衡"
        echo ""
        echo "数据结构:"
        echo "$(ls -la ./dataset/HandGrip_Consistent_Split/)"
        echo ""
        echo "现在可以运行完整的一致性训练:"
        echo "bash ./scripts/classification/HandGrip_Consistent_Training.sh"
    else
        echo "❌ 一致性划分创建失败"
    fi
else
    echo ""
    echo "❌ UEA格式测试失败，请检查数据格式"
fi