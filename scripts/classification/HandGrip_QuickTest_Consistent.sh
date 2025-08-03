#!/bin/bash

# 一致性数据划分快速测试脚本

export CUDA_VISIBLE_DEVICES=0

echo "=== 一致性数据划分快速测试 ==="

# 首先修复UEA格式
echo "步骤1: 修复UEA时间序列格式..."
if [ ! -d "./dataset/HandGrip_UEA_Fixed" ]; then
    echo "正在修复UEA格式问题..."
    python utils/fix_uea_format.py \
        --input_dir "./dataset/HandGrip_Enhanced" \
        --output_dir "./dataset/HandGrip_UEA_Fixed"
else
    echo "UEA格式修复文件已存在"
fi

echo ""
echo "步骤2: 快速测试UEA格式修复后的健康状态分类 (1个epoch)..."

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/HandGrip_UEA_Fixed/ \
  --data_path HandGrip_health_status.ts \
  --model_id HandGrip_test_uea_fixed \
  --model TimesNet \
  --data UEA \
  --e_layers 2 \
  --batch_size 8 \
  --d_model 32 \
  --d_ff 64 \
  --top_k 2 \
  --num_kernels 3 \
  --des 'QuickTest_UEA_Format_Fixed' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 1 \
  --patience 5

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ UEA格式修复测试成功！"
    echo ""
    echo "重要特性:"
    echo "✅ 修复了UEA时间序列格式头部信息"
    echo "✅ 解决了sktime解析错误"
    echo "✅ 数据可以正常加载和训练"
    echo ""
    echo "数据文件结构:"
    echo "$(ls -la ./dataset/HandGrip_UEA_Fixed/)"
    echo ""
    echo "下一步: 创建一致性数据划分和完整训练"
    echo "# 需要基于修复后的UEA格式创建8:2划分"
else
    echo ""
    echo "❌ 测试失败，请检查数据集格式或配置"
fi