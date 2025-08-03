#!/bin/bash

# 为HandGrip数据集创建一致性8:2划分
# 确保四个分类任务使用完全相同的样本划分

echo "=== HandGrip数据集一致性8:2划分工具 ==="

# 源数据目录
SOURCE_DIR="./dataset/HandGrip_Enhanced"
# 输出目录 
OUTPUT_DIR="./dataset/HandGrip_Consistent_Split"

echo "重要提醒: 四个分类任务将使用完全相同的样本划分"
echo "这确保了结果的可比性和一致性"
echo ""

# 使用一致性划分工具
echo "开始创建一致性划分..."

python utils/create_consistent_splits.py \
    --base_data_dir "$SOURCE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --train_ratio 0.8 \
    --test_ratio 0.2 \
    --random_state 42

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 一致性划分创建成功"
    echo ""
    echo "数据集保存位置: $OUTPUT_DIR"
    echo ""
    echo "验证划分一致性:"
    python utils/create_consistent_splits.py \
        --output_dir "$OUTPUT_DIR" \
        --verify_only
    echo ""
    echo "现在可以使用以下命令训练模型:"
    echo "bash ./scripts/classification/HandGrip_Consistent_Training.sh"
else
    echo "❌ 一致性划分创建失败"
    exit 1
fi