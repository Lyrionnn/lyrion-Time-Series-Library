#!/bin/bash

# HandGrip数据集快速修复和测试脚本
# 解决文件路径和格式问题

export CUDA_VISIBLE_DEVICES=0

echo "=== HandGrip数据集快速修复和测试 ==="

# 检查原始数据是否存在
echo "步骤1: 检查数据文件..."
if [ ! -d "./dataset/HandGrip_Enhanced" ]; then
    echo "❌ 原始数据目录不存在: ./dataset/HandGrip_Enhanced"
    echo "请确保HandGrip数据集已正确放置"
    exit 1
fi

# 列出原始数据文件
echo "原始数据文件:"
ls -la ./dataset/HandGrip_Enhanced/

# 检查是否有.ts文件
TS_FILES=$(find ./dataset/HandGrip_Enhanced/ -name "*.ts" | wc -l)
if [ "$TS_FILES" -eq 0 ]; then
    echo "❌ 原始数据目录中没有找到.ts文件"
    echo "可用文件:"
    ls -la ./dataset/HandGrip_Enhanced/
    exit 1
fi

echo "找到 $TS_FILES 个.ts文件"

# 直接使用原始数据进行测试，不进行格式修复
echo ""
echo "步骤2: 使用原始数据进行快速测试..."

# 找到第一个.ts文件
FIRST_TS_FILE=$(find ./dataset/HandGrip_Enhanced/ -name "*.ts" | head -1)
FILENAME=$(basename "$FIRST_TS_FILE")

echo "使用文件: $FILENAME"
echo "完整路径: $FIRST_TS_FILE"

# 检查文件内容（前几行）
echo ""
echo "文件头部内容:"
head -10 "$FIRST_TS_FILE"

# 使用原始数据直接测试
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/HandGrip_Enhanced/ \
  --data_path "$FILENAME" \
  --model_id HandGrip_direct_test \
  --model TimesNet \
  --data UEA \
  --e_layers 2 \
  --batch_size 8 \
  --d_model 32 \
  --d_ff 64 \
  --top_k 2 \
  --num_kernels 3 \
  --des 'Direct_Test_Original_Data' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 1 \
  --patience 5

RESULT_CODE=$?

if [ $RESULT_CODE -eq 0 ]; then
    echo ""
    echo "✅ 直接使用原始数据测试成功！"
    echo "原始数据格式可以正常使用"
else
    echo ""
    echo "❌ 直接测试失败，开始修复数据格式..."
    echo ""
    
    # 如果直接测试失败，运行格式修复
    echo "步骤3: 修复UEA格式..."
    python utils/fix_uea_format.py \
        --input_dir "./dataset/HandGrip_Enhanced" \
        --output_dir "./dataset/HandGrip_UEA_Fixed"
    
    if [ $? -eq 0 ]; then
        echo "格式修复完成，重新测试..."
        
        # 使用修复后的数据测试
        python -u run.py \
          --task_name classification \
          --is_training 1 \
          --root_path ./dataset/HandGrip_UEA_Fixed/ \
          --data_path "$FILENAME" \
          --model_id HandGrip_fixed_test \
          --model TimesNet \
          --data UEA \
          --e_layers 2 \
          --batch_size 8 \
          --d_model 32 \
          --d_ff 64 \
          --top_k 2 \
          --num_kernels 3 \
          --des 'Fixed_Format_Test' \
          --itr 1 \
          --learning_rate 0.001 \
          --train_epochs 1 \
          --patience 5
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ 修复后数据测试成功！"
        else
            echo ""
            echo "❌ 修复后测试仍然失败"
            echo "请检查数据格式或联系支持"
        fi
    else
        echo "❌ 格式修复失败"
    fi
fi