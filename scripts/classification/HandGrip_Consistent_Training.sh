#!/bin/bash

# 手部抓握时间序列分类任务 - 一致性数据划分版本
# 四个任务使用完全相同的样本划分(训练集80%，测试集20%)

# GPU设置
export CUDA_VISIBLE_DEVICES=0

echo "开始手部抓握分类任务训练 (一致性数据划分)..."

# 数据根目录 - 使用一致性划分的数据
DATA_ROOT="./dataset/HandGrip_Consistent_Split"

# 验证数据划分一致性
echo "=== 验证数据划分一致性 ==="
python utils/create_consistent_splits.py \
    --output_dir "$DATA_ROOT" \
    --verify_only

if [ $? -ne 0 ]; then
    echo "❌ 数据划分验证失败，请先运行数据划分脚本"
    echo "bash ./scripts/data_processing/create_handgrip_splits.sh"
    exit 1
fi

echo ""
echo "✅ 数据划分一致性验证通过，开始训练..."
echo ""

# 任务1: 健康状态分类 (主任务: 正常人 vs 患者)
echo "=== 训练任务1: 健康状态分类 (一致性划分) ==="
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path $DATA_ROOT/HandGrip_health_status/ \
  --data_path HandGrip_health_status_TRAIN.ts \
  --model_id HandGrip_health_status_consistent \
  --model TimesNet \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 128 \
  --top_k 3 \
  --num_kernels 6 \
  --des 'HealthStatus_Binary_Consistent_Split' \
  --itr 5 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 15

echo ""
echo "=== 训练任务2: 抓握次数分类 (一致性划分) ==="
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path $DATA_ROOT/HandGrip_grip_count_level/ \
  --data_path HandGrip_grip_count_level_TRAIN.ts \
  --model_id HandGrip_grip_count_consistent \
  --model TimesNet \
  --data UEA \
  --e_layers 2 \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 64 \
  --top_k 3 \
  --num_kernels 4 \
  --des 'GripCount_MultiClass_Consistent_Split' \
  --itr 5 \
  --learning_rate 0.001 \
  --train_epochs 40 \
  --patience 12

echo ""
echo "=== 训练任务3: 运动质量评估 (一致性划分) ==="
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path $DATA_ROOT/HandGrip_motion_quality/ \
  --data_path HandGrip_motion_quality_TRAIN.ts \
  --model_id HandGrip_motion_quality_consistent \
  --model TimesNet \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 48 \
  --d_ff 96 \
  --top_k 2 \
  --num_kernels 5 \
  --des 'MotionQuality_Assessment_Consistent_Split' \
  --itr 5 \
  --learning_rate 0.0008 \
  --train_epochs 45 \
  --patience 12

echo ""
echo "=== 训练任务4: 抓握频率分析 (一致性划分) ==="
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path $DATA_ROOT/HandGrip_grip_frequency_level/ \
  --data_path HandGrip_grip_frequency_level_TRAIN.ts \
  --model_id HandGrip_frequency_consistent \
  --model TimesNet \
  --data UEA \
  --e_layers 2 \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 64 \
  --top_k 3 \
  --num_kernels 4 \
  --des 'FrequencyLevel_Analysis_Consistent_Split' \
  --itr 5 \
  --learning_rate 0.001 \
  --train_epochs 35 \
  --patience 10

echo ""
echo "=== 所有任务训练完成 (一致性划分) ==="
echo "结果保存在: ./results/classification/"
echo ""
echo "关键优势:"
echo "✅ 四个任务使用完全相同的样本划分"
echo "✅ 确保结果的可比性和一致性"
echo "✅ 训练集: 80%, 测试集: 20%"
echo "✅ 基于健康状态进行分层划分，保持类别平衡"
echo ""
echo "可以安全地比较四个任务的性能表现"