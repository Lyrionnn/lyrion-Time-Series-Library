#!/bin/bash

# 手部抓握时间序列分类任务 - 8:2数据划分版本
# 训练集80%，测试集20%（也作为验证集）

# GPU设置
export CUDA_VISIBLE_DEVICES=0

echo "开始手部抓握分类任务训练 (8:2数据划分)..."

# 数据根目录
DATA_ROOT="./dataset/HandGrip_Split"

# 任务1: 健康状态分类 (主任务: 正常人 vs 患者)
echo "=== 训练任务1: 健康状态分类 (8:2划分) ==="
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path $DATA_ROOT/HandGrip_health_status/ \
  --data_path HandGrip_health_status_TRAIN.ts \
  --model_id HandGrip_health_status_8_2 \
  --model TimesNet \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 128 \
  --top_k 3 \
  --num_kernels 6 \
  --des 'HealthStatus_Binary_80_20_Split' \
  --itr 5 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 15

echo "=== 训练任务2: 抓握次数分类 (8:2划分) ==="
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path $DATA_ROOT/HandGrip_grip_count_level/ \
  --data_path HandGrip_grip_count_level_TRAIN.ts \
  --model_id HandGrip_grip_count_8_2 \
  --model TimesNet \
  --data UEA \
  --e_layers 2 \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 64 \
  --top_k 3 \
  --num_kernels 4 \
  --des 'GripCount_MultiClass_80_20_Split' \
  --itr 5 \
  --learning_rate 0.001 \
  --train_epochs 40 \
  --patience 12

echo "=== 训练任务3: 运动质量评估 (8:2划分) ==="
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path $DATA_ROOT/HandGrip_motion_quality/ \
  --data_path HandGrip_motion_quality_TRAIN.ts \
  --model_id HandGrip_motion_quality_8_2 \
  --model TimesNet \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 48 \
  --d_ff 96 \
  --top_k 2 \
  --num_kernels 5 \
  --des 'MotionQuality_Assessment_80_20_Split' \
  --itr 5 \
  --learning_rate 0.0008 \
  --train_epochs 45 \
  --patience 12

echo "=== 训练任务4: 抓握频率分析 (8:2划分) ==="
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path $DATA_ROOT/HandGrip_grip_frequency_level/ \
  --data_path HandGrip_grip_frequency_level_TRAIN.ts \
  --model_id HandGrip_frequency_8_2 \
  --model TimesNet \
  --data UEA \
  --e_layers 2 \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 64 \
  --top_k 3 \
  --num_kernels 4 \
  --des 'FrequencyLevel_Analysis_80_20_Split' \
  --itr 5 \
  --learning_rate 0.001 \
  --train_epochs 35 \
  --patience 10

echo "=== 所有任务训练完成 (8:2划分) ==="
echo "结果保存在: ./results/classification/"
echo ""
echo "数据集统计:"
echo "- 训练集: 80%"
echo "- 测试集: 20% (同时用作验证集)"
echo "- 这符合时间序列分类任务的常见做法"