#!/bin/bash

# 手部抓握时间序列分类任务
# 基于您设计的多任务分类框架

# GPU设置
export CUDA_VISIBLE_DEVICES=0

echo "开始手部抓握分类任务训练..."

# 任务1: 健康状态分类 (主任务: 正常人 vs 患者)
echo "=== 训练任务1: 健康状态分类 ==="
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/HandGrip_Enhanced/ \
  --data_path HandGrip_health_status.ts \
  --model_id HandGrip_health_status \
  --model TimesNet \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 128 \
  --top_k 3 \
  --num_kernels 6 \
  --des 'HealthStatus_Binary_Classification' \
  --itr 5 \
  --learning_rate 0.001 \
  --train_epochs 50 \
  --patience 15

echo "=== 训练任务2: 抓握次数分类 ==="
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/HandGrip_Enhanced/ \
  --data_path HandGrip_grip_count_level.ts \
  --model_id HandGrip_grip_count \
  --model TimesNet \
  --data UEA \
  --e_layers 2 \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 64 \
  --top_k 3 \
  --num_kernels 4 \
  --des 'GripCount_MultiClass' \
  --itr 5 \
  --learning_rate 0.001 \
  --train_epochs 40 \
  --patience 12

echo "=== 训练任务3: 运动质量评估 ==="
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/HandGrip_Enhanced/ \
  --data_path HandGrip_motion_quality.ts \
  --model_id HandGrip_motion_quality \
  --model TimesNet \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 48 \
  --d_ff 96 \
  --top_k 2 \
  --num_kernels 5 \
  --des 'MotionQuality_Assessment' \
  --itr 5 \
  --learning_rate 0.0008 \
  --train_epochs 45 \
  --patience 12

echo "=== 训练任务4: 抓握频率分析 ==="
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/HandGrip_Enhanced/ \
  --data_path HandGrip_grip_frequency_level.ts \
  --model_id HandGrip_frequency \
  --model TimesNet \
  --data UEA \
  --e_layers 2 \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 64 \
  --top_k 3 \
  --num_kernels 4 \
  --des 'FrequencyLevel_Analysis' \
  --itr 5 \
  --learning_rate 0.001 \
  --train_epochs 35 \
  --patience 10

echo "所有手部抓握分类任务训练完成！"
echo "检查结果: ./results/ 目录下查看各任务性能"