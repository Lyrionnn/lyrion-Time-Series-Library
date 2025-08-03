#!/bin/bash

# 手部抓握分类 - 多模型对比实验
# 对比TimesNet, iTransformer, PatchTST等模型的性能

export CUDA_VISIBLE_DEVICES=0

# 数据集路径
DATA_ROOT="./dataset/HandGrip_Enhanced/"

# 模型列表
models=("TimesNet" "iTransformer" "PatchTST" "MICN")

# 针对健康状态分类任务进行多模型对比
echo "=== 健康状态分类任务 - 多模型对比 ==="

for model in "${models[@]}"; do
    echo "训练模型: $model"
    
    if [ "$model" = "TimesNet" ]; then
        # TimesNet 专门参数
        python -u run.py \
          --task_name classification \
          --is_training 1 \
          --root_path $DATA_ROOT \
          --data_path HandGrip_health_status.ts \
          --model_id HandGrip_health_${model} \
          --model $model \
          --data UEA \
          --e_layers 3 \
          --batch_size 16 \
          --d_model 64 \
          --d_ff 128 \
          --top_k 3 \
          --num_kernels 6 \
          --des "HealthStatus_${model}" \
          --itr 3 \
          --learning_rate 0.001 \
          --train_epochs 50 \
          --patience 15
          
    elif [ "$model" = "iTransformer" ]; then
        # iTransformer 专门参数
        python -u run.py \
          --task_name classification \
          --is_training 1 \
          --root_path $DATA_ROOT \
          --data_path HandGrip_health_status.ts \
          --model_id HandGrip_health_${model} \
          --model $model \
          --data UEA \
          --e_layers 3 \
          --d_layers 1 \
          --batch_size 16 \
          --d_model 64 \
          --d_ff 128 \
          --factor 1 \
          --des "HealthStatus_${model}" \
          --itr 3 \
          --learning_rate 0.0005 \
          --train_epochs 50 \
          --patience 15
          
    elif [ "$model" = "PatchTST" ]; then
        # PatchTST 专门参数
        python -u run.py \
          --task_name classification \
          --is_training 1 \
          --root_path $DATA_ROOT \
          --data_path HandGrip_health_status.ts \
          --model_id HandGrip_health_${model} \
          --model $model \
          --data UEA \
          --e_layers 3 \
          --d_layers 1 \
          --batch_size 16 \
          --d_model 64 \
          --d_ff 128 \
          --patch_len 16 \
          --stride 8 \
          --des "HealthStatus_${model}" \
          --itr 3 \
          --learning_rate 0.001 \
          --train_epochs 50 \
          --patience 15
          
    elif [ "$model" = "MICN" ]; then
        # MICN 专门参数
        python -u run.py \
          --task_name classification \
          --is_training 1 \
          --root_path $DATA_ROOT \
          --data_path HandGrip_health_status.ts \
          --model_id HandGrip_health_${model} \
          --model $model \
          --data UEA \
          --e_layers 2 \
          --batch_size 16 \
          --d_model 32 \
          --d_ff 64 \
          --conv_kernel [2,4,8] \
          --des "HealthStatus_${model}" \
          --itr 3 \
          --learning_rate 0.001 \
          --train_epochs 50 \
          --patience 15
    fi
    
    echo "模型 $model 训练完成"
done

echo "=== 多模型对比实验完成 ==="
echo "查看结果对比: ls ./results/classification/HandGrip_health_*/"