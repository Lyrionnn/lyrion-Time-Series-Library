#!/usr/bin/env python3
"""
为UEA格式修复后的HandGrip数据集创建一致性8:2划分
不依赖sktime，直接解析.ts文件
"""

import os
import numpy as np
import argparse
from pathlib import Path
import re

def parse_ts_file_simple(filepath):
    """
    简单解析.ts文件，提取数据和标签
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 跳过头部，找到@data行
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() == '@data':
            data_start = i + 1
            break
    
    # 解析数据行
    X = []
    y = []
    
    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue
            
        # 格式: data:label
        if ':' in line:
            data_part, label = line.rsplit(':', 1)
            X.append(data_part.strip())
            y.append(label.strip())
        else:
            # 备用格式: data,label (最后一个是标签)
            parts = line.split(',')
            if len(parts) > 1:
                data_part = ','.join(parts[:-1])
                label = parts[-1].strip()
                X.append(data_part)
                y.append(label)
    
    return X, y

def create_stratified_split(y, train_ratio=0.8, random_state=42):
    """
    创建分层划分的索引
    """
    np.random.seed(random_state)
    
    # 获取唯一标签和它们的索引
    unique_labels = list(set(y))
    train_indices = []
    test_indices = []
    
    for label in unique_labels:
        # 找到该标签的所有索引
        label_indices = [i for i, l in enumerate(y) if l == label]
        n_label = len(label_indices)
        
        # 计算训练集大小
        n_train = int(n_label * train_ratio)
        
        # 随机打乱该标签的索引
        np.random.shuffle(label_indices)
        
        # 分配到训练集和测试集
        train_indices.extend(label_indices[:n_train])
        test_indices.extend(label_indices[n_train:])
    
    return sorted(train_indices), sorted(test_indices)

def create_split_file(input_file, output_dir, train_indices, test_indices):
    """
    根据索引创建训练集和测试集文件
    """
    # 读取原始文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 分离头部和数据
    header_lines = []
    data_lines = []
    data_start = 0
    
    for i, line in enumerate(lines):
        if line.strip() == '@data':
            header_lines = lines[:i+1]
            data_start = i + 1
            break
    
    # 收集所有数据行
    for line in lines[data_start:]:
        line = line.strip()
        if line:
            data_lines.append(line)
    
    # 创建输出目录
    dataset_name = Path(input_file).stem
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建训练集文件
    train_file = dataset_dir / f"{dataset_name}_TRAIN.ts"
    with open(train_file, 'w', encoding='utf-8') as f:
        # 写入头部
        for header_line in header_lines:
            f.write(header_line)
        
        # 写入训练数据
        for idx in train_indices:
            if idx < len(data_lines):
                f.write(data_lines[idx] + '\n')
    
    # 创建测试集文件
    test_file = dataset_dir / f"{dataset_name}_TEST.ts"
    with open(test_file, 'w', encoding='utf-8') as f:
        # 写入头部
        for header_line in header_lines:
            f.write(header_line)
        
        # 写入测试数据
        for idx in test_indices:
            if idx < len(data_lines):
                f.write(data_lines[idx] + '\n')
    
    print(f"✅ {dataset_name}: 训练集 {len(train_indices)} 样本, 测试集 {len(test_indices)} 样本")
    return train_file, test_file

def main():
    parser = argparse.ArgumentParser(description='为UEA格式HandGrip数据创建一致性划分')
    parser.add_argument('--input_dir', 
                       default='./dataset/HandGrip_UEA_Fixed',
                       help='UEA格式修复后的输入目录')
    parser.add_argument('--output_dir', 
                       default='./dataset/HandGrip_Consistent_Split',
                       help='一致性划分输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--random_state', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 需要处理的文件列表
    files_to_process = [
        'HandGrip_health_status.ts',
        'HandGrip_grip_count_level.ts', 
        'HandGrip_motion_quality.ts',
        'HandGrip_grip_frequency_level.ts'
    ]
    
    print("开始创建UEA格式一致性8:2数据划分...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"训练集比例: {args.train_ratio}")
    print(f"随机种子: {args.random_state}")
    print("-" * 50)
    
    # 使用第一个文件（健康状态）创建基准划分
    base_file = input_dir / files_to_process[0]
    if not base_file.exists():
        print(f"❌ 基准文件不存在: {base_file}")
        return
    
    print(f"基于 {files_to_process[0]} 创建一致性划分索引...")
    
    # 解析基准文件获取标签
    X_base, y_base = parse_ts_file_simple(base_file)
    print(f"基准文件样本数: {len(y_base)}")
    
    # 分析类别分布
    from collections import Counter
    label_counts = Counter(y_base)
    print(f"类别分布: {dict(label_counts)}")
    
    # 创建分层划分
    train_indices, test_indices = create_stratified_split(
        y_base, 
        train_ratio=args.train_ratio, 
        random_state=args.random_state
    )
    
    print(f"训练集样本数: {len(train_indices)}")
    print(f"测试集样本数: {len(test_indices)}")
    
    # 保存划分索引
    indices_file = output_dir / 'split_indices.npz'
    np.savez(indices_file, 
             train_indices=np.array(train_indices),
             test_indices=np.array(test_indices))
    print(f"划分索引已保存: {indices_file}")
    
    print("-" * 30)
    
    # 对所有文件应用相同的划分
    for filename in files_to_process:
        input_file = input_dir / filename
        
        if input_file.exists():
            try:
                # 验证样本数量一致性
                X_current, y_current = parse_ts_file_simple(input_file)
                if len(y_current) != len(y_base):
                    print(f"⚠️  {filename} 样本数量不一致: {len(y_current)} vs {len(y_base)}")
                
                # 创建划分文件
                train_file, test_file = create_split_file(
                    input_file, output_dir, train_indices, test_indices
                )
                
            except Exception as e:
                print(f"❌ {filename} 处理失败: {e}")
        else:
            print(f"⚠️  文件不存在: {input_file}")
    
    print("-" * 50)
    print("✅ 一致性数据划分创建完成!")
    print(f"所有文件保存在: {output_dir}")
    print("")
    print("验证划分结果:")
    
    # 验证划分一致性
    for filename in files_to_process:
        dataset_name = Path(filename).stem
        train_file = output_dir / dataset_name / f"{dataset_name}_TRAIN.ts"
        test_file = output_dir / dataset_name / f"{dataset_name}_TEST.ts"
        
        if train_file.exists() and test_file.exists():
            # 检查文件大小
            _, y_train = parse_ts_file_simple(train_file)
            _, y_test = parse_ts_file_simple(test_file)
            
            train_counts = Counter(y_train)
            test_counts = Counter(y_test)
            
            print(f"  {dataset_name}:")
            print(f"    训练集: {len(y_train)} 样本, 分布: {dict(train_counts)}")
            print(f"    测试集: {len(y_test)} 样本, 分布: {dict(test_counts)}")
    
    print("")
    print("现在可以运行一致性训练:")
    print("bash ./scripts/classification/HandGrip_Consistent_Training.sh")

if __name__ == "__main__":
    main()