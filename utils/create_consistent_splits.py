#!/usr/bin/env python3
"""
为HandGrip数据集创建一致的训练/测试划分
确保四个分类任务使用完全相同的样本划分
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import argparse
from sktime.datasets import load_from_tsfile_to_dataframe

def create_consistent_splits(base_data_dir, output_dir, train_ratio=0.8, test_ratio=0.2, random_state=42):
    """
    为所有HandGrip分类任务创建一致的数据划分
    
    Args:
        base_data_dir: 包含所有.ts文件的输入目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        test_ratio: 测试集比例
        random_state: 随机种子
    """
    
    print("=== HandGrip数据集一致性划分工具 ===")
    print(f"数据划分比例 - 训练集:{train_ratio:.1%}, 测试集:{test_ratio:.1%}")
    print(f"随机种子: {random_state}")
    
    # 数据集文件列表
    datasets = [
        "HandGrip_health_status",
        "HandGrip_grip_count_level", 
        "HandGrip_motion_quality",
        "HandGrip_grip_frequency_level"
    ]
    
    # 首先读取所有数据集，检查样本数量一致性
    print("\n步骤1: 检查数据集一致性...")
    dataset_info = {}
    
    for dataset_name in datasets:
        data_file = os.path.join(base_data_dir, f"{dataset_name}.ts")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"数据文件不存在: {data_file}")
        
        print(f"  读取: {dataset_name}")
        X, y = load_from_tsfile_to_dataframe(data_file, return_separate_X_and_y=True)
        
        dataset_info[dataset_name] = {
            'X': X,
            'y': y,
            'n_samples': len(X),
            'n_features': X.shape[1] if hasattr(X, 'shape') else len(X.columns),
            'classes': sorted(set(y))
        }
        
        print(f"    样本数: {len(X)}, 特征数: {dataset_info[dataset_name]['n_features']}")
        print(f"    类别: {dataset_info[dataset_name]['classes']}")
    
    # 检查所有数据集的样本数是否一致
    sample_counts = [info['n_samples'] for info in dataset_info.values()]
    if len(set(sample_counts)) != 1:
        raise ValueError(f"数据集样本数不一致: {sample_counts}")
    
    n_samples = sample_counts[0]
    print(f"\n✅ 所有数据集样本数一致: {n_samples}")
    
    # 步骤2: 创建统一的样本索引划分
    print(f"\n步骤2: 创建统一的样本索引划分...")
    
    # 使用第一个数据集的标签进行分层划分（保证类别平衡）
    base_dataset = datasets[0]  # 使用健康状态作为基准
    base_y = dataset_info[base_dataset]['y']
    
    # 创建标签编码器
    le = LabelEncoder()
    base_y_encoded = le.fit_transform(base_y)
    
    # 创建样本索引
    sample_indices = np.arange(n_samples)
    
    # 进行分层划分
    train_indices, test_indices = train_test_split(
        sample_indices, 
        test_size=test_ratio,
        random_state=random_state,
        stratify=base_y_encoded  # 根据健康状态进行分层
    )
    
    print(f"  训练集索引数: {len(train_indices)}")
    print(f"  测试集索引数: {len(test_indices)}")
    
    # 验证划分结果
    train_labels = base_y_encoded[train_indices]
    test_labels = base_y_encoded[test_indices]
    
    print(f"  训练集类别分布: {np.bincount(train_labels)}")
    print(f"  测试集类别分布: {np.bincount(test_labels)}")
    
    # 步骤3: 为每个数据集应用相同的划分
    print(f"\n步骤3: 应用统一划分到所有数据集...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存索引文件
    indices_file = os.path.join(output_dir, "split_indices.npz")
    np.savez(indices_file, 
             train_indices=train_indices, 
             test_indices=test_indices,
             random_state=random_state,
             train_ratio=train_ratio,
             test_ratio=test_ratio)
    print(f"  索引文件已保存: {indices_file}")
    
    for dataset_name in datasets:
        print(f"\n  处理数据集: {dataset_name}")
        
        X = dataset_info[dataset_name]['X']
        y = dataset_info[dataset_name]['y']
        
        # 应用相同的索引划分
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_train = [y[i] for i in train_indices]
        y_test = [y[i] for i in test_indices]
        
        # 创建数据集特定的输出目录
        dataset_output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # 保存训练集和测试集
        train_file = save_uea_format(X_train, y_train, 
                                   os.path.join(dataset_output_dir, f"{dataset_name}_TRAIN.ts"))
        test_file = save_uea_format(X_test, y_test, 
                                  os.path.join(dataset_output_dir, f"{dataset_name}_TEST.ts"))
        
        print(f"    训练集: {train_file}")
        print(f"    测试集: {test_file}")
        
        # 验证类别分布
        y_train_series = pd.Series(y_train)
        y_test_series = pd.Series(y_test)
        print(f"    训练集类别分布: {y_train_series.value_counts().to_dict()}")
        print(f"    测试集类别分布: {y_test_series.value_counts().to_dict()}")
    
    print(f"\n=== 一致性划分完成 ===")
    print(f"所有数据集使用相同的样本划分")
    print(f"输出目录: {output_dir}")
    print(f"随机种子: {random_state} (可重现)")
    
    return output_dir

def save_uea_format(X_data, y_data, filepath):
    """保存为UEA格式"""
    
    # 获取所有唯一类别
    unique_classes = sorted(set(y_data))
    
    with open(filepath, 'w') as f:
        # 写入类别信息（可选，某些情况下需要）
        
        # 写入每个样本
        for i in range(len(X_data)):
            # 获取时间序列数据
            sample_data = []
            for col in X_data.columns:
                series = X_data.iloc[i][col]
                if hasattr(series, '__iter__'):
                    series_str = ','.join(map(str, series))
                else:
                    series_str = str(series)
                sample_data.append(series_str)
            
            # 写入格式: feature1:feature2:...:label
            line = ':'.join(sample_data) + ':' + str(y_data[i])
            f.write(line + '\n')
    
    return filepath

def verify_consistency(split_dir):
    """验证所有数据集的划分一致性"""
    
    print("\n=== 验证划分一致性 ===")
    
    # 加载索引文件
    indices_file = os.path.join(split_dir, "split_indices.npz")
    if not os.path.exists(indices_file):
        print("❌ 索引文件不存在，无法验证")
        return False
    
    indices_data = np.load(indices_file)
    train_indices = indices_data['train_indices']
    test_indices = indices_data['test_indices']
    
    print(f"基准划分 - 训练集: {len(train_indices)}, 测试集: {len(test_indices)}")
    
    # 检查各数据集
    datasets = [
        "HandGrip_health_status",
        "HandGrip_grip_count_level", 
        "HandGrip_motion_quality",
        "HandGrip_grip_frequency_level"
    ]
    
    all_consistent = True
    
    for dataset_name in datasets:
        dataset_dir = os.path.join(split_dir, dataset_name)
        train_file = os.path.join(dataset_dir, f"{dataset_name}_TRAIN.ts")
        test_file = os.path.join(dataset_dir, f"{dataset_name}_TEST.ts")
        
        if os.path.exists(train_file) and os.path.exists(test_file):
            # 简单计数验证
            with open(train_file, 'r') as f:
                train_count = len(f.readlines())
            with open(test_file, 'r') as f:
                test_count = len(f.readlines())
            
            consistent = (train_count == len(train_indices) and test_count == len(test_indices))
            status = "✅" if consistent else "❌"
            print(f"{status} {dataset_name}: 训练{train_count}, 测试{test_count}")
            
            if not consistent:
                all_consistent = False
        else:
            print(f"❌ {dataset_name}: 文件缺失")
            all_consistent = False
    
    if all_consistent:
        print("✅ 所有数据集划分一致")
    else:
        print("❌ 发现不一致的划分")
    
    return all_consistent

def main():
    parser = argparse.ArgumentParser(description='创建HandGrip数据集的一致性划分')
    parser.add_argument('--base_data_dir', type=str, 
                       default='./dataset/HandGrip_Enhanced',
                       help='包含原始.ts文件的目录')
    parser.add_argument('--output_dir', type=str,
                       default='./dataset/HandGrip_Consistent_Split',
                       help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.8, 
                       help='训练集比例')
    parser.add_argument('--test_ratio', type=float, default=0.2, 
                       help='测试集比例')
    parser.add_argument('--random_state', type=int, default=42, 
                       help='随机种子')
    parser.add_argument('--verify_only', action='store_true',
                       help='仅验证现有划分的一致性')
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_consistency(args.output_dir)
    else:
        create_consistent_splits(
            args.base_data_dir,
            args.output_dir,
            args.train_ratio,
            args.test_ratio,
            args.random_state
        )
        
        # 自动验证
        verify_consistency(args.output_dir)

if __name__ == "__main__":
    main()