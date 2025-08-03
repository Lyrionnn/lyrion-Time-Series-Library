#!/usr/bin/env python3
"""
为HandGrip数据集创建训练/验证/测试划分的脚本
解决TSLib分类任务没有显式验证集的问题
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import argparse

def create_train_val_test_split(data_file, output_dir, 
                               train_ratio=0.8, test_ratio=0.2,
                               random_state=42):
    """
    将UEA格式的数据集划分为训练集(80%)和测试集(20%)
    验证集与测试集相同，符合时间序列分类任务惯例
    
    Args:
        data_file: 输入的.ts文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例 (默认80%)
        test_ratio: 测试集比例 (默认20%)，验证集与测试集相同
        random_state: 随机种子
    """
    
    print(f"正在处理数据集: {data_file}")
    print(f"数据划分比例 - 训练集:{train_ratio:.1%}, 测试集(也作验证集):{test_ratio:.1%}")
    
    # 确保比例加起来为1
    assert abs(train_ratio + test_ratio - 1.0) < 1e-6, "比例总和必须为1"
    
    # 读取UEA格式数据
    from sktime.datasets import load_from_tsfile_to_dataframe
    
    X, y = load_from_tsfile_to_dataframe(data_file, return_separate_X_and_y=True)
    
    print(f"数据集大小: {len(X)} 样本")
    print(f"类别分布: {pd.Series(y).value_counts().to_dict()}")
    
    # 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 划分为训练集(80%)和测试集(20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=test_ratio, 
        random_state=random_state,
        stratify=y_encoded  # 保持类别比例
    )
    
    print(f"划分结果:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集(也作验证集): {len(X_test)} 样本")
    print("  注意: 验证集与测试集相同，符合时间序列分类任务惯例")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存划分后的数据集
    base_name = os.path.splitext(os.path.basename(data_file))[0]
    
    def save_uea_format(X_data, y_data, filepath, class_names):
        """保存为UEA格式"""
        with open(filepath, 'w') as f:
            # 写入类别名称（第一行）
            f.write(' '.join(class_names) + '\n')
            
            # 写入每个样本
            for i in range(len(X_data)):
                # 获取类别名称
                class_name = class_names[y_data[i]]
                
                # 写入样本数据
                sample_data = []
                for col in X_data.columns:
                    series = X_data.iloc[i][col]
                    series_str = ','.join(map(str, series))
                    sample_data.append(series_str)
                
                f.write(f"{':'.join(sample_data)}:{class_name}\n")
    
    # 获取类别名称
    class_names = le.classes_
    
    # 保存训练集
    train_file = os.path.join(output_dir, f"{base_name}_TRAIN.ts")
    save_uea_format(X_train, y_train, train_file, class_names)
    
    # 保存测试集（同时也是验证集）
    test_file = os.path.join(output_dir, f"{base_name}_TEST.ts")
    save_uea_format(X_test, y_test, test_file, class_names)
    
    print(f"数据集已保存到 {output_dir}")
    print(f"训练文件: {train_file}")
    print(f"测试文件(也作验证): {test_file}")
    
    return train_file, test_file

def main():
    parser = argparse.ArgumentParser(description='划分HandGrip数据集为8:2比例')
    parser.add_argument('--data_file', type=str, required=True, help='输入的.ts文件')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例(默认0.8)')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='测试集比例(默认0.2)')
    parser.add_argument('--random_state', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    create_train_val_test_split(
        args.data_file, 
        args.output_dir,
        args.train_ratio,
        args.test_ratio,
        args.random_state
    )

if __name__ == "__main__":
    main()