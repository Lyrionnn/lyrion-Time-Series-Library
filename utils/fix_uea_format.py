#!/usr/bin/env python3
"""
修复UEA时间序列格式，确保sktime能正确解析头部信息
解决 TypeError: 'NoneType' object cannot be interpreted as an integer 错误
"""

import os
import re
import argparse
from pathlib import Path

def fix_uea_header_format(input_file, output_file):
    """
    修复UEA格式的头部信息，确保符合sktime标准
    """
    print(f"正在修复UEA格式: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return False
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return False
    
    print(f"  文件行数: {len(lines)}")
    
    # 标准UEA头部格式
    header_template = """@problemName {problem_name}
@timeStamps false
@missing false
@univariate false
@dimensions {dimensions}
@equalLength true
@seriesLength {series_length}
@classLabel true {class_labels}
@data
"""
    
    # 解析现有头部信息
    problem_name = "HandGrip_Dataset"
    dimensions = 20
    series_length = 382
    class_labels = "0 1"
    
    data_start_idx = 0
    
    for i, line in enumerate(lines):
        line_strip = line.strip()
        if line_strip.startswith('@problemName'):
            problem_name = line_strip.split()[1] if len(line_strip.split()) > 1 else problem_name
        elif line_strip.startswith('@dimensions'):
            try:
                dimensions = int(line_strip.split()[1])
            except:
                dimensions = 20
        elif line_strip.startswith('@seriesLength'):
            try:
                series_length = int(line_strip.split()[1])
            except:
                series_length = 382
        elif line_strip.startswith('@classLabel'):
            # 提取类标签信息
            parts = line_strip.split()
            if len(parts) >= 4:  # @classLabel true 0 1
                class_labels = ' '.join(parts[2:])
        elif line_strip.startswith('@data') or (not line_strip.startswith('@') and line_strip):
            data_start_idx = i
            break
    
    # 收集所有数据行（跳过重复的@data行）
    data_lines = []
    for i in range(data_start_idx, len(lines)):
        line = lines[i].strip()
        if line and not line.startswith('@'):
            data_lines.append(line)
    
    print(f"  数据开始行: {data_start_idx}")
    print(f"  数据行数: {len(data_lines)}")
    
    if len(data_lines) == 0:
        print("❌ 没有找到有效的数据行")
        return False
    
    # 分析实际的类标签
    actual_labels = set()
    for line in data_lines[:10]:  # 只检查前10行以提高速度
        if ':' in line:
            # 格式: data:label
            label = line.split(':')[-1].strip()
            actual_labels.add(label)
        else:
            # 格式可能是 data,label （最后一个是标签）
            parts = line.split(',')
            if parts:
                label = parts[-1].strip()
                # 检查是否是数字标签
                try:
                    float(label)
                    actual_labels.add(label)
                except:
                    pass
    
    # 更新类标签
    if actual_labels:
        sorted_labels = sorted(actual_labels, key=lambda x: float(x) if x.replace('.','').replace('-','').isdigit() else float('inf'))
        class_labels = ' '.join(sorted_labels)
    
    print(f"  - 问题名称: {problem_name}")
    print(f"  - 维度数: {dimensions}")
    print(f"  - 序列长度: {series_length}")
    print(f"  - 类标签: {class_labels}")
    
    # 生成标准头部
    header = header_template.format(
        problem_name=problem_name,
        dimensions=dimensions,
        series_length=series_length,
        class_labels=class_labels
    )
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 写入修复后的文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(header)
            for line in data_lines:
                f.write(line + '\n')
        
        print(f"✅ UEA格式修复完成: {output_file}")
        return True
    except Exception as e:
        print(f"❌ 写入文件失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='修复UEA时间序列格式')
    parser.add_argument('--input_dir', 
                       default='./dataset/HandGrip_Enhanced',
                       help='输入数据目录')
    parser.add_argument('--output_dir', 
                       default='./dataset/HandGrip_UEA_Fixed',
                       help='输出数据目录')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # 检查输入目录是否存在
    if not input_dir.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有.ts文件
    ts_files = list(input_dir.glob("*.ts"))
    
    if not ts_files:
        print(f"❌ 在输入目录中没有找到.ts文件: {input_dir}")
        print("可用文件:")
        for file in input_dir.iterdir():
            if file.is_file():
                print(f"  {file.name}")
        return
    
    print("开始修复UEA时间序列格式...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"找到 {len(ts_files)} 个.ts文件")
    print("-" * 50)
    
    success_count = 0
    
    for input_file in ts_files:
        output_file = output_dir / input_file.name
        
        try:
            if fix_uea_header_format(str(input_file), str(output_file)):
                print(f"✅ {input_file.name} 修复成功")
                success_count += 1
            else:
                print(f"❌ {input_file.name} 修复失败")
        except Exception as e:
            print(f"❌ {input_file.name} 修复失败: {e}")
        
        print("-" * 30)
    
    print(f"UEA格式修复完成! 成功: {success_count}/{len(ts_files)}")
    
    if success_count > 0:
        print(f"修复后的文件保存在: {output_dir}")
        print("\n可用的修复后文件:")
        for file in output_dir.glob("*.ts"):
            print(f"  {file.name}")
        print("\n下一步: 使用修复后的数据进行训练")
        print("bash ./scripts/classification/HandGrip_Debug_Test.sh")
    else:
        print("❌ 没有文件修复成功，请检查输入数据格式")

if __name__ == "__main__":
    main()