import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

class HandGripClassificationDataset:
    """
    手部抓握时间序列分类数据集类
    支持多种分类任务设计
    """
    
    def __init__(self, csv_data_dir):
        self.csv_data_dir = csv_data_dir
        self.scaler = StandardScaler()
        
    def create_multi_task_labels(self, file_path, features):
        """
        创建多任务标签
        """
        labels = {}
        
        # 任务1: 健康状态分类 (主要任务)
        if 'normal' in file_path:
            labels['health_status'] = 0  # 正常
        elif 'jzb' in file_path:
            labels['health_status'] = 1  # 患者
        
        # 任务2: 抓握次数分类
        grip_count = features.get('grip_count', 0)
        if grip_count <= 2:
            labels['grip_count_level'] = 0  # 低频
        elif grip_count <= 5:
            labels['grip_count_level'] = 1  # 中频
        else:
            labels['grip_count_level'] = 2  # 高频
        
        # 任务3: 运动质量评估
        # 基于平滑度和协调性
        smoothness_avg = np.mean([features.get(f'{finger}_smoothness', 0) 
                                for finger in ['Thumb (WT)', 'Index (WI)', 'Middle (WM)', 'Ring (WR)', 'Pinky (WP)']])
        coordination = features.get('finger_coordination_mean', 0)
        
        # 综合评分
        quality_score = smoothness_avg * 0.6 + coordination * 0.4
        if quality_score < np.percentile([quality_score], 33):
            labels['motion_quality'] = 0  # 差
        elif quality_score < np.percentile([quality_score], 67):
            labels['motion_quality'] = 1  # 中等
        else:
            labels['motion_quality'] = 2  # 好
        
        # 任务4: 抓握频率分类
        grip_freq = features.get('grip_frequency', 0)
        if grip_freq < 0.5:
            labels['grip_frequency_level'] = 0  # 低频
        elif grip_freq < 1.0:
            labels['grip_frequency_level'] = 1  # 中频
        else:
            labels['grip_frequency_level'] = 2  # 高频
            
        return labels
    
    def create_sliding_windows(self, time_series, window_size=60, stride=30):
        """
        创建滑动窗口，用于分析局部运动模式
        
        参数:
        window_size: 窗口大小（帧数）
        stride: 滑动步长
        """
        windows = []
        n_frames = len(time_series)
        
        for start in range(0, n_frames - window_size + 1, stride):
            end = start + window_size
            window = time_series[start:end]
            windows.append(window)
        
        return np.array(windows)
    
    def augment_time_series(self, time_series, augmentation_factor=3):
        """
        时间序列数据增强
        """
        augmented_data = [time_series]  # 原始数据
        
        for _ in range(augmentation_factor - 1):
            # 1. 添加噪声
            noise_level = 0.02
            noisy_series = time_series + np.random.normal(0, noise_level, time_series.shape)
            
            # 2. 时间扭曲（轻微的时间伸缩）
            time_warp_factor = np.random.uniform(0.95, 1.05)
            original_length = len(time_series)
            new_length = int(original_length * time_warp_factor)
            
            if new_length > 0:
                indices = np.linspace(0, original_length-1, new_length)
                warped_series = np.array([np.interp(indices, range(original_length), time_series[:, i]) 
                                        for i in range(time_series.shape[1])]).T
                
                # 调整回原始长度
                if len(warped_series) != original_length:
                    indices_back = np.linspace(0, len(warped_series)-1, original_length)
                    warped_series = np.array([np.interp(indices_back, range(len(warped_series)), warped_series[:, i]) 
                                            for i in range(warped_series.shape[1])]).T
                
                augmented_data.append(warped_series)
            
            # 3. 幅值缩放
            scale_factor = np.random.uniform(0.9, 1.1)
            scaled_series = time_series * scale_factor
            augmented_data.append(scaled_series)
        
        return augmented_data
    
    def save_to_ts_format(self, X_data, y_data, file_names, task_type, output_dir):
        """
        保存数据为.ts格式文件
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 文件名
        ts_filename = f"HandGrip_{task_type}.ts"
        mapping_filename = f"HandGrip_{task_type}_mapping.txt"
        
        ts_path = os.path.join(output_dir, ts_filename)
        mapping_path = os.path.join(output_dir, mapping_filename)
        
        # 数据集信息
        n_samples, series_length, n_dimensions = X_data.shape
        class_labels = np.unique(y_data)
        
        # 写入.ts文件
        with open(ts_path, 'w') as f:
            # 写入元数据头
            f.write(f"@problemName HandGrip_{task_type}\n")
            f.write(f"@timeStamps false\n")
            f.write(f"@missing false\n")
            f.write(f"@univariate false\n")
            f.write(f"@dimensions {n_dimensions}\n")
            f.write(f"@equalLength true\n")
            f.write(f"@seriesLength {series_length}\n")
            f.write(f"@classLabel true {' '.join(map(str, class_labels))}\n")
            f.write("@data\n")
            
            # 写入数据行
            for i in range(n_samples):
                # 将每个时间步的所有维度值按顺序排列
                series_data = []
                for t in range(series_length):
                    series_data.extend([f"{val:.6f}" for val in X_data[i, t]])
                # 将整个序列连接起来，然后添加标签
                line = ",".join(series_data) + f":{y_data[i]}\n"
                f.write(line)
        
        # 写入映射文件
        with open(mapping_path, 'w') as f:
            f.write(f"# {task_type} 分类任务的数据行与原始文件名映射关系\n")
            f.write("# 格式: 行号 -> 原始文件名 (标签)\n")
            f.write(f"# 标签含义: {self.get_label_meaning(task_type)}\n")
            for i, (filename, label) in enumerate(zip(file_names, y_data)):
                f.write(f"{i+1} -> {filename} (标签: {label})\n")
        
        print(f"✅ .ts文件已保存: {ts_path}")
        print(f"✅ 映射文件已保存: {mapping_path}")
        
        return ts_path, mapping_path
    
    def get_label_meaning(self, task_type):
        """获取标签含义说明"""
        meanings = {
            'health_status': '0=正常人, 1=患者',
            'grip_count_level': '0=低频抓握(≤2次), 1=中频抓握(3-5次), 2=高频抓握(>5次)',
            'motion_quality': '0=运动质量差, 1=运动质量中等, 2=运动质量好',
            'grip_frequency_level': '0=低频率(<0.5Hz), 1=中频率(0.5-1Hz), 2=高频率(>1Hz)'
        }
        return meanings.get(task_type, '未知任务类型')
    
    def prepare_classification_dataset(self, task_type='health_status', 
                                     use_sliding_window=True, 
                                     window_size=60,
                                     use_augmentation=True,
                                     save_to_ts=True,
                                     output_dir="../dataset/HandGrip_Enhanced/"):
        """
        准备分类数据集
        
        参数:
        task_type: 分类任务类型 ('health_status', 'grip_count_level', 'motion_quality', 'grip_frequency_level')
        use_sliding_window: 是否使用滑动窗口
        window_size: 窗口大小
        use_augmentation: 是否使用数据增强
        """
        from feature_engineering import create_enhanced_time_series
        
        X_data = []
        y_data = []
        file_names = []
        
        # 处理normal目录
        normal_dir = os.path.join(self.csv_data_dir, 'normal')
        if os.path.exists(normal_dir):
            for file in os.listdir(normal_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(normal_dir, file)
                    try:
                        enhanced_ts, features = create_enhanced_time_series(file_path)
                        labels = self.create_multi_task_labels(file_path, features)
                        
                        if use_sliding_window and len(enhanced_ts) > window_size:
                            windows = self.create_sliding_windows(enhanced_ts, window_size)
                            for window in windows:
                                X_data.append(window)
                                y_data.append(labels[task_type])
                                file_names.append(file.replace('.csv', ''))
                        else:
                            X_data.append(enhanced_ts)
                            y_data.append(labels[task_type])
                            file_names.append(file.replace('.csv', ''))
                            
                        # 数据增强
                        if use_augmentation:
                            augmented_series = self.augment_time_series(enhanced_ts)
                            for aug_ts in augmented_series[1:]:  # 跳过原始数据
                                if use_sliding_window and len(aug_ts) > window_size:
                                    windows = self.create_sliding_windows(aug_ts, window_size)
                                    for window in windows:
                                        X_data.append(window)
                                        y_data.append(labels[task_type])
                                        file_names.append(file.replace('.csv', '') + '_aug')
                                else:
                                    X_data.append(aug_ts)
                                    y_data.append(labels[task_type])
                                    file_names.append(file.replace('.csv', '') + '_aug')
                                    
                    except Exception as e:
                        print(f"处理文件 {file} 时出错: {e}")
        
        # 处理jzb目录
        jzb_dir = os.path.join(self.csv_data_dir, 'jzb')
        if os.path.exists(jzb_dir):
            for file in os.listdir(jzb_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(jzb_dir, file)
                    try:
                        enhanced_ts, features = create_enhanced_time_series(file_path)
                        labels = self.create_multi_task_labels(file_path, features)
                        
                        if use_sliding_window and len(enhanced_ts) > window_size:
                            windows = self.create_sliding_windows(enhanced_ts, window_size)
                            for window in windows:
                                X_data.append(window)
                                y_data.append(labels[task_type])
                                file_names.append(file.replace('.csv', ''))
                        else:
                            X_data.append(enhanced_ts)
                            y_data.append(labels[task_type])
                            file_names.append(file.replace('.csv', ''))
                            
                        # 数据增强
                        if use_augmentation:
                            augmented_series = self.augment_time_series(enhanced_ts)
                            for aug_ts in augmented_series[1:]:  # 跳过原始数据
                                if use_sliding_window and len(aug_ts) > window_size:
                                    windows = self.create_sliding_windows(aug_ts, window_size)
                                    for window in windows:
                                        X_data.append(window)
                                        y_data.append(labels[task_type])
                                        file_names.append(file.replace('.csv', '') + '_aug')
                                else:
                                    X_data.append(aug_ts)
                                    y_data.append(labels[task_type])
                                    file_names.append(file.replace('.csv', '') + '_aug')
                                    
                    except Exception as e:
                        print(f"处理文件 {file} 时出错: {e}")
        
        # 统一长度处理
        if X_data:
            max_length = max(len(x) for x in X_data)
            X_processed = []
            
            for x in X_data:
                if len(x) < max_length:
                    # 用最后一个值填充
                    padding = np.tile(x[-1:], (max_length - len(x), 1))
                    x_padded = np.vstack([x, padding])
                else:
                    x_padded = x[:max_length]
                X_processed.append(x_padded)
            
            X_data = np.array(X_processed)
            y_data = np.array(y_data)
            
            print(f"数据集准备完成:")
            print(f"样本数量: {len(X_data)}")
            print(f"时间序列长度: {X_data.shape[1]}")
            print(f"特征维度: {X_data.shape[2]}")
            print(f"标签分布: {np.bincount(y_data)}")
            
            # 保存为.ts文件
            if save_to_ts:
                self.save_to_ts_format(X_data, y_data, file_names, task_type, output_dir)
            
            return X_data, y_data, file_names
        else:
            print("没有找到有效的数据文件")
            return None, None, None

# 使用示例
if __name__ == "__main__":
    # 创建数据集
    dataset = HandGripClassificationDataset("./csv_data")
    
    print("=== 生成多个分类任务的.ts数据集 ===")
    
    # 1. 健康状态分类数据集（主任务）
    print("\n1. 准备健康状态分类数据集...")
    X1, y1, files1 = dataset.prepare_classification_dataset(
        task_type='health_status',
        use_sliding_window=False,  # 先不使用滑动窗口
        use_augmentation=False,    # 先不使用数据增强
        save_to_ts=True,
        output_dir="../dataset/HandGrip_Enhanced/"
    )
    
    # 2. 抓握次数分级数据集
    print("\n2. 准备抓握次数分级数据集...")
    X2, y2, files2 = dataset.prepare_classification_dataset(
        task_type='grip_count_level',
        use_sliding_window=False,
        use_augmentation=False,
        save_to_ts=True,
        output_dir="../dataset/HandGrip_Enhanced/"
    )
    
    # 3. 运动质量评估数据集
    print("\n3. 准备运动质量评估数据集...")
    X3, y3, files3 = dataset.prepare_classification_dataset(
        task_type='motion_quality',
        use_sliding_window=False,
        use_augmentation=False,
        save_to_ts=True,
        output_dir="../dataset/HandGrip_Enhanced/"
    )
    
    # 4. 抓握频率分析数据集
    print("\n4. 准备抓握频率分析数据集...")
    X4, y4, files4 = dataset.prepare_classification_dataset(
        task_type='grip_frequency_level',
        use_sliding_window=False,
        use_augmentation=False,
        save_to_ts=True,
        output_dir="../dataset/HandGrip_Enhanced/"
    )
    
    print("\n=== 数据集生成完成 ===")
    print("生成的.ts文件位置:")
    print("📁 ../dataset/HandGrip_Enhanced/")
    print("   ├── HandGrip_health_status.ts          (健康状态分类)")
    print("   ├── HandGrip_grip_count_level.ts       (抓握次数分级)")
    print("   ├── HandGrip_motion_quality.ts         (运动质量评估)")
    print("   ├── HandGrip_grip_frequency_level.ts   (抓握频率分析)")
    print("   └── 对应的_mapping.txt文件")
    
    print("\n使用方法:")
    print("python run.py --task_name classification --is_training 1 \\")
    print("  --model TimesNet --data UEA \\")
    print("  --root_path ./dataset/HandGrip_Enhanced/ \\")
    print("  --model_id HandGrip_health_status")