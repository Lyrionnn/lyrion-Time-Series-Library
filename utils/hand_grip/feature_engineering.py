import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
import os

def extract_hand_grip_features(df):
    """
    从手部抓握数据中提取多维特征
    
    参数:
    df: DataFrame，包含Frame, Thumb (WT), Index (WI), Middle (WM), Ring (WR), Pinky (WP)列
    
    返回:
    feature_dict: 包含各种特征的字典
    """
    # 原始距离特征
    fingers = ['Thumb (WT)', 'Index (WI)', 'Middle (WM)', 'Ring (WR)', 'Pinky (WP)']
    finger_data = df[fingers].values
    
    features = {}
    
    # 1. 基础统计特征
    for i, finger in enumerate(fingers):
        finger_series = finger_data[:, i]
        features[f'{finger}_mean'] = np.mean(finger_series)
        features[f'{finger}_std'] = np.std(finger_series)
        features[f'{finger}_range'] = np.max(finger_series) - np.min(finger_series)
        features[f'{finger}_skew'] = skew(finger_series)
        features[f'{finger}_kurtosis'] = kurtosis(finger_series)
    
    # 2. 运动学特征（速度、加速度）
    for i, finger in enumerate(fingers):
        finger_series = finger_data[:, i]
        # 一阶导数（速度）
        velocity = np.diff(finger_series)
        features[f'{finger}_velocity_mean'] = np.mean(velocity)
        features[f'{finger}_velocity_std'] = np.std(velocity)
        features[f'{finger}_velocity_max'] = np.max(np.abs(velocity))
        
        # 二阶导数（加速度）
        acceleration = np.diff(velocity)
        features[f'{finger}_acceleration_mean'] = np.mean(acceleration)
        features[f'{finger}_acceleration_std'] = np.std(acceleration)
        features[f'{finger}_acceleration_max'] = np.max(np.abs(acceleration))
    
    # 3. 手指间协调性特征
    # 计算手指间的相关性
    finger_corr = np.corrcoef(finger_data.T)
    features['finger_coordination_mean'] = np.mean(finger_corr[np.triu_indices_from(finger_corr, k=1)])
    features['finger_coordination_std'] = np.std(finger_corr[np.triu_indices_from(finger_corr, k=1)])
    
    # 手部整体收缩度（所有手指距离的平均值）
    hand_closure = np.mean(finger_data, axis=1)
    features['hand_closure_mean'] = np.mean(hand_closure)
    features['hand_closure_std'] = np.std(hand_closure)
    features['hand_closure_range'] = np.max(hand_closure) - np.min(hand_closure)
    
    # 4. 抓握周期性特征
    # 检测峰值（抓握动作）
    hand_closure_inverted = -hand_closure  # 反转，因为抓握时距离减小
    peaks, properties = signal.find_peaks(hand_closure_inverted, 
                                        height=np.mean(hand_closure_inverted),
                                        distance=10)  # 最小间隔10帧
    
    features['grip_count'] = len(peaks)  # 抓握次数
    
    if len(peaks) > 1:
        # 抓握间隔
        grip_intervals = np.diff(peaks)
        features['grip_interval_mean'] = np.mean(grip_intervals)
        features['grip_interval_std'] = np.std(grip_intervals)
        features['grip_frequency'] = len(peaks) / (len(hand_closure) / 30)  # 假设30fps
    else:
        features['grip_interval_mean'] = 0
        features['grip_interval_std'] = 0
        features['grip_frequency'] = 0
    
    # 5. 频域特征
    # 对手部整体收缩度进行FFT分析
    fft_closure = np.fft.fft(hand_closure)
    fft_magnitude = np.abs(fft_closure)
    freqs = np.fft.fftfreq(len(hand_closure))
    
    # 主要频率成分
    dominant_freq_idx = np.argmax(fft_magnitude[1:len(fft_magnitude)//2]) + 1
    features['dominant_frequency'] = freqs[dominant_freq_idx]
    features['dominant_magnitude'] = fft_magnitude[dominant_freq_idx]
    
    # 频谱能量分布
    features['low_freq_energy'] = np.sum(fft_magnitude[1:len(fft_magnitude)//10])
    features['mid_freq_energy'] = np.sum(fft_magnitude[len(fft_magnitude)//10:len(fft_magnitude)//4])
    features['high_freq_energy'] = np.sum(fft_magnitude[len(fft_magnitude)//4:len(fft_magnitude)//2])
    
    # 6. 运动质量特征
    # 平滑度（抖动程度）
    for i, finger in enumerate(fingers):
        finger_series = finger_data[:, i]
        jerk = np.diff(finger_series, n=3)  # 三阶导数
        features[f'{finger}_smoothness'] = -np.mean(np.abs(jerk))  # 负值，越小越平滑
    
    # 对称性（左右手对比，如果有的话）
    if 'L' in df.columns.str.cat() and 'R' in df.columns.str.cat():
        # 这里可以添加左右手对称性分析
        pass
    
    return features

def create_enhanced_time_series(csv_file_path):
    """
    创建增强的时间序列特征
    """
    df = pd.read_csv(csv_file_path)
    
    # 提取所有特征
    features = extract_hand_grip_features(df)
    
    # 原始时间序列
    fingers = ['Thumb (WT)', 'Index (WI)', 'Middle (WM)', 'Ring (WR)', 'Pinky (WP)']
    original_series = df[fingers].values
    
    # 增强时间序列：添加衍生特征
    enhanced_series = []
    
    # 1. 原始距离
    enhanced_series.append(original_series)
    
    # 2. 速度序列
    velocity_series = np.diff(original_series, axis=0)
    velocity_series = np.vstack([velocity_series[0:1], velocity_series])  # 补齐长度
    enhanced_series.append(velocity_series)
    
    # 3. 手部整体收缩度
    hand_closure = np.mean(original_series, axis=1, keepdims=True)
    enhanced_series.append(np.tile(hand_closure, (1, 5)))
    
    # 4. 归一化特征
    normalized_series = (original_series - np.mean(original_series, axis=0)) / np.std(original_series, axis=0)
    enhanced_series.append(normalized_series)
    
    # 合并所有特征维度
    enhanced_ts = np.concatenate(enhanced_series, axis=1)
    
    return enhanced_ts, features

# 示例使用
if __name__ == "__main__":
    # 处理单个文件示例
    sample_file = "./csv_data/normal/h1000L.csv"
    if os.path.exists(sample_file):
        enhanced_ts, features = create_enhanced_time_series(sample_file)
        print(f"原始维度: 5")
        print(f"增强后维度: {enhanced_ts.shape[1]}")
        print(f"提取的特征数量: {len(features)}")
        print("\n部分特征示例:")
        for i, (key, value) in enumerate(features.items()):
            if i < 10:  # 只显示前10个特征
                print(f"{key}: {value:.4f}")