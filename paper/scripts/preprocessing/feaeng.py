import pandas as pd
import numpy as np
from scipy.fft import fft

data_path = '/Users/zhengyawen/Desktop/zywhar70/har70dataset/501.csv'
data = pd.read_csv(data_path)

# 填充任何可能的NaN值
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

# 定义窗口大小
window_size = 25

# 使用列表来存储每个窗口的特征计算结果
windows = []

# 定义传感器数据列
sensor_columns = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']

# 分割数据为不重叠的窗口并计算特征
for start in range(0, len(data), window_size):
    end = start + window_size
    # 切片窗口
    window_data = data[sensor_columns][start:end]
    
    if len(window_data) != window_size:
        # 如果窗口大小不够，跳过这个窗口
        continue
    
    # 计算特征
    features = {
        'timestamp': data['timestamp'].iloc[start],
        'signal_flipping_index': ((window_data.diff() < 0).sum() / (window_size - 1)).mean(),
        'magnitude_area': np.sqrt((window_data**2).sum(axis=1)).mean(),
        'mean': window_data.mean().mean(),
        'std_dev': window_data.std().mean(),
        'variance': window_data.var().mean(),
        'skewness': window_data.skew().mean(),
        'kurtosis': window_data.kurtosis().mean(),
        'first_discrete_diff': window_data.diff().iloc[1:].mean().mean(),
        'max_value': window_data.max().max(),
        'min_value': window_data.min().min(),
        'label': data['label'].iloc[start]  # 假设每个窗口的标签由窗口开始的标签决定
    }
    
    # 计算DFT并保留第一个系数
    dft_coefficients = {f'dft_first_coefficient_{col}': np.abs(fft(window_data[col].to_numpy())[0]) / window_size
                        for col in sensor_columns}
    features.update(dft_coefficients)
    
    # 添加窗口特征到列表中
    windows.append(features)

# 创建特征DataFrame
features_df = pd.DataFrame(windows)

cols = [col for col in features_df.columns if col != 'label']

# 添加 'label' 到列表的末尾
cols.append('label')

# 根据新的列顺序重新排列DataFrame
features_df = features_df[cols]

# 显示一些行来确认列已正确移动
print(features_df.head())



