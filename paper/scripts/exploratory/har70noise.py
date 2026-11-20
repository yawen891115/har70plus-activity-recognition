import pandas as pd
import numpy as np
#visualisation
import matplotlib.pyplot as plt
import seaborn as sns

#降噪504（体现噪声）
#求504人back传感器的合向量的模
#导入504df
df_5041 = pd.read_csv(f'/Users/zhengyawen/Desktop/zywhar70/har70plus/504.csv')
# 计算欧氏距离的函数
def euclidean_distance(row):
    return np.sqrt(row['back_x']**2 + row['back_y']**2 + row['back_z']**2)
# 应用函数并将结果添加为新列
df_5041['euclidean_distance'] = df_5041.apply(euclidean_distance, axis=1)
print(df_5041)

#对于每个label，我都画出对应的时间戳和欧氏距离的time series plot
# 循环遍历每个 label
for label in [1, 3, 4, 5, 6, 7, 8]:
    # 筛选出指定 label 的数据子集
    subset = df_5041[df_5041['label'] == label]
    # 绘制时间序列图
    x = range(1,150760)
    plt.figure(figsize=(10, 6))
    plt.plot(x, subset['euclidean_distance'], marker='o', linestyle='-')
    plt.title(f'Time Series Plot for Label {label}')
    plt.xlabel('Timestamp')
    plt.ylabel('Euclidean Distance for 504')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#由上面的合向量图看出来有噪声
#用均值滤波去噪，按照每列为总数据集merged_df去噪
# 创建数据副本
df_504 = df_5041.copy()
N = 5
# 指定要过滤的加速度数据列
columns_to_filter = ['euclidean_distance']
# 对指定的列应用移动均值滤波
for col in columns_to_filter:
    df_504[col + '_filtered'] = df_504[col].rolling(window=N).mean()
df_504= df_504.drop(columns=['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z','euclidean_distance'])
#对于每个label，我都画出对应的时间戳和欧氏距离的time series plot
# 循环遍历每个 label
for label in [1, 3, 4, 5, 6, 7, 8]:
    # 筛选出指定 label 的数据子集
    subset = df_504[df_504['label'] == label]
    # 绘制时间序列图
    plt.figure(figsize=(10, 6))
    plt.plot(subset['timestamp'], subset['euclidean_distance_filtered'], marker='o', linestyle='-')
    plt.title(f'Time Series Plot for Label {label}')
    plt.xlabel('Timestamp')
    plt.ylabel('Filtered Euclidean Distance for 504')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


#512（体现噪声）
    #求512人back传感器的合向量的模
#导入512df
df_5121 = pd.read_csv(f'')
# 计算欧氏距离的函数
def euclidean_distance(row):
    return np.sqrt(row['thigh_x']**2 + row['thigh_y']**2 + row['thigh_z']**2)
# 应用函数并将结果添加为新列
df_5121['euclidean_distance'] = df_5121.apply(euclidean_distance, axis=1)
print(df_5121)

#对于每个label，我都画出对应的时间戳和欧氏距离的time series plot
# 循环遍历每个 label
for label in [1, 3, 4, 5, 6, 7, 8]:
    # 筛选出指定 label 的数据子集
    subset = df_5121[df_5121['label'] == label]
    # 绘制时间序列图
    plt.figure(figsize=(10, 6))
    plt.plot(subset['timestamp'], subset['euclidean_distance'], marker='o', linestyle='-')
    plt.title(f'Time Series Plot for Label {label}')
    plt.xlabel('Timestamp')
    plt.ylabel('Euclidean Distance for 512')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#由上面的合向量图看出来有噪声
#用均值滤波去噪，按照每列为总数据集merged_df去噪
# 创建数据副本
df_512 = df_5121.copy()
N = 5
# 指定要过滤的加速度数据列
columns_to_filter = ['euclidean_distance']
# 对指定的列应用移动均值滤波
for col in columns_to_filter:
    df_512[col + '_filtered'] = df_512[col].rolling(window=N).mean()
df_512 = df_512.drop(columns=['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z','euclidean_distance'])
#对于每个label，我都画出对应的时间戳和欧氏距离的time series plot
# 循环遍历每个 label
for label in [1, 3, 4, 5, 6, 7, 8]:
    # 筛选出指定 label 的数据子集
    subset = df_512[df_512['label'] == label]
    # 绘制时间序列图
    plt.figure(figsize=(10, 6))
    plt.plot(subset['timestamp'], subset['euclidean_distance_filtered'], marker='o', linestyle='-')
    plt.title(f'Time Series Plot for Label {label}')
    plt.xlabel('Timestamp')
    plt.ylabel('Filtered Euclidean Distance for 512')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


#降噪
# 创建数据副本
merged_df = merged_df1.copy()
# 设置移动均值滤波的窗口大小
N = 5
# 指定要过滤的加速度数据列
columns_to_filter = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
# 对指定的列应用移动均值滤波，并直接更新这些列
for col in columns_to_filter:
    merged_df[col] = merged_df[col].rolling(window=N).mean()
# 删除移动均值滤波导致的前N-1行空值
merged_df = merged_df.iloc[N-1:].reset_index(drop=True)