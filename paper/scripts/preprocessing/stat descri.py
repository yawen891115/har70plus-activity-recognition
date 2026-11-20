import pandas as pd

# 循环读取并合并18个数据集
for i in range(501, 519):
    # 读取数据集
    df = pd.read_csv(f'/Users/zhengyawen/Desktop/zywhar70/har70dataset/{i}.csv')
    
    # 第一个数据集直接赋值给merged_df
    if i == 501:
        merged_df = df
    else:
        # 根据共同列进行合并
        merged_df = pd.concat([merged_df,df],ignore_index=True)

# 输出合并后的数据集
print(merged_df)

# 创建数据副本
merged_df = merged_df.copy()

# 设置移动均值滤波的窗口大小
N = 5

# 指定要过滤的加速度数据列
columns_to_filter = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']

# 对指定的列应用移动均值滤波，并直接更新这些列
for col in columns_to_filter:
    merged_df[col] = merged_df[col].rolling(window=N).mean()

# 删除移动均值滤波导致的前N-1行空值
merged_df = merged_df.iloc[N-1:].reset_index(drop=True)

# 查看数据集的前几行，以了解其结构
print(merged_df.head())

# 对每个不同label的不同特征进行标准的统计学分析
# 这将为每个传感器位置（背部和大腿）和每个轴（x, y, z）提供描述性统计信息
statistical_analysis = merged_df.groupby('label').describe()

# 以转置的形式打印统计分析结果，以便更清晰地查看每个标签的统计数据
print(statistical_analysis.transpose())


# Save the statistical analysis results to a CSV file
csv_output_path = '/Users/zhengyawen/Desktop/statistical_analysis_results.csv'
statistical_analysis.to_csv(csv_output_path)

csv_output_path

