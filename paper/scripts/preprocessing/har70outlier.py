import pandas as pd
import numpy as np
#visualisation
import matplotlib.pyplot as plt
import seaborn as sns
#异常值处理
#画出箱形图
merged_df = pd.DataFrame()
# 循环读取并合并18个数据集
for i in range(501, 518):
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
# Dropping the 'timestamp','label' columns
df_with_label = merged_df.drop(columns=['timestamp'])
df_with_label = merged_df.drop(columns=['label'])
# Plotting boxplots for each feature on the same plot, excluding the timestamp
plt.figure(figsize=(12, 8))
df_with_label.boxplot()
plt.title('Boxplot for Each Feature in df_merged')
plt.ylabel('Value')
plt.grid(False)  # Turning off the grid for aesthetics
plt.show()
