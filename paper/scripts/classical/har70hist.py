#hist base algo
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
###########################################
#import the csv into jupyter notebook
#the dataset
############################################
merged_df1 = pd.DataFrame()
# 循环读取并合并18个数据集
for i in range(501, 503):
    # 读取数据集
    df = pd.read_csv(f'/Users/zhengyawen/Desktop/zywhar70/har70dataset/{i}.csv')
    # 第一个数据集直接赋值给merged_df
    if i == 501:
        merged_df1 = df
    else:
        # 根据共同列进行合并
        merged_df1 = pd.concat([merged_df1,df],ignore_index=True)
# 输出合并后的数据集
print(merged_df1)



###########################
#data preprocessing
###########################
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




from sklearn.model_selection import KFold

# 假设你有一个特征矩阵X和对应的目标向量y
X = merged_df.drop(columns=['timestamp','label'])  # 删除目标列以获取特征矩阵
y = merged_df['label']  # 获取目标向量

# 初始化十折交叉验证对象
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 使用HistGradientBoostingClassifier
classifier = HistGradientBoostingClassifier(max_iter=100, random_state=42)

#then start the evaluation indx calaulation procedure
# 初始化列表以收集每个折的预测结果
all_y_true = []
all_y_pred = []

# 计算每折的评估指标并收集预测结果
#hint: y_test is the ground truth
for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    print(f"\nFold {i+1} Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='macro'))
    print("Recall:", recall_score(y_test, y_pred, average='macro'))
    print("F1-score:", f1_score(y_test, y_pred, average='macro'))
    
    # 将当前折的真实标签和预测标签添加到列表中
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

# 计算整体评估指标
print("\nOverall Evaluation:")
print("Accuracy:", accuracy_score(all_y_true, all_y_pred))
print("Precision:", precision_score(all_y_true, all_y_pred, average='macro'))
print("Recall:", recall_score(all_y_true, all_y_pred, average='macro'))
print("F1-score:", f1_score(all_y_true, all_y_pred, average='macro'))

# 打印分类报告
print("\nClassification Report:")
print(classification_report(all_y_true, all_y_pred))

# 打印混淆矩阵
print("\nConfusion Matrix:")
print(confusion_matrix(all_y_true, all_y_pred))

