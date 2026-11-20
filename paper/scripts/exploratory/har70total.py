########################################
#import some nessisary package
########################################
import pandas as pd
import numpy as np
#visualisation
import matplotlib.pyplot as plt
import seaborn as sns
# data preprocessing
from sklearn.preprocessing import StandardScaler
#import evaluation index such as confusion matrix, accuracy_score, f1_score, precision_score and recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
# data splitting
from sklearn.model_selection import train_test_split
#import important machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier



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

#check whether the dataset exist missing value
# 查看每列缺失值的数量
missing_values = merged_df1.isnull().sum()
# 查看整个数据集缺失值的总数量
total_missing = merged_df1.isnull().sum().sum()
# 打印缺失值情况
print(missing_values)
print(total_missing)




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
    plt.figure(figsize=(10, 6))
    plt.plot(subset['timestamp'], subset['euclidean_distance'], marker='o', linestyle='-')
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


#异常值处理
#画出箱形图
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

#################################
#重抽样技术
#################################
#10-fold CV
from sklearn.model_selection import KFold
# 假设你有一个特征矩阵X和对应的目标向量y
X = merged_df.drop(columns=['timestamp','label'])  # 删除目标列以获取特征矩阵
y = merged_df['label']  # 获取目标向量
# 初始化十折交叉验证对象
kf = KFold(n_splits=10, shuffle=True, random_state=42)


##################################
#统计学分析
##################################
#共线性检验
# 使用corr()函数计算相关系数矩阵
correlation_matrix = merged_df.drop(columns=['label']).corr()
# 打印相关系数矩阵
print(correlation_matrix)
# 使用Seaborn的heatmap函数绘制热力图
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# 设置热力图的标题
plt.title('Correlation Heatmap')
# 显示图形
plt.show()

#共线性检验
#使用VIF（方差膨胀因子）检验共线性
from statsmodels.stats.outliers_influence import variance_inflation_factor
# 添加一个截距列，否则VIF求不出来，VIF是基于线性回归模型的
X_vif = merged_df.drop(columns=['timestamp','label'])
X_vif['Intercept'] = 1
# 初始化一个DataFrame来存储VIF结果
vif_data = pd.DataFrame()
vif_data["Variable"] = X_vif.columns
# 计算每个变量的VIF
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
# 打印结果
print(vif_data)

#结果表明变量未存在明显的多重共线性（VIF<10），因此我们考虑使用多分类logistic模型
#统计学方法：使用多分类logistic进行分类
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
# 初始化你要用的分类器，比如随机森林
classifier = LogisticRegression(multi_class='multinomial', solver='sag', max_iter=1000)
# 已经有了十折交叉验证对象
# 使用交叉验证进行模型评估
scores = cross_val_score(classifier, X, y, cv=kf)
# 输出十折交叉验证的准确率
print("Cross-validated scores:", scores)
print("Mean accuracy:", np.mean(scores))


#########################################
#机器学习算法
#########################################
#Naiev Bayesian algo
classifier = GaussianNB()
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

#decision tree algo
classifier = DecisionTreeClassifier(criterion='gini',splitter='best')
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

#XGB algo
# 分离特征和标签
X = merged_df.drop(columns=['label'])   #transform the dataframe into numpy
y = merged_df['label']   #transform the dataframe into numpy
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 使用HistGradientBoostingClassifier
clf = HistGradientBoostingClassifier(max_iter=100, random_state=42)
clf.fit(X_train, y_train)
# 进行预测
y_pred = clf.predict(X_test)
# 评估模型
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)

#random forest algo
classifier = RandomForestClassifier(n_estimators=6,criterion='gini')
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

#KNN algo
classifier = KNeighborsClassifier(n_neighbors=10)
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

#SVM algo
# 提取特征和标签
X = merged_df[['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']]
y = merged_df['label']
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)
# 使用最佳参数创建SVM模型
classifier = SVC(kernel='linear', C=1)  # 以线性核和C=1为例
# 训练模型
classifier.fit(X_train, y_train)
# 预测测试集
y_pred = classifier.predict(X_test)
# 计算性能指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)


#quadratic discriminant analysis
classifier = QuadraticDiscriminantAnalysis()
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


###############################
#deep learning algorithm
###############################
#MLP algo
#Hint: I will not use the cross validation in this algo because of the complexity
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np
# 分离特征和标签
X = merged_df.drop(columns=['label']).values   #transform the dataframe into numpy
y = merged_df['label'].values   #transform the dataframe into numpy
# 将标签转换为one-hot编码
y = to_categorical(y)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(y_train)
# 创建MLP模型
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')
])
# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
# 预测测试集
y_pred = model.predict(X_test)
# 由于预测结果是one-hot编码的形式，需要转换为标签
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)
# 计算loss，accuracy
loss, accuracy = model.evaluate(X_test, y_test)
print(f'测试集上的损失: {loss:.4f}, 准确率: {accuracy:.4f}')
# 计算Precision
precision = precision_score(y_test_labels, y_pred_labels, average='macro')
print(f'总体Precision: {precision:.4f}')
# 计算Recall
recall = recall_score(y_test_labels, y_pred_labels, average='macro')
print(f'总体Recall: {recall:.4f}')
# 计算F1分数
f1 = f1_score(y_test_labels, y_pred_labels, average='macro')
print(f'总体F1分数: {f1:.4f}')
# 生成混淆矩阵
cm = confusion_matrix(y_test_labels, y_pred_labels)
print('混淆矩阵:')
print(cm)

#CNN algo
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np
# 提取特征和标签
X = merged_df[['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']].values
y = merged_df['label'].values
# 将标签转换为one-hot编码
y = to_categorical(y)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)
# 重构数据以适应CNN的输入
X_train = X_train.reshape((X_train.shape[0], 2, 3, 1))
X_test = X_test.reshape((X_test.shape[0], 2, 3, 1))
# 构建CNN模型
model = models.Sequential([
    layers.ZeroPadding2D(padding=(1, 1), input_shape=(2, 3, 1)),  # 加入零填充层
    layers.Conv2D(32, (2, 2), activation='relu'),  # 共32个卷积同时运算，激活函数为relu
    layers.MaxPooling2D((1, 1)),  # 其余层维持不变
    layers.Flatten(), #展平为一维向量
    layers.Dense(64, activation='relu'), #全连接层的神经元为64个
    layers.Dense(y.shape[1], activation='softmax')  # 输出层，用于分类
])
# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
# 预测测试集
y_pred = model.predict(X_test)
# 由于预测结果是one-hot编码的形式，需要转换为标签
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)
# 计算loss，accuracy
loss, accuracy = model.evaluate(X_test, y_test)
print(f'测试集上的损失: {loss:.4f}, 准确率: {accuracy:.4f}')
# 计算Precision
precision = precision_score(y_test_labels, y_pred_labels, average='macro')
print(f'总体Precision: {precision:.4f}')
# 计算Recall
recall = recall_score(y_test_labels, y_pred_labels, average='macro')
print(f'总体Recall: {recall:.4f}')
# 计算F1分数
f1 = f1_score(y_test_labels, y_pred_labels, average='macro')
print(f'总体F1分数: {f1:.4f}')
# 生成混淆矩阵
cm = confusion_matrix(y_test_labels, y_pred_labels)
print('混淆矩阵:')
print(cm)


#LSTM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
# 加载数据
df = pd.read_csv('/Users/zhengyawen/Desktop/zywhar70/har70dataset/504.csv')
# 假设 'timestamp' 列不是特征
df = df.drop(columns=['timestamp'])
# 分离特征和标签
X = df.drop(columns=['label']).values
y = df['label'].values
# 对特征进行标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 添加时间步长维度
X_scaled = np.expand_dims(X_scaled, axis=1)
# 将标签编码为整数
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 将整数标签转换为one-hot编码
y_onehot = to_categorical(y_encoded)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.3, random_state=42)

# 创建LSTM模型

#线性堆叠层.两层LSTM将按顺序执行，其中每层的输入都来自前一层的输出
model = Sequential()
#第一个LSTM层，激活函数使用默认
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.5))
#第二个LSTM层
model.add(LSTM(64, return_sequences=False ))
model.add(Dropout(0.5))


#加了全连接层Dense，用于将加速度数据分类
model.add(Dense(y_train.shape[1], activation='softmax'))


# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 添加早停以防止过拟合
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[early_stopping])

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'测试集上的损失: {loss:.4f}, 准确率: {accuracy:.4f}')









