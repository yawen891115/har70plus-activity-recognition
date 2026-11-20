import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

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


# 分离特征和标签
X = merged_df.drop(columns=['label']).values   #transform the dataframe into numpy
y = merged_df['label'].values   #transform the dataframe into numpy

# 将标签转换为one-hot编码
y = to_categorical(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(y_train)
#定义评估指标函数
#函数1
def get_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

#函数2
def print_evaluation_index(y_test,y_pred):
    print("Confusion matrix\n")
    print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
    accuracy, precision, recall, f1 = get_metrics(y_test,y_pred)
    print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision ,recall, f1))

# 创建MLP模型
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')
])

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

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
