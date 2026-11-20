import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

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


# 假设 'timestamp' 列不是特征
merged_df = merged_df.drop(columns=['timestamp'])

# 分离特征和标签
X = merged_df.drop(columns=['label']).values
y = merged_df['label'].values

# 添加时间步长维度
X = np.expand_dims(X, axis=1)

# 将标签编码为整数
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 将整数标签转换为one-hot编码
y_onehot = to_categorical(y_encoded)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3, random_state=42)

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
