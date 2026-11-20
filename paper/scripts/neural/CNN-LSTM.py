import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, Dropout, ZeroPadding2D
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2,l1

# 加载和预处理数据
merged_df1 = pd.read_csv('/Users/zhengyawen/Desktop/zywhar70/har70dataset/501.csv')
merged_df1['label'] = merged_df1['label'].apply(lambda x: int(x))  # 确保标签是整数类型

# 降噪
N = 5
columns_to_filter = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
for col in columns_to_filter:
    merged_df1[col] = merged_df1[col].rolling(window=N).mean()
merged_df = merged_df1.iloc[N-1:].reset_index(drop=True)

# 划分特征和标签
X = merged_df[['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']].values
y = merged_df['label'].values
y = to_categorical(y)  # one-hot 编码

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train = X_train.reshape((X_train.shape[0], 1, 2, 3, 1))  # 重塑为 (样本数, 时间步长, 高, 宽, 通道数)
X_test = X_test.reshape((X_test.shape[0], 1, 2, 3, 1))

# 构建模型
# 设置L2正则化的系数
l2_reg = 0.001

model = models.Sequential([
    layers.TimeDistributed(layers.ZeroPadding2D(padding=(1, 1)), input_shape=(None, 2, 3, 1)),  # 加入零填充层
    layers.TimeDistributed(layers.Conv2D(32, (2, 2), activation='relu', kernel_regularizer=l1(l2_reg))),  # 在每个时间步应用卷积，加入L2正则化
    layers.TimeDistributed(layers.MaxPooling2D((1, 1))),  # 在每个时间步应用池化
    layers.TimeDistributed(layers.Flatten()),  # 在每个时间步展平
    layers.LSTM(128, return_sequences=True),  # 第一个LSTM层
    layers.Dropout(0.5),  # Dropout层
    layers.LSTM(64, return_sequences=False),  # 第二个LSTM层
    layers.Dropout(0.5),  # 另一个Dropout层
    layers.Dense(y_train.shape[1], activation='softmax', kernel_regularizer=l1(l2_reg))  # 输出层，使用softmax激活函数，加入L2正则化
])

# LSTM,add dropout: 0.9243
# CNN_LSTM,add dropout: 0.9636
# CNN_LSTM,add l2&dropout: 0.9593
# CNN_LSTM,add l2: 0.9640
# CNN_LSTM,add l1&dropout: 0.9648
# CNN_LSTM,add l1: 0.9653

# 编译和训练模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'],)
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'测试集上的损失: {loss:.4f}, 准确率: {accuracy:.4f}')
