import numpy as np

#np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics.regression import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from dbn import SupervisedDBNRegression
import pandas as pd


# Loading dataset
# boston = load_boston()
# X, Y = boston.data, boston.target

# Splitting data
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1337)

# 读取数据
train_data = np.array(pd.read_csv("C:\\Users\\Administrator\\Desktop\\30train.csv"))
test_data = np.array(pd.read_csv("C:\\Users\\Administrator\\Desktop\\30255.csv"))

# 提取特征列,即X (共18列，代表18个变量)
# train_feature = np.array(train_data[0:317, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]])
X_train = np.array(train_data[0:316, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]])

# 提取预测结果列，即Y
Y_train = np.array(train_data[0:316, [30]])

# 提取测试集特征列
# test_x = np.array(test_data[0:7, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]])
X_test = np.array(test_data[0:1000, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]])
#Y_test = np.array(train_data[0:6, [30]])

# Data scaling
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)

# Training
regressor = SupervisedDBNRegression(hidden_layers_structure=[130],
                                    learning_rate_rbm=0.05,
                                    learning_rate=0.05,
                                    n_epochs_rbm=20,
                                    n_iter_backprop=20,
                                    batch_size=16,
                                    activation_function='sigmoid')
regressor.fit(X_train, Y_train)

# Test
X_test = min_max_scaler.transform(X_test)
Y_pred = regressor.predict(X_test)
print("测试集预测结果")
print(Y_pred)
#print('Done.\nR-squared: %f\nMSE: %f' % (r2_score(Y_test, Y_pred), mean_squared_error(Y_test, Y_pred)))
