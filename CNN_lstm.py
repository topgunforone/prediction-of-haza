from matplotlib import pyplot
from pandas import DataFrame, concat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pandas import read_csv
from math import sqrt
import keras
from keras.layers import StackedRNNCells

from keras.models import Sequential
from keras.layers import Input,Conv2D,merge ,Lambda,Reshape,GRU,Dense,Concatenate,BatchNormalization,Dropout,Bidirectional,Flatten
from keras.layers import LSTM
from prepare_data import prepare_data
'''
from prepare_data2 import prepare_data
import keras
train_x,train_y,val_x,val_y =prepare_data()
print('data shape\n','\t tran_x\t',train_x.shape)
print('\t train_y\t',train_y.shape)
print('\t val_y\t',val_y.shape)
batch_size= 32
time_step = train_x.shape[1]
input_shape = (time_step,10,10,9)  # one sample size

input_0 = Input(input_shape,name = 'input') #[12,10,10,9]
'''
train_X,train_y,test_X,test_y = prepare_data()
print('data shape\n','\t tran_x\t',train_X.shape)
print('\t train_y\t',train_y.shape)
print('\t val_x\t',test_X.shape)
print('\t val_y\t',test_y.shape)

'''
# load dataset
dataset = read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
#对第五列非数值参数进行编码
#print(values[:,4],'打印完毕')
# integer encode direction
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
print(reframed.head())

values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
'''
from keras.layers import Flatten ,ConvLSTM2D
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(10,10,9)))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(10,10,9),data_format='channel_first'))
model.add(Dropout(0.4))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.4))
model.add(Conv2D(1, kernel_size=(3, 3), activation='relu'))#output_size = (6*6*32)
model.add(Dropout(0.4))
# model.add(Flatten())
model.add(Reshape(target_shape=(-1,1)))
################
#数据拉直代码
################

model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, return_sequences=True))#,input_shape=(6, 6, 32)
model.add(Flatten())
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

history = model.fit(train_X, train_y, epochs=20, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

pyplot.title('loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

from numpy import concatenate
from sklearn.metrics import mean_squared_error
yhat = model.predict(test_X)#提取的是预测值？

#test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

'''
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
'''

print(yhat)
print(test_y)
pyplot.plot(yhat[:100], label='Pred')
pyplot.plot(test_y[:100], label='True')
pyplot.legend()
pyplot.show()

i = 0
count = 0
while i<len(yhat):
    if -50<= yhat[i] - test_y[i] <=50:
        count = count + 1
    i = i + 1
print(count/len(yhat))
