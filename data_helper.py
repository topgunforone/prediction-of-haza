from pandas import read_csv
from datetime import datetime
import keras
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation
from keras.layers import Embedding
from keras.layers import LSTM

# # load data
# def parse(x):
#     return datetime.strptime(x, '%Y %m %d %H')
#
#
# def load_data():
#     dataset = read_csv('PRSA_data_2010.1.1-2014.12.31.csv', parse_dates=[['year', 'month', 'day', 'hour']], index_col=0,
#                        date_parser=parse)
#     # print(dataset)
#
#     dataset = dataset[24:]
#     y = dataset['pm2.5']
#     x = dataset[['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']]
#     # print(type(cbwd))
#     cbwd = pd.get_dummies(dataset['cbwd'])
#     # print(cbwd)
#     x = pd.concat([x, cbwd], axis=1, join_axes=[x.index])
#     # print(x)
#     # scaler = MinMaxScaler(feature_range=(0, 1))
#     # scaled = scaler.fit_transform(x)
#     # print(scaled)
#     return x,y
# x ,y = load_data()
#
# x = x.values.reshape(x.shape[0], 1, x.shape[1])
# y= y.values.reshape(-1,1)
# print(y)
#
# print(x.shape)
#
# model = Sequential()
# model.add(LSTM(50, input_shape=(x.shape[1], x.shape[2])))
# model.add(Dense(1))
# model.compile(loss='mae', optimizer='adam')
# # fit network
# history = model.fit(x, y, epochs=50, batch_size=72, validation_split=(0.3), verbose=2, shuffle=False)


from pandas import read_csv
from datetime import datetime
# load data
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')
dataset = read_csv('PRSA_data_2010.1.1-2014.12.31.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
print(dataset[:10])
dataset.drop('o', axis=1, inplace=True)
# manually specify column names
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
# mark all NA values with 0
dataset['pollution'].fillna(0, inplace=True)
# drop the first 24 hours
dataset = dataset[24:]
# summarize first 5 rows
print(dataset.head(5))
# save to file
dataset.to_csv('pollution11.csv')

