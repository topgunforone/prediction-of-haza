from matplotlib import pyplot
from pandas import DataFrame, concat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pandas import read_csv
from math import sqrt
import keras
from keras import Model
from keras.layers.core import Layer
from keras.models import Sequential
from keras.layers import Input,Conv2D,merge ,Lambda,StackedRNNCells,Reshape,GRU,Dense,Concatenate,BatchNormalization,Dropout,Bidirectional,Flatten
from keras.layers import LSTM
from prepare_data import prepare_data
import keras.backend as K
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
# model = Sequential()
from keras.engine import InputSpec
import tensorflow
class Stacked(Layer):
    def __init__(self,filters,kernel_size=[(3,3)],layers = 3,data_format ='channels_last'):
        super(Stacked, self).__init__()
        self.layers = layers
        self.filters = filters
        self.kernel_size = kernel_size
        self.data_format = data_format


    def build(self, input_shape):
        super(Stacked,self).build(input_shape)
    # def __call__(self, inputs,*args, **kwargs):
    #     self.input_shape = inputs._shape_as_list()
    #     super(Stacked,self).__call__(*args,**kwargs)
    def slice(self,x):
        return x[:,-1,...]
    # def call(self, inputs ):
    #     assert len(self.kernel_size) == self.layers,"layer numbers must be same as self.filters numbers!  "
    #     self.ip_shape  = inputs._shape_as_list()                    # [bs, ts, channels,rows,cols]
    #     output = []
    #     final_output = []
    #     if not self.initial_state:
    #         self.initial_state = [None]*self.layers
    #     for i in range(self.layers):
    #         output.append(ConvLSTM2D(self.filters[i], kernel_size=self.kernel_size[i], activation='relu', return_sequences=True,
    #                        data_format='channels_last')(inputs))
    #         inputs = output[-1]
    #         _shape  = inputs._shape_as_list()
    #         final_output.append(Lambda(self.slice)(inputs))
    #         # final_output.append(K.slice(inputs,[0,-1,0,0,0],size = [_shape[0],1,_shape[2],_shape[3],_shape[4]]))
    #     self.output_rows = final_output[-1]._shape_as_list()[-2]
    #     self.output_cols = final_output[-1]._shape_as_list()[-1]
    #     return output, final_output


    def call(self, inputs, initial_states):
        assert len(self.kernel_size) == self.layers,"layer numbers must be same as self.filters numbers!  "
        self.ip_shape  = inputs._shape_as_list()                    # [bs, ts, channels,rows,cols]
        output = []
        final_output = []

        for i in range(self.layers):
            output.append(ConvLSTM2D(self.filters[i], kernel_size=self.kernel_size[i], activation='relu', return_sequences=True,
                           data_format='channels_last')(inputs,initial_state =  initial_states[i] if initial_states else initial_states))
            inputs = output[-1]
            _shape  = inputs._shape_as_list()
            final_output.append(Lambda(self.slice)(inputs))
            # final_output.append(K.slice(inputs,[0,-1,0,0,0],size = [_shape[0],1,_shape[2],_shape[3],_shape[4]]))
        self.output_rows = final_output[-1]._shape_as_list()[-2]
        self.output_cols = final_output[-1]._shape_as_list()[-1]
        # return output, final_output

        return output+final_output



    def compute_output_shape(self, input_shape):
        # return [(self.layers,self.ip_shape[0],self.ip_shape[1],self.filters[-1],\
        #          self.output_rows,self.output_cols),(self.layers,self.ip_shape[0],self.filters[-1],self.output_rows,self.output_cols)]
        return [(self.layers,self.ip_shape[0],self.ip_shape[1],self.filters[-1],self.output_rows,self.output_cols)]*self.layers+\
               [(self.ip_shape[0], self.filters[-1], self.output_rows, self.output_cols)]*self.layers





model_id = 2
if model_id ==1:
    conv_0 = ConvLSTM2D(64, kernel_size=(3, 3), activation='relu',return_sequences=True,input_shape = (batch_size,time_step,10,10,9),data_format= 'channels_last')(input_0)
    # model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(10,10,9),data_format='channel_first'))
    drop_0 = Dropout(0.4)(conv_0)
    conv_1 = ConvLSTM2D(64, kernel_size=(3, 3), activation='relu',return_sequences=True,data_format= 'channels_last')(drop_0)
    drop_1 = Dropout(0.4)(conv_1)
    conv_2 = ConvLSTM2D(1, kernel_size=(3, 3), activation='relu',return_sequences=True,data_format= 'channels_last')(drop_1)#output_size = (6*6*32)
    drop_2 = Dropout(0.4)(conv_2)
    # model.add(Flatten())
    reshape_0 = Reshape(target_shape=(-1,1))(drop_2)
    ################
    #数据拉直代码
    ################

    rnn_0 = LSTM(64, return_sequences=True)(reshape_0)
    rnn_1 = LSTM(64, return_sequences=True)(rnn_0)
    rnn_2 = LSTM(64, return_sequences=True)(rnn_1)#,input_shape=(6, 6, 32)
    flat_0 =  Flatten()(rnn_2)
    output =  Dense(1)(flat_0)
elif model_id ==2:
   stack_output = Stacked(filters=[300,200,100],kernel_size=[(3,3),(3,3),(3,3)],layers = 3,data_format ='channels_last')(input_0,initial_state = None)
   output,hiddent = stack_output[:3] ,stack_output[3:]  #hiddent :
   stack_output = Stacked(filters=[300, 200, 100], kernel_size=[(3, 3), (3, 3), (3, 3)], layers=3, data_format='channels_last',
                          )(input_0,initial_state = hiddent)
   hiddent_1 = stack_output[-1]
   flat = Flatten()(hiddent_1)
   output = Dense(1)(flat)
model = Model(inputs = input_0,outputs = output)
model.compile(loss='mae', optimizer='adam')

history = model.fit(train_x, train_y, epochs=20, batch_size=72, validation_data=(val_x, val_y), verbose=2, shuffle=False)

pyplot.title('loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

from numpy import concatenate
from sklearn.metrics import mean_squared_error
# yhat = model.predict(test_X)#提取的是预测值？

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
#
# print(yhat)
# print(test_y)
# pyplot.plot(yhat[:100], label='Pred')
# pyplot.plot(test_y[:100], label='True')
# pyplot.legend()
# pyplot.show()
#
# i = 0
# count = 0
# while i<len(yhat):
#     if -50<= yhat[i] - test_y[i] <=50:
#         count = count + 1
#     i = i + 1
# print(count/len(yhat))
