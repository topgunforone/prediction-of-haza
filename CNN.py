import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os
from PIL import Image
import numpy as np
from skimage import io, transform
batch_size = 128
num_classes = 4
epochs = 12

def load_data():
    #Return a new array of given shape and type, without initializing entries.
    data = []
    label = []
    #os.listdir(filename)返回filename中所有文件的文件名列表
    classes = os.listdir('image')
    print(classes)
    # print(pd.get_dummies(classes))
    num = len(classes)
    for i in range(num):
        imgs_name = os.listdir('image/'+classes[i])
        print(imgs_name)
        for name in imgs_name:
            import matplotlib.pyplot as plt
        #PIL 的 open() 函数用于创建 PIL 图像对象
            img = io.imread('image/'+classes[i]+'/'+name)
            # io.imshow(img)
            print('image/'+classes[i]+'/'+name)
            print(img.shape)
            # plt.figure('resize')
            #
            # plt.subplot(121)
            # plt.title('before resize')
            # plt.imshow(img, plt.cm.gray)
            img = transform.resize(img, (128, 128))
            print(img.shape)
            # io.imshow(img)
            # plt.subplot(122)
            # plt.title('before resize')
            # plt.imshow(img, plt.cm.gray)
            # plt.show()
            data.append(img)
            label.append(classes[i])
        # print(img)
        #Convert the input to an array
        # arr = np.asarray(img,dtype='float32')
        #
        # data[i,:,:,:] = arr
        # label[i] = imgs[i]
    data = np.array(data)
    label = np.array(label)
    print(data.shape)
    print(label.shape)
    label = pd.get_dummies(label)

    return data,label


data, label = load_data()

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.33)

model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
model.add(Dropout(0.4))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=0.0002),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=400, batch_size=28)

score = model.evaluate(X_test, y_test, batch_size=28)
print(score)