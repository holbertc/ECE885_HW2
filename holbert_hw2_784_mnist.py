'''Trains a simple deep RNN on the MNIST dataset.
Christopher Holbert
'''
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD

batch_size = 32
nb_epoch = 20
nb_classes = 10

#parameters for LSTM network
nb_lstm_outputs = 100

#load MNIST dataset
(X_train,y_train),(X_test,y_test) = mnist.load_data()

#Reshape input data to 784 pointwise
X_train = X_train.reshape(60000, 784, 1)
X_test = X_test.reshape(10000, 784, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train,nb_classes = nb_classes)
Y_test = np_utils.to_categorical(y_test,nb_classes = nb_classes)

#Build LSTM network
model = Sequential()
model.add(LSTM(
    nb_lstm_outputs,
    input_shape = (784,1),
    #input_dim=1,
    #input_length=784,
    consume_less='mem'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes,activation = 'softmax'))
model.summary()

sgd = SGD(lr=0.1)

model.compile(optimizer = sgd,loss = 'categorical_crossentropy',metrics = ['accuracy'])
history = model.fit(X_train,Y_train,nb_epoch = nb_epoch,batch_size=batch_size,shuffle = True,validation_split = 0.1)
score = model.evaluate(X_test,Y_test)
print 'test loss',score[0]
print 'test accuracy',score[1]