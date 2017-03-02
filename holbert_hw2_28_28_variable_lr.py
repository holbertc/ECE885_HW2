'''Trains a simple deep NN on the MNIST dataset.
Using a variable learning rate as an exponential function
of cost.
Christopher Holbert
'''
from __future__ import print_function
import numpy as np
import keras.callbacks as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler

sd=[]
class LossHistory(K.Callback):
    def on_train_begin(self, logs={}):
        self.losses = [1,1]

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        sd.append(step_decay(len(self.losses)))
        print('learning rate:', step_decay(len(self.losses)))
        print('derivative of loss:', 2*np.sqrt((self.losses[-1])))



def step_decay(losses):
    if float((np.array(temp_history.losses[-1])))<3.0:
        lrate=0.060*np.exp(np.array(temp_history.losses[-1]))
        return lrate
    else:
        lrate=0.01
        return lrate

batch_size = 256
nb_epoch = 50
img_rows,img_cols = 28,28
nb_classes = 10        

#parameters for LSTM network
nb_lstm_outputs = 50
nb_time_steps = img_rows
dim_input_vector = img_cols

#load MNIST dataset
(X_train,y_train),(X_test,y_test) = mnist.load_data()
input_shape = (nb_time_steps,dim_input_vector)
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255
Y_train = np_utils.to_categorical(y_train,nb_classes = nb_classes)
Y_test = np_utils.to_categorical(y_test,nb_classes = nb_classes)

#Build LSTM network
model = Sequential()
model.add(LSTM(
    nb_lstm_outputs,
    input_shape = input_shape,
    consume_less='mem'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes,activation = 'softmax'))#,init = init_weights))
model.summary()




model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


temp_history = LossHistory()
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate, temp_history]


history = model.fit(X_train,Y_train,nb_epoch = nb_epoch,
                    batch_size=batch_size,shuffle = True,
                    validation_split = 0.1,
                    callbacks=callbacks_list)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])