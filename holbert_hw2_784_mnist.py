from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras import initializations
def init_weights(shape,name=None):
    return initializations.normal(shape,scale=0.01,name = name)

batch_size = 256
nb_epoch = 10
nb_classes = 10

#parameters for LSTM network
nb_lstm_outputs = 100

#load MNIST dataset
(X_train,y_train),(X_test,y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], -1, 1)
X_test = X_test.reshape(X_test.shape[0], -1, 1)

print X_train.shape
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255
Y_train = np_utils.to_categorical(y_train,nb_classes = nb_classes)
Y_test = np_utils.to_categorical(y_test,nb_classes = nb_classes)

#Build LSTM network
model = Sequential()
model.add(LSTM(
    nb_lstm_outputs,
    input_shape = (784,1)))
model.add(Dense(nb_classes,activation = 'softmax',init = init_weights))
model.summary()
model.compile(optimizer = 'rmsprop',loss = 'categorical_crossentropy',metrics = ['accuracy'])
history = model.fit(X_train,Y_train,nb_epoch = nb_epoch,batch_size=batch_size,shuffle = True,validation_split = 0.1)
score = model.evaluate(X_test,Y_test)
print 'test loss',score[0]
print 'test accuracy',score[1]