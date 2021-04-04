import os
import keras.activations as constant
n_classes=len(os.listdir("dataset"))
from keras.layers import Convolution2D,Dense,Dropout,MaxPooling2D,Flatten,Activation
import keras.layers
from keras.models import Sequential
from constants import input_dim as shape

def create_model():
    model = Sequential()
    model.add(Convolution2D(32, (3,3), input_shape=shape))
    model.add(Activation(constant.relu))
    model.add(Convolution2D(32,(3,3)))
    model.add(Activation(constant.relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64,(3,3) ))
    model.add(Activation(constant.relu))
    model.add(Convolution2D(64,(3,3)))
    model.add(Activation(constant.relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation(constant.relu))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation(constant.softmax))
    #model.summary()
    model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])
    return model


















def create_model_old():
    import os
    no_of_classes=len(os.listdir("dataset"))
    from keras.layers import Conv2D,Dense,Dropout,MaxPooling2D,Flatten,Activation
    from keras.models import Sequential
    from constants import input_dim
    model=Sequential()
    model.add(Conv2D(32,(3,3),input_shape=input_dim,activation="relu"))
    model.add(Conv2D(32,(3,3),input_shape=input_dim,activation="relu",strides=(3,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64,(3,3),activation="relu"))
    model.add(Conv2D(64,(3,3),activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128,activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(no_of_classes,activation="softmax"))
    model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])
    return model





#create_model()
