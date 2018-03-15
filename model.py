from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization, Cropping2D, Lambda, Activation, Dropout

def nvidia():
  model = Sequential()
  model.add(Cropping2D(cropping=((35,12), (0,0)), input_shape=(80,160,3)))
  model.add(BatchNormalization())
  
  model.add(Conv2D(24, 5, strides=1, padding='valid'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  model.add(Conv2D(36, 5, strides=1, padding='valid'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  model.add(Conv2D(48, 5, strides=1, padding='valid'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  model.add(Conv2D(64, 3, padding='valid'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  model.add(Conv2D(64, 3, padding='valid'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  model.add(Flatten())

  model.add(Dense(100))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  model.add(Dense(50))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  model.add(Dense(10))  
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  
  model.add(Dense(1))
  
  return model