import cv2
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization, Cropping2D, Lambda, Activation, Dropout
from keras.optimizers import Adam
from keras.initializers import glorot_normal
from sklearn.utils import shuffle

def Nvidia_small(dropout=0.0):
  model = Sequential()
  model.add(Cropping2D(cropping=((75,25), (0,0)), input_shape=(160,320,3), name='crop'))
  model.add(BatchNormalization()) # 60 x 320 x 3

  xavier_initializer = glorot_normal(seed=1)
 
  model.add(Conv2D(
    4, 5, strides=(1,2), padding='valid', 
    kernel_initializer=xavier_initializer, bias_initializer='zeros',
    name='conv1'
  )) 
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  
  model.add(Conv2D(
    4, 5, strides=(1,2), padding='valid',
    kernel_initializer=xavier_initializer, bias_initializer='zeros',
    name='conv2'
  )) 
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  
  model.add(Conv2D(
    4, 5, strides=1, padding='valid',
    kernel_initializer=xavier_initializer, bias_initializer='zeros',
    name='conv3'
  )) 
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  
  model.add(Conv2D(
    6, 3, padding='valid',
    kernel_initializer=xavier_initializer, bias_initializer='zeros',
    name='conv4'
  ))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  
  model.add(Conv2D(
    8, 3, padding='valid',
    kernel_initializer=xavier_initializer, bias_initializer='zeros',
    name='conv5'
  ))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  
  model.add(Flatten())

  model.add(Dense(100, kernel_initializer=xavier_initializer, bias_initializer='zeros'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  
  model.add(Dense(50, kernel_initializer=xavier_initializer, bias_initializer='zeros'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(dropout))

  model.add(Dense(10, kernel_initializer=xavier_initializer, bias_initializer='zeros'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  
  model.add(Dense(1, kernel_initializer=xavier_initializer, bias_initializer='zeros'))
  
  return model

def Nvidia(dropout=0.0):
  model = Sequential()
  model.add(Cropping2D(cropping=((75,25), (0,0)), input_shape=(160,320,3), name='crop'))
  model.add(BatchNormalization()) # 60 x 320 x 3
  
  xavier_initializer = glorot_normal(seed=1)
  
  model.add(Conv2D(
    24, 5, strides=(1,2), padding='valid', 
    kernel_initializer=xavier_initializer, bias_initializer='zeros',
    name='conv1'
  )) 
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  
  model.add(Conv2D(
    36, 5, strides=(1,2), padding='valid',
    kernel_initializer=xavier_initializer, bias_initializer='zeros',
    name='conv2'
  )) 
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  
  model.add(Conv2D(
    48, 5, strides=1, padding='valid',
    kernel_initializer=xavier_initializer, bias_initializer='zeros',
    name='conv3'
  )) 
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  
  model.add(Conv2D(
    64, 3, padding='valid',
    kernel_initializer=xavier_initializer, bias_initializer='zeros',
    name='conv4'
  ))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  
  model.add(Conv2D(
    64, 3, padding='valid',
    kernel_initializer=xavier_initializer, bias_initializer='zeros',
    name='conv5'
  ))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  
  model.add(Flatten())

  model.add(Dense(100, kernel_initializer=xavier_initializer, bias_initializer='zeros'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  
  model.add(Dense(50, kernel_initializer=xavier_initializer, bias_initializer='zeros'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  
  model.add(Dense(10, kernel_initializer=xavier_initializer, bias_initializer='zeros'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  
  model.add(Dense(1, kernel_initializer=xavier_initializer, bias_initializer='zeros'))
  
  return model

def processFilename(track, filename):
  filename = filename.split('/')
  filename = '{}/{}/{}'.format(track, filename[-2], filename[-1])
  return filename
# enddef

def generator(data_set, batch_size, track, augment=True):
  N = len(data_set)
  while True:
    data_set = shuffle(data_set)
    for offset in range(0, N, batch_size):
      rows = data_set.iloc[offset:offset+batch_size]
      images = []
      angles = []
      for row in rows.itertuples():
        angle = row.angle
        
        # for center image
        image = plt.imread( processFilename(track, row.center) )
        images.append(image)
        angles.append(angle)
        if augment:
          images.append(cv2.flip(image, 1))
          angles.append(-angle)

        # for left image
        image = plt.imread( processFilename(track, row.left) )
        images.append(image)
        angles.append(angle + 0.2)
        if augment:
          images.append(cv2.flip(image, 1))
          angles.append(-angle - 0.2)
        
        # for right image
        image = plt.imread( processFilename(track, row.right) )
        images.append(image)
        angles.append(angle - 0.2)
        if augment:
          images.append(cv2.flip(image, 1))
          angles.append(-angle + 0.2)
        
      #end for
      images = np.array(images)
      angles = np.array(angles)
      
      yield shuffle(images, angles)
    #end for
  #end while
#end def 

if __name__ == '__main__':
    
  parser = argparse.ArgumentParser(description='Remote Driving')
  parser.add_argument(
    '--track',
    type=str,
    default='all_data',
    help='Path to track data. (default: all_data)'
  )
  
  parser.add_argument(
    '--no-validation', 
    action="store_true",
    default=False,
    help='off validation and use all data for training. (default: False)'
  )

  parser.add_argument(
    '--model-params',
    type=str,
    help='Path to previous trained model. (default: None)'
  )

  parser.add_argument(
    '--batch-size',
    type=int,
    default=50,
    help='Batch size. (default: 50)'
  )

  parser.add_argument(
    '--epochs',
    type=int,
    default=80,
    help='Number of epochs. (default: 80)'
  )

  parser.add_argument(
    '--lr',
    type=float,
    default=1e-3,
    help='Learning rate. (default: 1e-3)'
  )

  args = parser.parse_args()
  print(args)

  # read the driving_log.csv file
  driving_log = pd.read_csv(
      '{}/driving_log.csv'.format(args.track), 
      names=['center','left','right','angle','throttle','brake','speed']
  )
  # only retain the required columns
  center_left_right_angle = driving_log[['center', 'left', 'right', 'angle']]

  np.random.seed(1) # set the random number seed

  # no validation is to test whether we can fit to the training set
  if args.no_validation:
    # center_left_right_angle contains all the rows
    train_set = center_left_right_angle 
  else:
    npts = len(center_left_right_angle)
    # split into training and validation with a 0.8, 0.2 split
    
    npts_rand = np.random.rand(npts)
    train_set = center_left_right_angle[npts_rand <= 0.8]
    valid_set = center_left_right_angle[npts_rand >  0.8]
    
    valid_generator = generator(valid_set, args.batch_size, args.track)
    validation_steps = np.rint(len(valid_set) / args.batch_size).astype(int)
  # endif

  train_generator = generator(train_set, args.batch_size, args.track)
  steps_per_epoch  = np.rint(len(train_set) / args.batch_size).astype(int)
  
  if args.model_params is None:
    # use the architecture that generates smaller parameters file
    model = Nvidia_small(dropout=0.25)
  else:
    # if parameters file is provided, load it and resume training from there
    model = load_model(args.model_params)
  # end if
  optimizer = Adam(lr=args.lr)
  model.compile(loss='mse', optimizer=optimizer)

  if args.no_validation:
    model.fit_generator(
      train_generator, 
      steps_per_epoch=steps_per_epoch, 
      epochs=args.epochs
    )
  else:
    model.fit_generator(
      train_generator, 
      steps_per_epoch=steps_per_epoch, 
      epochs=args.epochs, 
      validation_data=valid_generator, 
      validation_steps=validation_steps
    )

  model.save('params/{}_model.h5'.format(args.track))
  # model.save_weights('params/{}_model_weights.h5'.format(args.track))
