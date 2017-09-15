import pickle
import tensorflow as tf
import numpy as np
# TODO: import Keras layers you need here
import csv
import os
import cv2
import sklearn

import matplotlib.pyplot as plt
from keras.layers import Input, Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
from keras.models import Model, Sequential
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2, activity_l2
from keras.layers.advanced_activations import ELU
import random

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 10, "The number of epochs.")
flags.DEFINE_integer('batchSize', 64, "The batch size.")

#Input data
samples = []

# Reading in data from csv file
with open("data3/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
   
# Splitting data into training and validation 
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)       
    

def crop_image(image, top_percent=0.36, bottom_percent=0.15):
    """
    Function to crop the image by specified percentage from top and bottom
    """
    
    top = int(np.ceil(image.shape[0] * top_percent))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))
    return image[top:bottom, :]

### Feference for brightness and trans functions taken from some other submission
### https://github.com/toluwajosh/CarND-Behavioural-Cloning

def augment_brightness(image):
    	"""
    	apply random brightness on the image
    	"""
    	image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    	random_bright = .25+np.random.uniform()
    	
    	# scaling up or down the V channel of HSV
    	image[:,:,2] = image[:,:,2]*random_bright
    	return image
    
    
def trans_image(image,steer,trans_range=50, trans_y=False):
    
    
    
    """
	   translate image and compensate for the translation on the steering angle
    """
        
    rows, cols, chan = image.shape
    # horizontal translation with 0.008 steering compensation per pixel
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*.4
    
    # option to disable vertical translation (vertical translation not necessary)
    if trans_y:
        tr_y = 40*np.random.uniform()-40/2
    else:
        tr_y = 0
    	
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    	
    return image_tr,steer_ang
    
    
def im_process(image, measurmement_angle, train=True):
    """"
    Apply Processing to the image
    """
    
    # translate image and compensate for steering angle
    image, measurmement_angle = trans_image(image, measurmement_angle) # , trans_y=True
    
    # crop image region of interest
    image = crop_image(image,0.35,0.12)
    
    # flip image (randomly)
    if np.random.uniform()>= 0.5: 
        image = cv2.flip(image, 1)
        measurmement_angle = -measurmement_angle
    
    # augment brightness
    image = augment_brightness(image)
    
    #Resize the images to fit to NVIDIA CNN architecture
    image = cv2.resize(image, (200,66))
    
    return image, measurmement_angle
    
def generatorTrain(samples, batch_size=FLAGS.batchSize):
    num_samples=len(samples)
    path = "data3/IMG/"
    while 1:
        
        sklearn.utils.shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            
            batch_samples =  samples[offset:offset+batch_size]
            
            images = []
            angles = []
            #count_zeroangle = 0
            for batch_sample in batch_samples:
                
                
                cam_direction = random.randint(1,3)
              
                
                if cam_direction == 1:
                    ## Centre Image
                    image = plt.imread(path+batch_sample[0].split("\\")[-1])
                    image, measurement_angle = im_process(image, float(batch_sample[3])+.25)
  
                elif cam_direction == 2: 
                    ## Left Image                       
                    image = plt.imread(path+batch_sample[1].split("\\")[-1])
                    image, measurement_angle = im_process(image, float(batch_sample[3]))
   
                elif cam_direction == 3:
                    ## Right Image
                    image = plt.imread(path+batch_sample[2].split("\\")[-1])
                    image, measurement_angle = im_process(image, float(batch_sample[3])-.25)
   
                #Appending the angles and images to array
                images.append(image)
                angles.append(measurement_angle)
   
            
            X_train = np.array(images)
            Y_train = np.array(angles)
        
            yield sklearn.utils.shuffle(X_train, Y_train)
        
def generatorValidate(samples, batch_size=FLAGS.batchSize):
    num_samples=len(samples)
    path = "data3/IMG/"
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples =  samples[offset:offset+batch_size]
            
            images = []
            angles = []
            #count_zeroangle = 0
            for batch_sample in batch_samples:
                
                           
                image = plt.imread(path+batch_sample[0].split("\\")[-1])
                measurement_angle = float(batch_sample[3])
                #crop region of interest and resize to model input size
                crop_image(image,0.35,0.12)
                #Resize the images to fit to NVIDIA CNN architecture
                image = cv2.resize(image, (200,66))
                #change colourspace
                image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
                
                #Appending the angles and images to array
                images.append(image)
                angles.append(measurement_angle)
           
            
            X_train = np.array(images)
            Y_train = np.array(angles)
        
            yield sklearn.utils.shuffle(X_train, Y_train)        


def Train_model():
    model = Sequential()
            
      # Preprocess incoming data, centered around zero with small standard deviation               
    model.add(Lambda(lambda x : x / 127.5 - 1. ,input_shape=(66,200,3),output_shape=(66,200,3)))
     # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dropout(0.2))
    
    # Add two 3x3 convolution layers (output depth 64, and 64)
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    
    
    # Add four fully connected layers (depth 100, 50, 10)
    
    model.add(Dense(100))
    model.add(ELU())
    model.add(Dropout(0.2))
    
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dropout(0.2))
    
    model.add(Dense(10))
    model.add(ELU())
    
    
    # Add a fully connected output layer
    model.add(Dense(1))
    return model



            
train_generator = generatorTrain(train_samples, batch_size=FLAGS.batchSize)
validation_generator = generatorValidate(validation_samples, batch_size = FLAGS.batchSize)

model = Train_model()
model.compile(loss='mse',optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=FLAGS.epochs,verbose=2)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

#Save the Keras model to be able to use on test data
model.save('model.h5')

exit()






