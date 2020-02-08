# Waste segregation or classification
# By Team : One last time

# Importing libraries required for the process.
from flask import Flask, render_template, request, Response
from werkzeug import secure_filename
from PIL import Image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array, array_to_img

from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
from keras.models import Sequential

import keras as K
import asyncio, pickle
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from PyQt5 import QtCore, QtGui, QtWidgets
import serial  # add serial port lib. 
import time, threading
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import load_model
from keras.layers import Layer
from keras.utils import conv_utils
import matplotlib.pyplot as plt
import os, sys, math, random, base64
import tensorflow as tf

global ser
ser = serial.Serial('COM24', baudrate=9600, timeout=10,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE,bytesize=serial.EIGHTBITS)
base_path = "Dataset"           # To connect to the serial monitor of arduino.
base_path1 = "new.jpeg"
app = Flask(__name__)           # Flask framework for post request

##############################################

# Data generator function of keras for train data
train_datagen = ImageDataGenerator(     # Data Augmentation , Data Preprocessing for training data
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        height_shift_range=0.1,
        width_shift_range=0.1,
        vertical_flip=True,
        horizontal_flip=True,
        validation_split=0.1
    )

# Data generator function of keras for validation data
test_datagen = ImageDataGenerator(     # Data Augmentation , Data Preprocessing for testing data
        rescale=1./255,
        validation_split=0.15
    )

train_generator = train_datagen.flow_from_directory(  #Reading training data from directory.
        base_path,
        target_size=(300, 300),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        seed=0
    )

validation_generator = test_datagen.flow_from_directory(  #Reading testing data from directory
        base_path,
        target_size=(300, 300),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        seed=0
    )
# Find the required labels of the dataset (here --2)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())   
print(labels)   

##############################################
# Our main architecture for classification
# CNN model definition

model = Sequential([
    # Feature extractor part
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(300, 300, 3)),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    # Classification part
    Flatten(),                      #Flattening
    Dense(64, activation='relu'),       
    Dense(2, activation='softmax')
    ])


model.load_weights("model1.h5")         #Loading the weights  in the model.

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['boss']
        f.save(secure_filename("bhai.jpg"))
        im = Image.open("bhai.jpg")
        np_im = np.array(im)
        print(np_im)
       
        return 'tr'
@app.route("/")
def hello():

    return "Hello World!"

@app.route('/uploader2', methods = ['GET', 'POST'])     #creating post route to accept the image from phone and store it in computer
def upload_file2(): 
     if request.method == 'POST':
      print("...................................................")       
      #print(request.files['boss'])
      #f= open("guru99.txt","w+")
      #f.write(request.files['boss'])
      #f.close();
     if ser.inWaiting():
        l=ser.readline().decode("UTF-8")
        c=l.split()
        print(l)
      
     data = dict(request.form)
     img=data['boss']
     imgdata = base64.b64decode(img)
     filename = 'bhai.jpeg'  
     with open(filename, 'wb') as f:
      f.write(imgdata)
    #  im = Image.open("bhai.jpeg")
    #  np_im = np.array(im)
    #  print(np_im.shape)
     im1 = Image.open(filename)
     width = 300
     height = 300
     im5 = im1.resize((width, height), Image.ANTIALIAS)    # Down-sizing of image to 300x300
     im5.save("new.jpeg")
     numarr= np.array(im5)
     print(numarr.shape)
    
     with graph.as_default():
        # test_x = test_generator.__getitem__(1)
      test_x =  load_img(base_path1)
      test_x = img_to_array(test_x, dtype=np.uint8)     #processing on received data and then  sending it to model for predicting
      print(test_x.shape)
      test_x = np.expand_dims(test_x, axis=0)
      print(test_x.shape)
     
      out = model.predict(test_x)
      print(out)
      # Printing of test images with prediction labels
        # plt.figure(figsize=(16, 16))
        # for i in range(16):
        #     plt.subplot(4, 4, i+1)
        #     plt.title('pred:%s ' % (labels[np.argmax(out[i])]))
        #     plt.imshow(test_x[i])
        #     plt.show()
      print(out.shape)
      if out[0][0]==1:                  # As the model generates result the arduino receives b for biodegradable product
        ser.write(b'b')
        l=ser.readline().decode("UTF-8")
        c=l.split()
        print(l)
      else:                             # As the model generates result the arduino receives b for non biodegradable product
        ser.write(b'n')
        l=ser.readline().decode("UTF-8")
        c=l.split()
        print(l)

# Main function 
if __name__ == "__main__":
    
    global graph
    graph = tf.get_default_graph()
    app.run('0.0.0.0',3000)
    