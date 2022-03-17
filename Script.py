# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 14:09:04 2021

@author: AJ
"""

#import libs
import cv2
import os
import random
import matplotlib.pylab as plt
from glob import glob
import pandas as pd
import numpy as np
import pickle
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import keras
from keras.models import *
from keras.layers import *
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import *
from keras.callbacks import EarlyStopping
from keras import regularizers,optimizers
from keras.callbacks import LearningRateScheduler
from keras import *
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3

os.chdir("D:/_IRRI-SOUTH ASIA/personal projects/Activity recognition using tensorflow & keras/activity_recognition")

#Importing training & testing data
train_run = glob(os.path.join('walk_run_train/train_run',"*.png"))
train_walk = glob(os.path.join('walk_run_train/train_walk',"*.png"))
train_data = pd.DataFrame()
train_data['file'] = train_run + train_walk

test_run = glob(os.path.join('walk_run_test/run_test',"*.png"))
test_walk = glob(os.path.join('walk_run_test/walk_test',"*.png"))
test_data = pd.DataFrame()
test_data['file'] = test_run + test_walk


#Adding labels
train_data['label'] = [1 if j in train_run else 0 for j in train_data["file"]]
test_data['label'] = [1 if j in test_run else 0 for j in test_data["file"]]


#Showing image
plt.figure(figsize=(12,12))
plt.imshow(cv2.imread(train_walk[12]))



#data-augmentation
def dataug(files, labels, batch_size=10,randomized=True, random_seed=1):
    randomizer = np.random.RandomState(random_seed)
    img_batch = []
    label_batch = []
    while True:
        ind = np.arange(len(files))
        if randomized:
            randomizer.shuffle(ind)
        for index in ind:
            image = cv2.imread(files[index])[:,:,0:3]/255
            label = labels[index]
            img_batch.append(image)
            label_batch.append(label)
            if len(img_batch) == batch_size:
                yield np.array(img_batch), np.array(label_batch)
                img_batch = []
                label_batch = []
        
        if len(img_batch) > 0:
                yield np.array(img_batch), np.array(label_batch)
                img_batch = []
                label_batch = []


transfered=InceptionV3(include_top=False,weights='imagenet',input_tensor=None,input_shape=(None,None,3),pooling='avg',classes=1000)
model=Sequential()
model.add((InputLayer(None,None,3)))
model.add(transfered)
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))


transfered.trainable=False
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
batch_size=100
epochs=100

model.fit(dataug(train_data['file'],train_data['label'],batch_size=batch_size,randomized=True,random_seed=1),steps_per_epoch=int(np.ceil(len(train_data)/batch_size)), epochs=epochs,
          validation_data=dataug(test_data['file'],test_data['label'],batch_size=batch_size,randomized=True),validation_steps=int(np.ceil(len(test_data)/batch_size)),
          callbacks=[ModelCheckpoint(filepath='./weights.hdf5',monitor='val_loss',verbose=0,save_best_only=True)],
          verbose=2)


transfered.trainable=True
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
batch_size=10
epochs=100

model.fit(dataug(train_data['file'],train_data['label'],batch_size=batch_size,randomized=True,random_seed=1),steps_per_epoch=int(np.ceil(len(train_data)/batch_size)), epochs=epochs,
          validation_data=dataug(test_data['file'],test_data['label'],batch_size=batch_size,randomized=True),validation_steps=int(np.ceil(len(test_data)/batch_size)),
          callbacks=[ModelCheckpoint(filepath='./weights.hdf5',monitor='val_loss',verbose=0,save_best_only=True)],
          verbose=2)
model.load_weights('weights.hdf5')













