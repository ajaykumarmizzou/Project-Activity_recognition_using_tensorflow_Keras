"""
Created on Wed Jun  2 14:09:04 2021

@author: xenificity
"""
#import libraries
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
from keras.utils import np_utils
from keras.layers import *
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import *
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from keras import regularizers,optimizers
from keras.callbacks import LearningRateScheduler
from keras import *
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
os.chdir("D:/AI test") #setting working directory
#taking file path and labesl
train_data = pd.read_csv('train.csv')
val_data = pd.read_csv('val.csv')
test_data = pd.read_csv('test_set.csv')

#One-hot encoding labels
encoder_1 = LabelEncoder()
encoder = LabelEncoder()
encoder.fit(train_data['Class'])
encoder_1.fit(val_data['Class'])
encoded_Y = encoder.transform(train_data['Class'])
encoded_Y_val = encoder_1.transform(val_data['Class'])
trainy = np_utils.to_categorical(encoded_Y)
val_y = np_utils.to_categorical(encoded_Y_val)

#Importing training & testing data
train_Bicycle = pd.concat([pd.DataFrame(glob(os.path.join('Training/Bicycle',"*.png"))), pd.DataFrame(glob(os.path.join('Training/Bicycle',"*.jpg")) )])
train_Boat = pd.concat([pd.DataFrame(glob(os.path.join('Training/Boat',"*.jpg"))),pd.DataFrame(glob(os.path.join('Training/Boat',"*.png")))]) 
train_Cat = pd.concat([pd.DataFrame(glob(os.path.join('Training/Cat',"*.jpg"))) , pd.DataFrame(glob(os.path.join('Training/Cat',"*.png")))])
train_Motorbike = pd.concat([pd.DataFrame(glob(os.path.join('Training/Motorbike',"*.jpg"))),pd.DataFrame(glob(os.path.join('Training/Motorbike',"*.png")))])
train_People = pd.concat([pd.DataFrame(glob(os.path.join('Training/People',"*.jpg"))),pd.DataFrame(glob(os.path.join('Training/People',"*.JPEG")))])
train_Table =pd.concat( [pd.DataFrame(glob(os.path.join('Training/Table',"*.jpg"))),pd.DataFrame(glob(os.path.join('Training/Table',"*.png")))])
train_data = pd.DataFrame()
train_data = pd.concat([train_Bicycle, train_Boat, train_Cat, train_Motorbike ,train_People , train_Table ])
#del train_bicycle_png, train_Bicycle,train_Boat,train_Boat_png,train_Cat,train_Cat_png,train_Motorbike,train_Motorbike_png,train_People,train_People_JPEG,train_Table,train_Table_png

test_data = pd.DataFrame()
test_jpg = glob(os.path.join('Testing/',"*.jpg"))
test_png = glob(os.path.join('Testing/',"*.png"))
test_JPEG = glob(os.path.join('Testing/',"*.jPEG"))
test_data['file'] = test_jpg + test_png + test_JPEG
#del test_jpg, test_png,test_JPEG


count = 0
#Adding labels
for i in train_data:
    print(i)
    for j in train_Bicycle:
        if(i==j):
            count = count + 1
    
    else:
        print('no')
        
train_data['label'] = [1 if j in train_run else 0 for j in train_data["file"]]
test_data['label'] = [1 if j in test_run else 0 for j in test_data["file"]]

#Showing image
plt.figure(figsize=(12,12))
plt.imshow(cv2.imread(train_walk[12]))

train_Bicycle.columns = ['test']

#data-augmentation
def dataug(files,labels, batch_size=10,randomized=True, random_seed=1):
    randomizer = np.random.RandomState(random_seed)
    img_batch = []
    label_batch = []
    while True:
        ind = np.arange(len(files))
        if randomized:
            randomizer.shuffle(ind)
        for index in ind:
            image = cv2.imread(files[index])
            label = labels[index]
            img_batch.append(image)
            label_batch.append(label)
            yield np.array(img_batch), np.array(label_batch)
            img_batch = []
            label_batch = []
        if len(img_batch) > 0:
                yield np.array(img_batch), np.array(label_batch)
                img_batch = []
                label_batch = []

#data-augmentation
def dataug_test(files, batch_size=2,randomized=True, random_seed=1):
    randomizer = np.random.RandomState(random_seed)
    img_batch = []
    #label_batch = []
    while True:
        ind = np.arange(len(files))
        if randomized:
            randomizer.shuffle(ind)
        for index in ind:
            image = cv2.imread(files[index])[:,:,0:3]/255
            #label = labels[index]
            img_batch.append(image)
            #label_batch.append(label)
            yield np.array(img_batch)#, np.array(label_batch)
            img_batch = []
            #label_batch = []
        if len(img_batch) > 0:
                yield np.array(img_batch)#, np.array(label_batch)
                img_batch = []
               # label_batch = []


transfered=InceptionV3(include_top=False,weights='imagenet',input_tensor=None,input_shape=(None,None,3),pooling='avg',classes=1000)
model=Sequential()
model.add((InputLayer(None,None,3)))
model.add(transfered)
model.add(Dropout(0.2))
model.add(Dense(6,activation='softmax'))
transfered.trainable=False
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
batch_size=10
epochs=10
model.summary()
model.fit(dataug(train_data['Image_Path'],trainy,batch_size=batch_size,randomized=True,random_seed=1),steps_per_epoch=int(np.ceil(len(train_data)/batch_size)), epochs=epochs,
          validation_data=dataug(val_data['Image_Path'],val_y,batch_size=batch_size,randomized=True),validation_steps=int(np.ceil(len(val_data)/batch_size)),
          callbacks=[ModelCheckpoint(filepath='./weights.hdf5',monitor='acc',verbose=1,save_best_only=True)],
          verbose=2)


model.load_weights('weights.hdf5')

model.evaluate(dataug(test_data['Image_Path']))





















train_data['file'][1,1]









transfered.trainable=False
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
batch_size=100
epochs=100

model.fit(dataug(train_data['file'],trainy,batch_size=batch_size,randomized=True,random_seed=1),steps_per_epoch=int(np.ceil(len(train_data)/batch_size)), epochs=epochs,
          validation_data=dataug(test_data['file'],test_data['label'],batch_size=batch_size,randomized=True),validation_steps=int(np.ceil(len(test_data)/batch_size)),
          callbacks=[ModelCheckpoint(filepath='./weights.hdf5',monitor='val_loss',verbose=0,save_best_only=True)],
          verbose=2)




