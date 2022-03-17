"""
Created on Wed Jun  2 14:09:04 2021

@author: xenificity
"""
#import libraries
from keras.layers import MaxPool2D,Flatten,Dense,Dropout, InputLayer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.preprocessing.image import *
from keras.utils import np_utils
import matplotlib.pylab as plt
import tensorflow as tf
from PIL import Image as im
import pandas as pd
import numpy as np
import cv2
import os

os.chdir("D:/AI test") #setting working directory
#taking file path and labesl
train = ImageDataGenerator(rescale=1/255)
val = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory('training',target_size=(300,300),batch_size=50)
validation_dataset = val.flow_from_directory('validation',target_size=(300,300),batch_size=50)
testing_dataset = test.flow_from_directory('testing',target_size=(300,300))

#Image showing
train_labels= train_dataset.classes
classes = train_dataset.class_indices
img = train_dataset[0][0]
img = np.reshape(img,(300,300,3))
cv2.imshow("image_instance",img)
cv2.waitKey(0) # wait for ay key to exit window
cv2.destroyAllWindows() # close all windows




#Model-1
model = Sequential()
model.add(Conv2D(128,(3,3),activation='swish',input_shape=(300,300,3)))
model.add(MaxPool2D(2,2))
model.add(Conv2D(64,(3,3),activation='swish'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(32,(3,3),activation='swish'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(16,(3,3),activation='swish'))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(264,activation='swish'))
model.add(Dense(6,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
epochs=20
model.summary() 

#Model-2
transfered=InceptionV3(include_top=False,weights='imagenet',input_tensor=None,input_shape=(None,None,3),pooling='avg',classes=6)
model=Sequential()
model.add((InputLayer(None,None,3)))
model.add(transfered)
model.add(Dropout(0.1))
model.add(Dense(6,activation='softmax'))
transfered.trainable=True
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
epochs=100
model.summary()

#Model-3
transfered=EfficientNetB3(include_top=False,weights='imagenet',input_shape=(None,None,3),classes=6)
model=Sequential()
model.add((InputLayer(300,300)))
model.add(transfered)
model.add(Dropout(0.2))
model.add(Dense(6,activation='softmax'))
transfered.trainable=False
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
batch_size=10
epochs=10
model.summary()



model_fit = model.fit(train_dataset,epochs=epochs,validation_data=validation_dataset,callbacks=[ModelCheckpoint(filepath='./weights.hdf5',monitor='val_acc',verbose=1,save_best_only=True)])
model.evaluate(train_dataset[1][1])
model.predict(train_dataset[1][1])

validation_dataset.class_indices
train_dataset.class_indices
img = image.load_img('testing/2015_00401.jpg',target_size=(300,300))
plt.imshow(img)
X = image.img_to_array(img)
a=model.predict(testing_dataset)
c = model.evaluate(train_dataset)
b=testing_dataset.filenames
train_dataset.class_indices
train_dataset.filenames


model.fit(train_dataset,epochs=10,batch_size=10)
















train_data = pd.read_csv('train.csv')
val_data = pd.read_csv('val.csv')
test_data = pd.read_csv('test_set.csv')

#One-hot encoding labels
encoder_1 = LabelEncoder()
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




