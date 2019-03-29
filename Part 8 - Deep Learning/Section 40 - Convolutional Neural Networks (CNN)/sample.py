# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 19:43:56 2019

@author: Shubham
"""

#part-1 Building CNN

#importing libraries
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense

#initialization
classifier = Sequential()

#step 1(convolution)
classifier.add(Convolution2D(32,3,3, input_shape = (64,64,3), activation='relu'))
#step 2(pooling)
classifier.add(MaxPooling2D())

#adding second layer
classifier.add(Convolution2D(32,3,3, activation='relu'))
classifier.add(MaxPooling2D())

#step 3(flattening)
classifier.add(Flatten())
#step 4(full connection)
classifier.add(Dense(units = 128, activation= 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid' ))

#compiling CNN
classifier.compile(optimizer='adam' , loss= 'binary_crossentropy', metrics=['accuracy'])

#part2 fitting CNN
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

classifier.fit_generator(
    train_set,
    steps_per_epoch=8000,
    epochs=5,
    validation_data=test_set,
    validation_steps=2000)