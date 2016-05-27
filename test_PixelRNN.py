"""
A Keras implementation of PixelRNN by Van Den Oord et al. Google
arxiv.org/pdf/1601.06759

Copyright Shir Gur, 2016
me@gurshir.com
"""


import theano
import theano.sandbox.cuda
#theano.sandbox.cuda.use("gpu0")

import pickle
import os

from scipy.misc import imread

from keras.datasets import cifar10

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import LearningRateScheduler
from keras.models import Model
from keras.layers import Input, Convolution2D
from keras.engine import merge
from keras.optimizers import RMSprop
from Layers import MaskedConvolution2D, SoftmaxLayer, GetColors, DiagLSTM, PyramidSTM
from Utils import  image_categorical_crossentropy

class PixelRNN(object):
    def __init__(self):

        self.img_rows, self.img_cols = 32, 32
        self.img_channels = 3

        self.margin = 10
        self.batch_size = 10

        self.h = 300
        self.h_2 = self.h//2

        rms = RMSprop(lr=(1e-6))
        self.optimizer = rms

        self.Xtrain = []
        self.Rtrain = []
        self.Gtrain = []
        self.Btrain = []


        self.comp_net = None

        self.build_data_cifar10()

    def build_data_cifar10(self):
        print("Loading data...")

        (self.Xtrain, _), (_, _) = cifar10.load_data()
        self.Xtrain = self.Xtrain[:501]

        self.extra = np.shape(self.Xtrain)[0] % self.batch_size

        Ytrain = self.Xtrain.copy()
        self.Xtrain[:,:,:,-self.margin:] = 0

        RGB_Shapr = (np.shape(self.Xtrain)[0], np.shape(self.Xtrain)[2], np.shape(self.Xtrain)[3], 256)
        RGB_Shapr_train = (np.shape(self.Xtrain)[0], np.shape(self.Xtrain)[2]*np.shape(self.Xtrain)[3], 256)
        self.Rtrain = np.zeros(RGB_Shapr, dtype=self.Xtrain.dtype)
        self.Gtrain = np.zeros(RGB_Shapr, dtype=self.Xtrain.dtype)
        self.Btrain = np.zeros(RGB_Shapr, dtype=self.Xtrain.dtype)

        for s in range(np.shape(self.Xtrain)[0]):
            for j in range(3):
                for i in range(self.img_rows):
                    for k in range(self.img_cols):
                        if j == 0:
                            self.Rtrain[s, i, k, Ytrain[s,j,i,k]] = 1
                        elif j == 1:
                            self.Gtrain[s, i, k, Ytrain[s,j,i,k]] = 1
                        else:
                            self.Btrain[s, i, k, Ytrain[s,j,i,k]] = 1

        self.Rtrain = np.reshape(self.Rtrain,RGB_Shapr_train)
        self.Gtrain = np.reshape(self.Gtrain,RGB_Shapr_train)
        self.Btrain = np.reshape(self.Btrain,RGB_Shapr_train)

    def build_net_DiagLSTM(self, load_weights = False):

        img = Input(batch_shape=(10, self.img_channels, self.img_rows, self.img_cols), name='input_img')

        model_in = MaskedConvolution2D(self.h,7,7,mask_type='a', direction='Right', border_mode='same', init='he_uniform')(img)

        for _ in range(1):
            model_LSTM_F = DiagLSTM(self.h_2,3, return_sequences=True, init='he_uniform', inner_init='he_uniform', direction='Right')(model_in)
            model_LSTM_B = DiagLSTM(self.h_2,3, return_sequences=True, init='he_uniform', inner_init='he_uniform', direction='Right', reverse=True)(model_in)
            model_LSTM = merge([model_LSTM_F, model_LSTM_B], mode='sum')
            model_per = Convolution2D(self.h,1,1, init='he_normal')(model_LSTM)
            model_in = merge([model_in, model_per], mode='sum')

        model_out = MaskedConvolution2D(self.h,1,1,mask_type='b', direction='Right', border_mode='same', activation='relu', init='he_uniform')(model_in)
        model_out = MaskedConvolution2D(256*3,1,1,mask_type='b', direction='Right', border_mode='same', activation='relu', init='he_uniform')(model_out)

        Red = GetColors(0)(model_out)
        Green = GetColors(1)(model_out)
        Blue = GetColors(2)(model_out)

        Red_out = SoftmaxLayer(name='Red_out')(Red)
        Green_out = SoftmaxLayer(name='Green_out')(Green)
        Blue_out = SoftmaxLayer(name='Blue_out')(Blue)

        Col_Model = Model(img, [Red_out, Green_out, Blue_out])

        if load_weights:
            Col_Model.load_weights('Data/comp_model.h5')

        print("Compiling...")
        Col_Model.compile(optimizer=self.optimizer,
                              loss={'Red_out': image_categorical_crossentropy,
                                    'Green_out': image_categorical_crossentropy,
                                    'Blue_out': image_categorical_crossentropy},
                              metrics={'Red_out': 'accuracy',
                                       'Green_out': 'accuracy',
                                       'Blue_out': 'accuracy'})
        self.comp_net = Col_Model

    def build_net_PyramidSTM(self, load_weights = False):

        img = Input(batch_shape=(10, self.img_channels, self.img_rows, self.img_cols), name='input_img')

        model_in = MaskedConvolution2D(self.h,7,7,mask_type='a', direction='Right', border_mode='same', init='he_uniform')(img)

        for _ in range(2):
            model_LSTM = PyramidSTM(self.h_2,3, return_sequences=True, init='he_uniform', inner_init='he_uniform', direction='Right')(model_in)
            model_per = Convolution2D(self.h,1,1, init='he_normal')(model_LSTM)
            model_in = merge([model_in, model_per], mode='sum')

        model_out = MaskedConvolution2D(self.h,1,1,mask_type='b', direction='Right', border_mode='same', activation='relu', init='he_uniform')(model_in)
        model_out = MaskedConvolution2D(256*3,1,1,mask_type='b', direction='Right', border_mode='same', activation='relu', init='he_uniform')(model_out)

        Red = GetColors(0)(model_out)
        Green = GetColors(1)(model_out)
        Blue = GetColors(2)(model_out)

        Red_out = SoftmaxLayer(name='Red_out')(Red)
        Green_out = SoftmaxLayer(name='Green_out')(Green)
        Blue_out = SoftmaxLayer(name='Blue_out')(Blue)

        Col_Model = Model(img, [Red_out, Green_out, Blue_out])

        if load_weights:
            Col_Model.load_weights('Data/comp_model.h5')

        print("Compiling...")
        Col_Model.compile(optimizer=self.optimizer,
                              loss={'Red_out': image_categorical_crossentropy,
                                    'Green_out': image_categorical_crossentropy,
                                    'Blue_out': image_categorical_crossentropy},
                              metrics={'Red_out': 'accuracy',
                                       'Green_out': 'accuracy',
                                       'Blue_out': 'accuracy'})
        self.comp_net = Col_Model

    def build_net_CNN(self, load_weights = False):

        img = Input(batch_shape=(10, self.img_channels, self.img_rows, self.img_cols), name='input_img')

        model_in = MaskedConvolution2D(self.h,7,7,mask_type='a', direction='Right', border_mode='same', init='he_uniform')(img)

        for _ in range(2):
            model_in = MaskedConvolution2D(self.h,3,3,mask_type='b', direction='Right', border_mode='same', init='he_uniform')(model_in)

        model_out = MaskedConvolution2D(self.h,1,1,mask_type='b', direction='Right', border_mode='same', activation='relu', init='he_uniform')(model_in)
        model_out = MaskedConvolution2D(256*3,1,1,mask_type='b', direction='Right', border_mode='same', activation='relu', init='he_uniform')(model_out)

        Red = GetColors(0)(model_out)
        Green = GetColors(1)(model_out)
        Blue = GetColors(2)(model_out)

        Red_out = SoftmaxLayer(name='Red_out')(Red)
        Green_out = SoftmaxLayer(name='Green_out')(Green)
        Blue_out = SoftmaxLayer(name='Blue_out')(Blue)

        Col_Model = Model(img, [Red_out, Green_out, Blue_out])

        if load_weights:
            Col_Model.load_weights('Data/comp_model.h5')

        print("Compiling...")
        Col_Model.compile(optimizer=self.optimizer,
                              loss={'Red_out': image_categorical_crossentropy,
                                    'Green_out': image_categorical_crossentropy,
                                    'Blue_out': image_categorical_crossentropy},
                              metrics={'Red_out': 'accuracy',
                                       'Green_out': 'accuracy',
                                       'Blue_out': 'accuracy'})
        self.comp_net = Col_Model

    def fit_net_DiagLSTM(self, batch_size=None, nb_epoch=5):
        if self.comp_net is None:
            self.build_net_DiagLSTM(load_weights=False)
        if batch_size is not None:
            self.batch_size = batch_size
        # Fit
        print("Fitting...")
        self.comp_net.fit({'input_img':self.Xtrain[:-self.extra]},
                          {'Red_out':self.Rtrain[:-self.extra],
                           'Green_out':self.Gtrain[:-self.extra],
                           'Blue_out':self.Btrain[:-self.extra]},
                  batch_size=self.batch_size, nb_epoch=nb_epoch, verbose=1,
                          callbacks=[LearningRateScheduler(self.scd)])

        # Save weights
        print("Saving weights...")
        self.comp_net.save_weights('Data/comp_model.h5', overwrite=True)

    def fit_net_PyramidSTM(self, batch_size=None, nb_epoch=5):
        if self.comp_net is None:
            self.build_net_PyramidSTM(load_weights=False)
        if batch_size is not None:
            self.batch_size = batch_size
        # Fit
        print("Fitting...")
        self.comp_net.fit({'input_img':self.Xtrain[:-self.extra]},
                          {'Red_out':self.Rtrain[:-self.extra],
                           'Green_out':self.Gtrain[:-self.extra],
                           'Blue_out':self.Btrain[:-self.extra]},
                  batch_size=self.batch_size, nb_epoch=nb_epoch, verbose=1,
                          callbacks=[LearningRateScheduler(self.scd)])

        # Save weights
        print("Saving weights...")
        self.comp_net.save_weights('Data/comp_model.h5', overwrite=True)

    def fit_net_CNN(self, batch_size=None, nb_epoch=5):
        if self.comp_net is None:
            self.build_net_CNN(load_weights=False)
        if batch_size is not None:
            self.batch_size = batch_size
        # Fit
        print("Fitting...")
        self.comp_net.fit({'input_img':self.Xtrain[:-self.extra]},
                          {'Red_out':self.Rtrain[:-self.extra],
                           'Green_out':self.Gtrain[:-self.extra],
                           'Blue_out':self.Btrain[:-self.extra]},
                  batch_size=self.batch_size, nb_epoch=nb_epoch, verbose=1,
                          callbacks=[LearningRateScheduler(self.scd)])

        # Save weights
        print("Saving weights...")
        self.comp_net.save_weights('Data/comp_model.h5', overwrite=True)

    def scd(self,x):
        if x%5 == 0:
            return float(0.1)
        if x%3 == 0:
            return float(1./x)
        return 0.001

p = PixelRNN()
p.fit_net_PyramidSTM()


predictions = p.comp_net.predict({'input_img':p.Xtrain[0:p.batch_size]})

result = np.concatenate([np.reshape(np.argmax(np.reshape(
        predictions[i][9], (p.img_rows,p.img_cols,256)),axis=-1), (p.img_rows,p.img_cols,1))
                     .astype(dtype=p.Xtrain.dtype)
                      for i in range(3)], axis=-1)

train = np.concatenate([np.reshape(np.argmax(np.reshape(p.Rtrain[9], (p.img_rows,p.img_cols,256)),axis=-1), (p.img_rows,p.img_cols,1)).astype(dtype=p.Xtrain.dtype),
                       np.reshape(np.argmax(np.reshape(p.Gtrain[9], (p.img_rows,p.img_cols,256)),axis=-1), (p.img_rows,p.img_cols,1)).astype(dtype=p.Xtrain.dtype),
                       np.reshape(np.argmax(np.reshape(p.Btrain[9], (p.img_rows,p.img_cols,256)),axis=-1), (p.img_rows,p.img_cols,1)).astype(dtype=p.Xtrain.dtype)], axis=-1)
plt.figure(1)
plt.imshow(result)
plt.figure(2)
plt.imshow(train)
plt.show()