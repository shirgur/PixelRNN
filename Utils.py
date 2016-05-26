"""
A Keras implementation of PixelRNN by Van Den Oord et al. Google
arxiv.org/pdf/1601.06759

Copyright Shir Gur, 2016
me@gurshir.com
"""

import theano
from theano import tensor as T
from keras import backend as K
from keras.backend.common import _EPSILON

def image_categorical_crossentropy(output, target, from_logits=False):
    output = T.clip(output, _EPSILON, 1.0 - _EPSILON)
    output_ = K.reshape(output, (-1, 256))
    target_ = K.reshape(target, (-1, 256))
    out = T.nnet.categorical_crossentropy(output_, target_)
    out = K.reshape(out,(K.shape(output)[0],-1))
    return T.mean(T.mean(out, axis=1))

def Skew(inputs):
    inputs_ = K.permute_dimensions(inputs, (3,0,1,2))
    buffer_ = T.zeros((K.shape(inputs)[3], K.shape(inputs)[0], K.shape(inputs)[1]+K.shape(inputs)[3]-1, K.shape(inputs)[2]))

    def fnc(buf, inp, i):
        return T.set_subtensor(buf[:, i:i+K.shape(inputs)[1], :], inp[:,:,:])

    res, _ = theano.scan(fn=fnc, sequences=[buffer_, inputs_, T.arange(K.shape(inputs)[3])])
    res = K.permute_dimensions(res, (1,2,3,0))

    return res


def Unskew(inputs):
    org_height = K.shape(inputs)[1] + 1 - K.shape(inputs)[3]
    inputs_ = K.permute_dimensions(inputs, (3,0,1,2))
    buffer_ = T.zeros((K.shape(inputs)[3], K.shape(inputs)[0], org_height, K.shape(inputs)[2]))

    def fnc(buf, inp, i):
        return T.set_subtensor(buf[:, :, :], inp[:,i:i+org_height,:])

    res, _ = theano.scan(fn=fnc, sequences=[buffer_, inputs_, T.arange(K.shape(inputs)[3])])
    res = K.permute_dimensions(res, (1,2,3,0))

    return res
