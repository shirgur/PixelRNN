"""
A Keras implementation of PixelRNN by Van Den Oord et al. Google
arxiv.org/pdf/1601.06759

Shir Gur BGU
"""

import numpy as np
import theano
from theano import tensor as T
from keras.engine import Layer, InputSpec
from keras import backend as K
from keras.backend.common import  _FLOATX
from keras import activations, initializations, regularizers, constraints
import keras.layers.convolutional as K_conv
import Utils
from keras.layers import Permute

class ColRecurrent(Layer):
    def __init__(self, weights=None, reverse=False,
                 return_sequences=False, go_backwards=False, stateful=False,
                 nb_filter=None, filter_length=None,
                 unroll=False, consume_less='cpu',
                 input_dim=None, input_length=None, **kwargs):
        self.return_sequences = return_sequences
        self.initial_weights = weights
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self.consume_less = consume_less
        self.reverse = reverse

        self.nb_filter = nb_filter
        self.filter_length = filter_length

        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=4)]
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(ColRecurrent, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.nb_filter, input_shape[3])
        else:
            return (input_shape[0], self.nb_filter, input_shape[3])

    def compute_mask(self, input, mask):
        if self.return_sequences:
            return mask
        else:
            return None

    def step(self, x, states):
        raise NotImplementedError

    def get_constants(self, x):
        return []

    def get_initial_states(self, x):
        init_h = self.init_h.dimshuffle(('x', 0, 1))
        init_h = T.extra_ops.repeat(init_h, x.shape[0], axis=0)

        init_c = self.init_c.dimshuffle(('x', 0, 1))
        init_c = T.extra_ops.repeat(init_c, x.shape[0], axis=0)

        return [init_h, init_c]

    def preprocess_input(self, x):
        return x

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def get_config(self):
        config = {'return_sequences': self.return_sequences,
                  'go_backwards': self.go_backwards,
                  'stateful': self.stateful,
                  'unroll': self.unroll,
                  'consume_less': self.consume_less}
        if self.stateful:
            config['batch_input_shape'] = self.input_spec[0].shape
        else:
            config['input_dim'] = self.input_dim
            config['input_length'] = self.input_length

        base_config = super(ColRecurrent, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class PyramidSTM(ColRecurrent):
    def __init__(self, nb_filter, filter_length, direction='Down',
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 border_mode="same", sub_sample=(1, 1),
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.nb_filter = nb_filter
        self.filter_length = filter_length
        self.border_mode = border_mode
        self.subsample = sub_sample
        self.direction = direction

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        kwargs["nb_filter"] = nb_filter
        kwargs["filter_length"] = filter_length

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(PyramidSTM, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if self.direction == 'Down':
            X = K.permute_dimensions(x, (0, 2, 1, 3))
        elif self.direction == 'Right':
            X = K.permute_dimensions(x, (0, 3, 1, 2))
        else:
            raise Exception('ERROR: Unknown direction')

        if self.direction == 'Down':
            return K.permute_dimensions(super(PyramidSTM, self).call(X, mask), (0, 2, 1, 3))
        elif self.direction == 'Right':
            return K.permute_dimensions(super(PyramidSTM, self).call(X, mask), (0, 2, 3, 1))
        else:
            raise Exception('ERROR: Unknown direction')

    def get_output_shape_for(self, input_shape):
        if self.direction == 'Down':
            dim_out = 3
        elif self.direction == 'Right':
            dim_out = 2
        else:
            raise Exception('ERROR: Unknown direction')

        if self.return_sequences:
            return (input_shape[0], self.nb_filter, input_shape[2], input_shape[3])
        else:
            return (input_shape[0], self.nb_filter, input_shape[dim_out])

    def build(self, input_shape):
        # Input shape :: (samples, channels, height, width)

        self.input_spec = [InputSpec(shape=input_shape)]
        if self.direction == 'Down':
            dims = self.input_spec[0].shape
            self.shuffeled_dims = (dims[0], dims[2], dims[1], dims[3])
        elif self.direction == 'Right':
            dims = self.input_spec[0].shape
            self.shuffeled_dims = (dims[0], dims[3], dims[1], dims[2])
        else:
            raise Exception('ERROR: Unknown direction')

        input_dim = self.shuffeled_dims[2]
        self.input_dim = input_dim
        self.Shape = (4*self.nb_filter, input_dim, 1, 1)
        self.Shape1 = (4*self.nb_filter, self.nb_filter, 3, 1)
        self.Shape2 = (self.nb_filter, self.shuffeled_dims[3])

        self.W_iof = self.init(self.Shape)
        self.U_iof = self.init(self.Shape1)
        self.b_iof = K.zeros((4*self.nb_filter,))

        self.init_h = K.zeros(self.Shape2)
        self.init_c = K.zeros(self.Shape2)

        if self.stateful:
            self.reset_states()
        else:
            self.states = [None, None]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W_iof)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U_iof)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b_iof)
            self.regularizers.append(self.b_regularizer)

        self.trainable_weights = [self.W_iof, self.U_iof, self.b_iof,
                                  self.init_h, self.init_c]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'

    def conv_step(self, x, W, b=None, border_mode="valid", filter_shape=None, mask_type='b'):

        mask = np.ones(filter_shape, dtype=_FLOATX)

        in_third = self.input_dim//3
        out_third = self.nb_filter//3
        mask[:out_third,in_third:,0,0] = 0
        mask[out_third:2*out_third,2*in_third:,0,0] = 0

        W = W * mask

        input_shape = self.shuffeled_dims

        x = K.expand_dims(x, -1)  # add a dimension of the right

        conv_out = T.nnet.conv2d(x, W, subsample=self.subsample,
                                 border_mode='half',
                                 filter_flip=False,
                                 input_shape=(input_shape[0],
                                              input_shape[2],
                                              input_shape[3],
                                              1),
                                 filter_shape=filter_shape)
        if b:
            conv_out = conv_out + K.reshape(b, (1, filter_shape[0], 1, 1))

        conv_out = K.squeeze(conv_out, 3)  # remove the dummy 3rd dimension

        return conv_out

    def conv_step_hidden(self, x, W, border_mode="valid", filters=None, filter_shape=None):

        input_shape = self.shuffeled_dims
        if filters == None:
            filters = self.nb_filter

        x = K.expand_dims(x, -1)  # add a dimension of the right

        conv_out = T.nnet.conv2d(x, W, subsample=(1, 1),
                                 border_mode='half',
                                 filter_flip=False,
                                 input_shape=(input_shape[0],
                                              filters,
                                              input_shape[3],
                                              1),
                                 filter_shape=filter_shape)

        conv_out = K.squeeze(conv_out, 3)  # remove the dummy 3rd dimension

        return conv_out

    def step(self, x, states):

        h_tm1 = states[0]
        c_tm1 = states[1]

        input_to_state = self.conv_step(x, self.W_iof, self.b_iof, border_mode=self.border_mode, filter_shape=self.Shape)
        state_to_state = self.conv_step_hidden(h_tm1, self.U_iof, border_mode="same", filters=4*self.nb_filter, filter_shape=self.Shape1)
        gates = input_to_state + state_to_state

        o_f_i =self.inner_activation(gates[:,:3*self.nb_filter,:])
        o = o_f_i[:,0*self.nb_filter:1*self.nb_filter,:]
        f = o_f_i[:,1*self.nb_filter:2*self.nb_filter,:]
        i = o_f_i[:,2*self.nb_filter:3*self.nb_filter,:]
        g = self.activation(gates[:,3*self.nb_filter:4*self.nb_filter,:])

        c = (f * c_tm1) + (i * g)
        h = o * self.activation(c)

        return h, [h ,c]

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U}
        base_config = super(PyramidSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DiagLSTM(ColRecurrent):
    def __init__(self, nb_filter, filter_length, direction='Down',
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 border_mode="same", sub_sample=(1, 1),
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):

        self.nb_filter = nb_filter
        self.filter_length = filter_length
        self.border_mode = border_mode
        self.subsample = sub_sample
        self.direction = direction

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        kwargs["nb_filter"] = nb_filter
        kwargs["filter_length"] = filter_length

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(DiagLSTM, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if self.direction == 'Down':
            X = K.permute_dimensions(x, (0, 3, 1, 2))
        elif self.direction == 'Right':
            X = K.permute_dimensions(x, (0, 2, 1, 3))
        else:
            raise Exception('ERROR: Unknown direction')

        if self.stateful:
            super(DiagLSTM, self).call(X, mask)
        else:
            if self.reverse:
                X = X[:,::-1,:,:]
            X = Utils.Skew(X)
            res = super(DiagLSTM, self).call(X, mask)
            unskew = Utils.Unskew(res)

            if self.reverse:
                unskew = unskew[:,::-1,:,:]

            if self.direction == 'Down':
                return K.permute_dimensions(unskew, (0, 2, 3, 1))
            elif self.direction == 'Right':
                return K.permute_dimensions(unskew, (0, 2, 1, 3))
            else:
                raise Exception('ERROR: Unknown direction')

    def get_output_shape_for(self, input_shape):
        if self.direction == 'Down':
            dim_out = 2
        elif self.direction == 'Right':
            dim_out = 3
        else:
            raise Exception('ERROR: Unknown direction')

        if self.return_sequences:
            return (input_shape[0], self.nb_filter, input_shape[2], input_shape[3])
        else:
            return (input_shape[0], self.nb_filter, input_shape[dim_out])

    def build(self, input_shape):
        # Input shape :: (samples, channels, height, width)

        self.input_spec = [InputSpec(shape=input_shape)]
        if self.direction == 'Down':
            dims = self.input_spec[0].shape
            self.shuffeled_dims = (dims[0], dims[3], dims[1], dims[2])
        elif self.direction == 'Right':
            dims = self.input_spec[0].shape
            self.shuffeled_dims = (dims[0], dims[2], dims[1], dims[3])
        else:
            raise Exception('ERROR: Unknown direction')

        input_dim = self.shuffeled_dims[2]
        self.input_dim = input_dim
        self.Shape = (4*self.nb_filter, input_dim, 1, 1)
        self.Shape1 = (4*self.nb_filter, self.nb_filter, 2, 1)
        self.Shape2 = (self.nb_filter, self.shuffeled_dims[3])

        self.W_iof = self.init(self.Shape)
        self.U_iof = self.init(self.Shape1)
        self.b_iof = K.zeros((4*self.nb_filter,))

        self.init_h = K.zeros(self.Shape2)
        self.init_c = K.zeros(self.Shape2)

        if self.stateful:
            self.reset_states()
        else:
            self.states = [None, None]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W_iof)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U_iof)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b_iof)
            self.regularizers.append(self.b_regularizer)

        self.trainable_weights = [self.W_iof, self.U_iof, self.b_iof,
                                  self.init_h, self.init_c]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'

    def conv_step(self, x, W, b=None, border_mode="valid", filter_shape=None, mask_type='b'):

        mask = np.ones(filter_shape, dtype=_FLOATX)

        in_third = self.input_dim//3
        out_third = self.nb_filter//3
        mask[:out_third,in_third:,0,0] = 0
        mask[out_third:2*out_third,2*in_third:,0,0] = 0

        W = W * mask

        input_shape = self.shuffeled_dims

        x = K.expand_dims(x, -1)  # add a dimension of the right

        conv_out = T.nnet.conv2d(x, W, subsample=self.subsample,
                                 border_mode='half',
                                 filter_flip=False,
                                 input_shape=(input_shape[0],
                                              input_shape[2],
                                              input_shape[3],
                                              1),
                                 filter_shape=filter_shape)
        if b:
            conv_out = conv_out + K.reshape(b, (1, filter_shape[0], 1, 1))

        conv_out = K.squeeze(conv_out, 3)  # remove the dummy 3rd dimension

        return conv_out

    def conv_step_hidden(self, x, W, border_mode="valid", filters=None, filter_shape=None):

        input_shape = self.shuffeled_dims
        if filters == None:
            filters = self.nb_filter

        x = K.expand_dims(x, -1)  # add a dimension of the right

        conv_out = T.nnet.conv2d(x, W, subsample=(1, 1),
                                 border_mode='valid',
                                 filter_flip=False,
                                 input_shape=(input_shape[0],
                                              filters,
                                              input_shape[3],
                                              1),
                                 filter_shape=filter_shape)
        conv_out = K.squeeze(conv_out, 3)  # remove the dummy 3rd dimension

        return conv_out

    def step(self, x, states):

        h_tm1 = states[0]
        h_tm1 = T.concatenate([
            T.zeros((K.shape(h_tm1)[0], K.shape(h_tm1)[1], 1), theano.config.floatX),
            h_tm1
        ], axis=-1)

        c_tm1 = states[1]

        input_to_state = self.conv_step(x, self.W_iof, self.b_iof, border_mode=self.border_mode, filter_shape=self.Shape)
        state_to_state = self.conv_step_hidden(h_tm1, self.U_iof, border_mode="same", filters=4*self.nb_filter, filter_shape=self.Shape1)
        gates = input_to_state + state_to_state

        o_f_i =self.inner_activation(gates[:,:3*self.nb_filter,:])
        o = o_f_i[:,0*self.nb_filter:1*self.nb_filter,:]
        f = o_f_i[:,1*self.nb_filter:2*self.nb_filter,:]
        i = o_f_i[:,2*self.nb_filter:3*self.nb_filter,:]
        g = self.activation(gates[:,3*self.nb_filter:4*self.nb_filter,:])

        c = (f * c_tm1) + (i * g)
        h = o * self.activation(c)

        return h, [h ,c]

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U}
        base_config = super(DiagLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MaskedConvolution2D(Layer):
    def __init__(self, nb_filter, nb_row, nb_col, mask_type=None, direction='Down',
                 init='glorot_uniform', activation='linear', weights=None,
                 border_mode='valid', subsample=(1, 1), dim_ordering='th',
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.mask_type = mask_type
        self.direction = direction
        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for Convolution2D:', border_mode)
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.activation = activations.get(activation)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.input_spec = [InputSpec(ndim=4)]
        self.initial_weights = weights
        super(MaskedConvolution2D, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            stack_size = input_shape[1]
            self.input_dim = input_shape[1]
            self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            stack_size = input_shape[3]
            self.input_dim = input_shape[3]
            self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
        if self.bias:
            self.b = K.zeros((self.nb_filter,), name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.b]
        else:
            self.trainable_weights = [self.W]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        rows = K_conv.conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = K_conv.conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):

        mask = np.ones(self.W_shape, dtype=_FLOATX)

        center_col = self.nb_col//2
        center_row = self.nb_row//2

        if self.direction == 'Down':
            for i in range(self.nb_col):
                for j in range(self.nb_row):
                        if (j > center_row) or (j>i) or ((i - self.nb_col + j)>0):
                            mask[:, :, j, i] = 0
        elif self.direction == 'Right':
            for i in range(self.nb_col):
                for j in range(self.nb_row):
                        if (i > center_col) or (i>j) or ((j - self.nb_row + i)>0):
                            mask[:, :, j, i] = 0
        else:
            raise Exception('ERROR: Unknown direction')


        in_third = self.input_dim//3
        out_third = self.nb_filter//3
        if self.mask_type == 'a':
            mask[:out_third,:,0,0] = 0
            mask[out_third:2*out_third,in_third:,center_row,center_col] = 0
            mask[2*out_third:3*out_third,2*in_third:,center_row,center_col] = 0
        elif self.mask_type == 'b':
            mask[:out_third,in_third:,0,0] = 0
            mask[out_third:2*out_third,2*in_third:,center_row,center_col] = 0

        W = self.W * mask
        output = T.nnet.conv2d(x, W, subsample=self.subsample,
                               border_mode='half',
                               filter_flip=False,
                               filter_shape=self.W_shape)
        if self.bias:
            if self.dim_ordering == 'th':
                output += K.reshape(self.b, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                output += K.reshape(self.b, (1, 1, 1, self.nb_filter))
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        output = self.activation(output)
        return output

    def get_config(self):
        config = {'nb_filter': self.nb_filter,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias}
        base_config = super(MaskedConvolution2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class GetColors(Layer):
    def __init__(self, color, **kwargs):
        assert color in (0,1,2)
        self.color = color
        super(GetColors, self).__init__(**kwargs)

    def call(self, x, mask=None):
        X = K.permute_dimensions(x, (0, 2, 3, 1))
        return X[:,:,:,(self.color*256):(self.color+1)*256]

    def get_output_shape_for(self, input_shape):
        output = list(input_shape)
        return (output[0],output[2],output[3],256)

class SoftmaxLayer(Layer):
    def __init__(self, **kwargs):
        super(SoftmaxLayer, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1]*input_shape[2], input_shape[3])

    def call(self, x, mask=None):
        X = K.T.reshape(x,  (-1, K.shape(x)[-1]))
        y = K.softmax(X)
        y = K.T.reshape(y, (-1, K.shape(x)[1]*K.shape(x)[2] ,K.shape(x)[-1]))
        return y
