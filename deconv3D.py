import numpy as np
import warnings

from keras import activations, initializations, regularizers
from keras.engine import Layer, InputSpec
from keras.utils.np_utils import conv_output_length
from keras.layers.convolutional import Convolution3D

import backend_updated as K

class Deconvolution3D(Convolution3D):
    def __init__(self, nb_filter, kernel_dim1, kernel_dim2, kernel_dim3, output_shape,
                 init='he_normal', activation=None, weights=None,
                 border_mode='valid', subsample=(1, 1, 1),
                 dim_ordering='default',
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        if border_mode not in {'valid', 'same', 'full'}:
            raise Exception('Invalid border mode for Deconvolution3D:', border_mode)

        self.output_shape_ = output_shape
        super(Deconvolution3D, self).__init__(nb_filter, kernel_dim1, kernel_dim2, kernel_dim3,
                                              init=init, activation=activation,
                                              weights=weights, border_mode=border_mode,
                                              subsample=subsample, dim_ordering=dim_ordering,
                                              W_regularizer=W_regularizer, b_regularizer=b_regularizer,
                                              activity_regularizer=activity_regularizer,
                                              W_constraint=W_constraint, b_constraint=b_constraint,
                                              bias=bias, **kwargs)

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            conv_dim1 = self.output_shape_[2]
            conv_dim2 = self.output_shape_[3]
            conv_dim3 = self.output_shape_[4]
        elif self.dim_ordering == 'tf':
            conv_dim1 = self.output_shape_[1]
            conv_dim2 = self.output_shape_[2]
            conv_dim3 = self.output_shape_[3]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, conv_dim1, conv_dim2, conv_dim3)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], conv_dim1, conv_dim2, conv_dim3, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        
    	input_shape = self.input_spec[0].shape
        output = K.deconv3d(x, self.W, self.output_shape_,
                            strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering,
			    volume_shape=input_shape,
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
        config = {'output_shape': self.output_shape_}
        base_config = super(Deconvolution3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
