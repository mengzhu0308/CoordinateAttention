#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/10/13 19:14
@File:          CoordinateAttention.py
'''

from keras import backend as K
from keras.layers import Layer, Conv2D, BatchNormalization

class CoordinateAttention(Layer):
    def __init__(self, reduction=4, **kwargs):
        super(CoordinateAttention, self).__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        super(CoordinateAttention, self).build(input_shape)
        inter_channels = input_shape[-1] // self.reduction
        self.conv = Conv2D(inter_channels, 1, use_bias=False)
        self.bn = BatchNormalization()
        self.conv_h = Conv2D(input_shape[-1], 1)
        self.conv_w = Conv2D(input_shape[-1], 1)

    def call(self, inputs, **kwargs):
        h = K.int_shape(inputs)[1]
        x_h = K.mean(inputs, axis=2, keepdims=True)
        x_w = K.mean(inputs, axis=1, keepdims=True)
        x_t = K.permute_dimensions(x_w, (0, 2, 1, 3))

        x = K.concatenate([x_h, x_t], axis=1)
        x = K.relu(self.bn(self.conv(x)))

        x_h, x_t = x[:, :h, :, :], x[:, h:, :, :]
        x_w = K.permute_dimensions(x_t, (0, 2, 1, 3))
        x_h = K.sigmoid(self.conv_h(x_h))
        x_w = K.sigmoid(self.conv_w(x_w))

        return inputs * x_h * x_w

    def get_config(self):
        config = {'reduction': self.reduction}
        base_config = super(CoordinateAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))