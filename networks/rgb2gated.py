import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model,Sequential
from tensorflow_addons.layers import InstanceNormalization

def conv(filters, size, stride, activation, apply_instnorm=True):
    result = Sequential()
    result.add(Conv2D(filters, size, stride, padding='SAME', use_bias=True))
    if apply_instnorm:
        result.add(InstanceNormalization())
    if not activation == None:
        result.add(activation())
    return result

def conv_transp(filters, size, stride, activation, apply_instnorm=True):
    result = Sequential()
    result.add(Conv2DTranspose(filters, size, stride, padding='SAME', use_bias=True))
    if apply_instnorm:
        result.add(InstanceNormalization())
    if not activation == None:
        result.add(activation())
    return result

def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(np.minimum(shape[2],shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer

def DilatedLayer(inputs, nfd, dilation_rate):
    x = Conv2D(nfd, 3, padding = 'same', dilation_rate = (dilation_rate, dilation_rate), kernel_initializer = identity_initializer())(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x


def RGB2Gated(input_shape):
    # Encoder
    inputs  = Input(shape=input_shape)

    down0   = conv(32 , 3, 1, LeakyReLU, apply_instnorm=False)(inputs)
    down0   = conv(32 , 3, 1, LeakyReLU, apply_instnorm=False)(down0)

    down1   = tf.nn.max_pool(down0, ksize=2, strides=2, padding='SAME')
    down1   = conv(64 , 3, 1, LeakyReLU, apply_instnorm=False)(down1)
    down1   = conv(64 , 3, 1, LeakyReLU, apply_instnorm=False)(down1)

    down2   = tf.nn.max_pool(down1, ksize=2, strides=2, padding='SAME')
    down2   = conv(128, 3, 1, LeakyReLU, apply_instnorm=False)(down2)
    down2   = conv(128, 3, 1, LeakyReLU, apply_instnorm=False)(down2)

    # Bridge
    bridge  = tf.nn.max_pool(down2, ksize=2, strides=2, padding='SAME')
    sum_bridge = []
    bridge  = DilatedLayer(bridge, 256, 2)
    sum_bridge.append(bridge)
    bridge  = DilatedLayer(bridge, 256, 8)
    sum_bridge.append(bridge)
    bridge  = DilatedLayer(bridge, 256, 32)
    sum_bridge.append(bridge)
    bridge = add(sum_bridge)

    # Decoder
    up2     = conv_transp(128, 2, 2, ReLU, apply_instnorm=False)(bridge)
    up2     = Concatenate()([down2, up2])
    up2     = conv(128, 3, 1, ReLU, apply_instnorm=False)(up2)

    up1     = conv_transp(64, 2, 2, ReLU, apply_instnorm=False)(up2)
    up1     = Concatenate()([down1, up1])
    up1     = conv(64, 3, 1, ReLU, apply_instnorm=False)(up1)

    up0     = conv_transp(32, 2, 2, ReLU, apply_instnorm=False)(up1)
    up0     = Concatenate()([down0, up0])
    up0     = conv(32, 3, 1, ReLU, apply_instnorm=False)(up0)

    outputs = conv(1, 3, 1, ReLU, apply_instnorm=False)(up0) # Single channel output
    outputs = tf.concat([outputs, outputs, outputs], axis=3) # Replicate to 3 channels for visualization and losses

    model = Model(inputs, outputs, name = 'RGB2Gated')
    print(model.summary())
    return model

def downsample(filters, size, stride, apply_instnorm=True, use_bias=True, activation=True):
    result = Sequential()
    result.add(Conv2D(filters, size, stride, padding='same', use_bias=use_bias))
    if apply_instnorm:
        result.add(InstanceNormalization())
    if activation:
        result.add(LeakyReLU())
    return result

def Disc(input_shape):
    inp = Input(shape=input_shape, name='input_image')
    tar = Input(shape=input_shape, name='target_image')
    x = Concatenate()([inp, tar])                  # (bs, H, W, channels*2)

    down1 = downsample(64 , 4, 2, False, True , True )(x)        # (bs, H//2 , W//2, 64)
    down2 = downsample(128, 4, 2, True , False, True )(down1)    # (bs, H//4 , W//4, 128)
    down3 = downsample(256, 4, 2, True , False, True )(down2)    # (bs, H//8 , W//8, 256)
    down4 = downsample(512, 4, 1, True , False, True )(down3)    # (bs, H//8 , W//8, 256)
    down5 = downsample(512, 4, 1, True , False, True )(down4)    # (bs, H//8 , W//8, 256)
    down6 = downsample(1  , 4, 1, False, True , False)(down5)    # (bs, H//8 , W//8, 256)

    model = tf.keras.Model(inputs=[inp, tar], outputs=down6, name = 'RGB2Gated_Disc')
    print(model.summary())
    return model    