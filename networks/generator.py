import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model,Sequential
import tensorflow.keras.backend as K
from tensorflow_addons.layers import InstanceNormalization

def ReflectionPadding2D(x, padding=(1, 1)):
    """https://stackoverflow.com/questions/50677544/reflection-padding-conv2d"""
    if type(padding) == int:
        padding = (padding, padding)
    w_pad,h_pad = padding
    return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

def ResnetBlock(x,n_filters, dropout = False):
    input = x
    x = ReflectionPadding2D(x, padding=1)
    x = Conv2D(n_filters, kernel_size=3, strides=1, padding='valid')(x)
    x = Activation('relu')(x)

    x = ReflectionPadding2D(x, padding=1)
    x = Conv2D(n_filters, kernel_size=3, strides=1, padding='valid')(x)

    # Residual Connection
    x = Add()([input, x])
    return x

def translation_lowres(img, nf, n_downsampling=3, n_blocks = 8):
    """
    Translation Block (low-resolution steam)
    img: input image, channel-last
    n_downsampling: number of downsampling
    nf: number of filters after parallel extended encoder
    n_blocks: number of resnet blocks
    """
    x = AveragePooling2D(3, strides=2, padding='same')(img)

    x = ReflectionPadding2D(x, 3)
    x = Conv2D(int(nf/2), kernel_size=7, strides=1, padding='valid')(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    x_local = x

    x = ReflectionPadding2D(x_local, 4)
    x = Conv2D(int(nf/2), kernel_size=5, strides=1, dilation_rate = 2, padding='valid')(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    x = ReflectionPadding2D(x, 4)
    x = Conv2D(int(nf/2), kernel_size=5, strides=1, dilation_rate = 2, padding='valid')(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    x = ReflectionPadding2D(x, 4)
    x = Conv2D(int(nf/2), kernel_size=5, strides=1, dilation_rate = 2, padding='valid')(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    x_global = x

    x = ReflectionPadding2D(x_local, 1)
    x = Conv2D(int(nf/2), kernel_size=3, strides=1, dilation_rate = 1, padding='valid')(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    x = ReflectionPadding2D(x, 1)
    x = Conv2D(int(nf/2), kernel_size=3, strides=1, dilation_rate = 1, padding='valid')(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    x = ReflectionPadding2D(x, 1)
    x = Conv2D(int(nf/2), kernel_size=3, strides=1, dilation_rate = 1, padding='valid')(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    x_local = x

    x = tf.concat([x_global,x_local], axis = -1)
    x = ReflectionPadding2D(x, 1)
    x = Conv2D(nf, kernel_size=3, strides=1, dilation_rate = 1, padding='valid')(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)

    # downsample
    for i in range(n_downsampling):
        mult = 2**i
        x = Conv2D(nf * mult * 2, kernel_size=3, strides=2, padding='same')(x)
        x = Activation('relu')(x)

    for i in range(n_blocks):
        x = ResnetBlock(x,nf * mult * 2)

    # upsample
    for i in range(n_downsampling):
        mult = 2**(n_downsampling - i)
        x = Conv2DTranspose(filters=int(nf * mult / 2), kernel_size=3, strides=2, padding='same')(x)
        x = Activation('relu')(x)

    return x

def Translation(input_shape, n_downsampling=4, nf=32, n_blocks_hr=3, n_blocks_lr = 6):
    """
    Translation Block 
    input_shape: input shape in channel-last format
    n_downsampling: number of total downsampling (1 in high-resolution steam, the rest in low-resolution steam)
    nf: number of filters after parallel extended encoder
    n_blocks_hr: number of resnet blocks in the high-resolution steam
    n_blocks_lr: number of resnet blocks in the low-resolution steam
    """

    inputs = Input(shape=input_shape)

    out_lowres = translation_lowres(inputs, nf*2, n_downsampling=n_downsampling-1, n_blocks = n_blocks_lr)

    x = inputs
    x = ReflectionPadding2D(x, 3)
    x = Conv2D(int(nf/2), kernel_size=7, strides=1, padding='valid')(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    x_local = x

    x = ReflectionPadding2D(x_local, 4)
    x = Conv2D(int(nf/2), kernel_size=5, strides=1, dilation_rate = 2, padding='valid')(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    x = ReflectionPadding2D(x, 4)
    x = Conv2D(int(nf/2), kernel_size=5, strides=1, dilation_rate = 2, padding='valid')(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    x = ReflectionPadding2D(x, 4)
    x = Conv2D(int(nf/2), kernel_size=5, strides=1, dilation_rate = 2, padding='valid')(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    x_global = x

    x = ReflectionPadding2D(x_local, 1)
    x = Conv2D(int(nf/2), kernel_size=3, strides=1, dilation_rate = 1, padding='valid')(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    x = ReflectionPadding2D(x, 1)
    x = Conv2D(int(nf/2), kernel_size=3, strides=1, dilation_rate = 1, padding='valid')(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    x = ReflectionPadding2D(x, 1)
    x = Conv2D(int(nf/2), kernel_size=3, strides=1, dilation_rate = 1, padding='valid')(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)
    x_local = x

    x = tf.concat([x_global,x_local], axis = -1)
    x = ReflectionPadding2D(x, 1)
    x = Conv2D(nf, kernel_size=3, strides=1, dilation_rate = 1, padding='valid')(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)

    # downsample
    x = Conv2D(nf*2, kernel_size=3, strides=2, padding='same')(x)
    x = Activation('relu')(x)
    x = Add()([out_lowres, x])

    # resnet blocks
    for i in range(n_blocks_hr):
        x = ResnetBlock(x,nf * 2)

    # upsample
    x = Conv2DTranspose(filters=nf, kernel_size=3, strides=2, padding='same')(x)
    x = Activation('relu')(x)

    # final convolution
    x = ReflectionPadding2D(x, 3)
    x = Conv2D(3, kernel_size=7, strides=1, padding='valid')(x)
    x = Activation('relu')(x)
    x = tf.clip_by_value(x,0,1)

    # create model graph
    model = Model(inputs, x, name = 'TranslationBlock')
    print(model.summary())
    return model


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
    if not activation == None:
        result.add(activation())
    return result

def Consistency(input_shape, nf = 32):
    """
    Consistency Block
    input_shape: input shape in channel-last format
    nf: base number of filters
    """
    inputs  = Input(shape=input_shape)

    down0   = conv(nf , 3, 1, LeakyReLU, apply_instnorm=False)(inputs)
    down0   = conv(nf , 3, 1, LeakyReLU, apply_instnorm=False)(down0)

    down1   = conv(nf*2 , 3, 2, LeakyReLU, apply_instnorm=False)(down0)
    down1   = conv(nf*2 , 3, 1, LeakyReLU, apply_instnorm=False)(down1)
    down1   = conv(nf*2 , 3, 1, LeakyReLU, apply_instnorm=False)(down1)

    down2   = conv(nf*4, 3, 2, LeakyReLU, apply_instnorm=False)(down1)
    down2   = conv(nf*4, 3, 1, LeakyReLU, apply_instnorm=False)(down2)
    down2   = conv(nf*4, 3, 1, LeakyReLU, apply_instnorm=False)(down2)

    down3   = conv(nf*8, 3, 2, LeakyReLU, apply_instnorm=False)(down2)
    down3   = conv(nf*8, 3, 1, LeakyReLU, apply_instnorm=False)(down3)
    down3   = conv(nf*8, 3, 1, LeakyReLU, apply_instnorm=False)(down3)

    # Bridge
    bridge  = conv(nf*16, 3, 2, LeakyReLU, apply_instnorm=False)(down3)
    bridge  = conv(nf*16, 3, 1, LeakyReLU, apply_instnorm=False)(bridge)
    bridge  = conv(nf*16, 3, 1, LeakyReLU, apply_instnorm=False)(bridge)

    # Decoder
    up3     = conv_transp(nf*8, 2, 2, ReLU, apply_instnorm=False)(bridge)
    up3     = Concatenate()([down3, up3])
    up3     = conv(nf*8, 3, 1, LeakyReLU, apply_instnorm=False)(up3)

    up2     = conv_transp(nf*4, 2, 2, ReLU, apply_instnorm=False)(up3)
    up2     = Concatenate()([down2, up2])
    up2     = conv(nf*4, 3, 1, ReLU, apply_instnorm=False)(up2)

    up1     = conv_transp(nf*2, 2, 2, ReLU, apply_instnorm=False)(up2)
    up1     = Concatenate()([down1, up1])
    up1     = conv(nf*2, 3, 1, ReLU, apply_instnorm=False)(up1)

    up0     = conv_transp(nf, 2, 2, ReLU, apply_instnorm=False)(up1)
    up0     = Concatenate()([down0, up0])
    up0     = conv(nf, 3, 1, ReLU, apply_instnorm=False)(up0)
    
    outputs = conv(3, 4, 1, ReLU, apply_instnorm=False)(up0)
    outputs = tf.clip_by_value(outputs,0,1)

    model = Model(inputs, outputs, name = 'ConsistencyBlock')
    print(model.summary())
    return model

def main():
    img_shape = [768,1280,3] # define image shape
    translation = Translation(img_shape) # construct Translation block
    consistency = Consistency(img_shape) # construct Consistency block 

    # img_in -> translation block -> consistency block -> img_out
    img_in = tf.random.uniform([1] + img_shape) 
    img_trans = translation(img_in)
    img_out = consistency(img_trans) 

if __name__ == '__main__':
    main()