from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def PatchGAN_Discriminator(inp, ndf=64, n_layers=3, use_sigmoid = False, max_nf = 512):

    kw = 4
    
    x = Conv2D(ndf, kw, strides=2, padding='same')(inp)
    x = LeakyReLU(0.2)(x)
    
    nf = ndf
    for n in range(1, n_layers):
        nf = min(nf*2, max_nf)
        x = Conv2D(nf, kw, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
    
    nf = min(nf*2, max_nf)
    x = Conv2D(nf, kw, strides=1)(x)
    x = ZeroPadding2D()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(1, kw, strides=1)(x)
    x = ZeroPadding2D()(x)
    
    if use_sigmoid:
        x = Activation('sigmoid')(x)
    return x


def Discriminator(input_shape, ndf=32, n_layers=4, use_sigmoid = False, num_D = 2):

    inp = Input(shape=input_shape, name='input_image')

    result = []
    input_downsampled = inp
    for i in range(num_D):
        in_shape = (int(input_shape[0]/(2**i)), int(input_shape[1]/(2**i)),input_shape[2])
        result.append(PatchGAN_Discriminator(input_downsampled, ndf = ndf, n_layers = n_layers, use_sigmoid = use_sigmoid))
        input_downsampled = AveragePooling2D(3,strides=2, padding = 'same')(input_downsampled)

    return Model(inp, result, name = "Discriminator")