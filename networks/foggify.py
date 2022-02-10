import numpy as np
import tensorflow as tf
from utils import *

def Foggify(img_l, img_ref, depth):
    """
    img_l, img_r: tf.tensor, 0 ~ 1
    """
    beta_class = [0.003, 0.005, 0.007, 0.01]
    beta_idx = np.random.choice(len(beta_class))
    beta = beta_class[beta_idx]
    t = tf.math.exp(-1*beta*depth)
    img_fog = img_l * t
    img_fog = img_fog + (tf.reduce_mean(img_ref)-tf.reduce_mean(img_fog))/tf.reduce_mean(1-t) * (1-t)
    return img_fog

