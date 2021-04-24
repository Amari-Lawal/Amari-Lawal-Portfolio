import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model

# Model
inputs = tf.keras.Input(shape=(28,28,1),name="Img")
x = tf.keras.layers.Conv2D(32,kernel_size=(3,3),input_shape=(28,28,1),activation='relu')(inputs)
x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
x = tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.1)(x)
outputs = tf.keras.layers.Dense(10,activation='softmax')(x)

model = tf.keras.Model(inputs,outputs,name="model")

#model.summary()

