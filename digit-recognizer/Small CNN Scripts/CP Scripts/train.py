import os 
import sys
import pickle 
import tensorflow as tf
import pandas as pd
import numpy as np
import glob 
import matplotlib.pyplot as plt
import matplotlib.image as img
from tensorflow import keras
from tensorflow.keras.layers import *
import sklearn
from model import MyModel 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

#%matplotlib inline

# Reads the data set files
#DATA_URL = "/tf/digit-recognizer/train.csv"
DATA_URL = "C:/Users/user1/Desktop/Image Processing/digit-recognizer/train.csv"
df = pd.read_csv(DATA_URL)    

# Drops Labels from dataset
labels = pd.DataFrame(df["label"])
df = df.drop("label",axis=1)

# Assigns the Training, test set and the labels to compare with
X_train = df.values.reshape(42000,28,28,1)
y_train = labels.values.reshape(42000,1)

# Normalise the data
X_train = X_train / 255

# Shuffles data to avoid systematic errors
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train, random_state = 25)

# Call in class from model.py
model = MyModel()

# Compiles model
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(X_train,y_train,validation_split=0.2,epochs=10,batch_size=64,verbose=1)

def model_Performance(history):
    fig = plt.figure(figsize=(10,5))

    plt.subplot(221)
    plt.plot(history.history['accuracy'],'bo--', label = "acc")
    plt.plot(history.history['val_accuracy'], 'ro--', label = "val_acc")
    plt.title("train_acc vs val_acc")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    plt.subplot(222)
    plt.plot(history.history['loss'],'bo--', label = "loss")
    plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")

    plt.legend()

    #fig.savefig('/tf/digit-recognizer/Model_Performance/Model_performance.jpg', bbox_inches='tight', dpi=150)
    fig.savefig("C:/Users/user1/Desktop/Image Processing/digit-recognizer/Model_Performance/Model_performance.jpg", bbox_inches='tight', dpi=150)
    plt.show()
model_Performance(history)

# Saves Model and its weights
model.save_weights('C:/Users/user1/Desktop/Image Processing/ModelCheckpoints/Digit_Recoginser.ckpt')
model.save('C:/Users/user1/Desktop/Image Processing/digit-recognizer/Saved_Model/model')
#model.save('C:/Users/user1/Desktop/Image Processing/digit-recognizer/Saved_Model/Digit_Recognizer.h5')
