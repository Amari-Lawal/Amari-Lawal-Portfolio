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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
#%matplotlib inline

# Reads the data set files
DATA_URL = "/tf/digit-recognizer/train.csv"
DATA_URL_TE = "/tf/digit-recognizer/test.csv"

df = pd.read_csv(DATA_URL)    
df_test = pd.read_csv(DATA_URL_TE)

# Drops Labels from dataset
labels = pd.DataFrame(df["label"])
df = df.drop("label",axis=1)

# Assigns the Training, test set and the labels to compare with
X_train = df.values.reshape(42000,28,28,1)
X_test = df_test.values.reshape(28000,28,28,1)
y_train = labels.values.reshape(42000,1)

# Normalise the data
X_train = X_train / 255
X_test = X_test / 255

# Shuffles data to avoid systematic errors
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train, random_state = 25)

# Makes CNN Model
model = keras.Sequential()
model.add(Conv2D(32,kernel_size=(3,3),input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(10,activation='softmax'))

# Compiles model
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Fits the model
history = model.fit(X_train,y_train,validation_split=0.2,epochs=10,batch_size=64,verbose=1)

# Plots Graph for Models accuracy and loss Performance then saves
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

fig.savefig('/tf/digit-recognizer/Model_Performance/Model_performance.jpg', bbox_inches='tight', dpi=150)
plt.show()

# Gets the probility predictions for test then gets the corresponding class label number 
yhat_test = model.predict(X_test)
class_labels = pd.DataFrame(np.argmax(yhat_test, axis=1))

# Evaluates the models Performance with the Test set
print(f'Evaluating Model...')
score4 = model.evaluate(X_test,class_labels)
print('Test accuracy Digit_Recoginser%:', (score4[1]*100))

# Turns into dataframe 
X_test = X_test.reshape(28000,784)
X_test = pd.DataFrame(X_test)

# Predicted Lable Numbers
randm_num = np.random.randint(0,28000)
fil = open('/tf/digit-recognizer/Predictions/Prediction_label_numbers.txt','+w',newline='\n')
fil.write(f'Predicted Label numbers:{class_labels.iloc[randm_num,:][0]}')
fil.close()
print(f'Predicted Label numbers: {class_labels.iloc[randm_num,:][0]}')

# Original Lable Numbers
fig2 = plt.figure(figsize=(10,5))
plt.imshow(X_test.iloc[randm_num,:].values.reshape(28,28))
plt.axis("off")
fig2.savefig('/tf/digit-recognizer/Predictions/Original_Label_Number.jpg', bbox_inches='tight', dpi=150)
plt.show()

# Saves Model and its weights
model.save_weights('/tf/digit-recognizer/ModelCheckpoints/Digit_Recoginser.ckpt')
model.save('/tf/digit-recognizer/Saved_Model/Digit_Recoginser_Model.h5')