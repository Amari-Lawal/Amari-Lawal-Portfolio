import tensorflow as tf
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from model import MyModel
from tensorflow.keras.utils import plot_model

DATA_URL_TE = "C:/Users/user1/Desktop/Image Processing/digit-recognizer/test.csv"

df_test = pd.read_csv(DATA_URL_TE)

# Assigns the Training, test set and the labels to compare with
X_test = df_test.values.reshape(28000,28,28,1)

# Normalise the data
X_test = X_test / 255

# Loads trained  and plots architecture
# Loading doesn't work!
#model = tf.keras.models.load_model("C:/Users/user1/Desktop/Image Processing/digit-recognizer/Saved_Model/model/",custom_objects={"model":model})
model = MyModel()
model = model.load_model()
model.load_weights("C:/Users/user1/Desktop/Image Processing/digit-recognizer/ModelCheckpoints/checkpoint")

print(f'Loading model...')
plot_model(model, to_file='C:/Users/user1/Desktop/Image Processing/digit-recognizer/Model Architecture/Model_Architecture.png')
print(f'Model Architecture Image Loaded!')

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

# Predicted Lable Numbers console
randm_num = np.random.randint(0,28000)
print(f'Predicted Label numbers: {class_labels.iloc[randm_num,:][0]}')

# Original Lable Numbers Plot
fig2 = plt.figure(figsize=(10,5))
plt.title(class_labels.iloc[randm_num,:][0])
plt.imshow(X_test.iloc[randm_num,:].values.reshape(28,28))
plt.axis("off")
#fig2.savefig('/tf/digit-recognizer/Predictions/Original_Label_Number.jpg', bbox_inches='tight', dpi=150)
fig2.savefig('C:/Users/user1/Desktop/Image Processing/digit-recognizer/Predictions/Original_Label_Number.jpg', bbox_inches='tight', dpi=150)
plt.show()

