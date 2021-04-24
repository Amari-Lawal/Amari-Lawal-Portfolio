import tensorflow as tf
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
# pip install pydot
# pip install pydotplus
# apt-get install graphiz
#"/tf/digit-recognizer/test.csv"
DATA_URL_TEST = pd.read_csv("/tf/digit-recognizer/test.csv")
X_test = DATA_URL_TEST
X_test = X_test.values.reshape(28000,28,28,1)
X_test = X_test / 255

new_model = tf.keras.models.load_model('/tf/digit-recognizer/Saved_Model/Digit_Recoginser_Model.h5')
print(f'Loading model...')
plot_model(new_model, to_file='/tf/digit-recognizer/Model Architecture/Model_Architecture.png')
print(f'Model Architecture Image Loaded!')
yhat_test = new_model.predict(X_test)
class_labels = pd.DataFrame(np.argmax(yhat_test, axis=1))

# Evaluates the models Performance with the Test set
print(f'Evaluating Model...')
score = new_model.evaluate(X_test,class_labels)
print('Test accuracy Digit_Recoginser%:', (score[1]*100))

# Turns into dataframe 
X_test = X_test.reshape(28000,784)
X_test = pd.DataFrame(X_test)

# Predicted Lable Numbers
randm_num = np.random.randint(0,28000)
print(f'Predicted Label numbers:{class_labels.iloc[randm_num,:][0]}')

# Original Lable Numbers
fig2 = plt.figure(figsize=(10,5))
plt.imshow(X_test.iloc[randm_num,:].values.reshape(28,28))
plt.title(f'Predicted Label numbers:{class_labels.iloc[randm_num,:][0]}')
plt.axis("off")
fig2.savefig('/tf/digit-recognizer/Predictions/Original_Label_Number.jpg', bbox_inches='tight', dpi=150)
plt.show()

    
    