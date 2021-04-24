import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel,self).__init__()
        # Makes CNN Model
        self.conv1 = tf.keras.layers.Conv2D(32,kernel_size=(3,3),input_shape=(28,28,1),activation='relu')
        self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.conv2 = tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.dense = tf.keras.layers.Dense(10,activation='softmax')
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.dense(x)

"""
class AuxNet(tf.keras.layers.Layer):
    def __init__(self, ratio=8):
        super(AuxNet, self).__init__()
        self.ratio = ratio
        self.avg = tf.keras.layers.GlobalAveragePooling2D()
        self.max = tf.keras.layers.GlobalMaxPooling2D()

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(input_shape[-1], 
                               kernel_size=1, strides=1, padding='same',
                               use_bias=True, activation=tf.nn.relu)
        self.conv2 = tf.keras.layers.Conv2D(input_shape[-1], 
                                               kernel_size=1, strides=1, padding='same',
                                               use_bias=True, activation=tf.nn.relu)
        
        super(AuxNet, self).build(input_shape)
        
    def call(self, inputs):
        avg = self.avg(inputs)
        max = self.max(inputs)
        avg = tf.keras.layers.Reshape((1, 1, avg.shape[1]))(avg)  
        max = tf.keras.layers.Reshape((1, 1, max.shape[1]))(max)   
        
        conv1a = self.conv1(avg)
        conv1b = self.conv2(max)
        
        return tf.nn.sigmoid(conv1a + conv1b)rrr
"""

