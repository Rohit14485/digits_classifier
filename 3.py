import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)
# model=tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# model.add(tf.keras.layers.Dense(units=130,activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(units=130,activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# model.fit(x_train,y_train, epochs=5)
# accuracy,loss=model.evaluate(x_test,y_test)
# model.save('digits.model')
model=models.load_model('digits.model')
img=cv.imread('b.png')[:,:,0]
img=np.invert(np.array([img]))
prediction=model.predict(img)
print(np.argmax(prediction))

