import tensorflow as tf
## input data
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)


import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import time
start = time.time()
from sklearn.metrics import classification_report

train = pd.read_csv("/Users/jing/Desktop/235final project/train.csv")
test = pd.read_csv("test.csv")

x_train = np.array(train.iloc[:, 1:])
y_train = np.array(train.iloc[:, 0])
x_test = np.array(test)
input_shape = (28, 28, 1)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)/255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)/255

X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train)

# Construct the convolutional neural network model
cnn = Sequential()
cnn.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten()) # Flattening the 2D arrays for fully connected layers
cnn.add(Dense(128, activation=tf.nn.relu))
cnn.add(Dropout(0.2))
cnn.add(Dense(10, activation=tf.nn.softmax))
cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# Apply the early stopping to the model
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
history = cnn.fit(x=x_train, y=y_train, epochs=20, callbacks=[early_stopping], validation_data=(X_valid, Y_valid))

predicted = cnn.predict(x_test)
predicted = np.argmax(predicted, axis = 1)
predicted = pd.Series(predicted, name="Label")
# submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predicted],axis = 1)
#
# submission.to_csv("cnn_mnist_datagen.csv",index=False)

plot_model(cnn, show_shapes=True, show_layer_names=True, to_file='model.png')
# Plot the error rate for training, validation and test for each epochs
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

print(classification_report(Y_valid, np.argmax(cnn.predict(X_valid), axis = 1)))

end = time.time()
print(end - start)
