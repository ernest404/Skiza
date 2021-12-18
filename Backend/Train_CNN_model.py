import pickle
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np


#load data for training.
# train data
images_train = np.array(pickle.load(open("Backend/pickled_data/images_train", "rb")))
labels_train = []
labels_train.append(pickle.load(open("Backend/pickled_data/labels_train", "rb")))
# test data
images_test = np.array(pickle.load(open("Backend/pickled_data/images_test", "rb")))
labels_test = []
labels_test.append(pickle.load(open("Backend/pickled_data/labels_test", "rb")))

# classes = len(np.unique(labels_train))

from keras.utils import np_utils

labels_train = np_utils.to_categorical(labels_train, 26)
labels_test = np_utils.to_categorical(labels_test, 26)

#normalize data
images_train = images_train/255
images_test = images_test/255

#ConV2D only accepts images with a minimum of 4 dimensions, we have to add channel to grayscale images to make it 4d
images_train = np.reshape(images_train, (images_train.shape[0], 50, 50, 1))
images_test = np.reshape(images_test, (images_test.shape[0], 50, 50,1))

# classes

# len(np.unique(labels_train))

# Create model
model = Sequential()
model.add(Conv2D(16, (2,2), input_shape= images_train.shape[1:], activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (3,3), padding = 'same'))
model.add(Conv2D(64, (5,5), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (5,5), padding = 'same'))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(26, activation='softmax'))
# Specify optimizer and loss function
sgd = optimizers.SGD(lr=1e-2)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])


model.fit(images_train, labels_train, validation_split=0.2,shuffle=True, epochs=10, batch_size=200)

scores = model.evaluate(images_test, labels_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
print("CNN Score: %.2f%%" % (scores[1]*100))

model.save('Backend/model/cnn_model_keras2.h5')