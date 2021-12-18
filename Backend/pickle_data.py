import cv2
from glob import glob
import numpy as np
from sklearn.utils import shuffle
import pickle
import os

images_labels = [] #initialize an empty list to store images and their class names once read into our script
paths = glob("Backend/signs/*/*.jpg")#Return a list of paths matching the pathname pattern.

for path in paths:#iterate through list of paths getting the image and class they belong to.
    label = path[path.find(os.sep)+1: path.rfind(os.sep)]
    img = cv2.imread(path, 0)
    images_labels.append((np.array(img, dtype=np.uint8), label))#store the image together with it's class name as tuple
                         
images_labels = shuffle(shuffle(shuffle(shuffle(images_labels))))#Shuffling data before splitting
images, labels = zip(*images_labels)#seperate all images and labels

# Train test split
from sklearn.model_selection import train_test_split
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.20)

#Since it takes alot of time to read the images into other scripts, store data as streams of binary in files for future use once the process complets
#The data can easily be recreated in other scripts for use.
img_train_file = open('Backend/pickled_data/images_train', 'wb')
pickle.dump(images_train, img_train_file)
img_train_file.close()
del images_train

img_test_file = open('Backend/pickled_data/images_test', 'wb')
pickle.dump(images_test, img_test_file)
img_test_file.close()
del images_test

labels_train_file = open('Backend/pickled_data/labels_train', 'wb')
pickle.dump(labels_train, labels_train_file)
labels_train_file.close()
del labels_train

labels_test_file = open('Backend/pickled_data/labels_test', 'wb')
pickle.dump(labels_test, labels_test_file)
labels_test_file.close()
del labels_test

print('Data successfully pickled')