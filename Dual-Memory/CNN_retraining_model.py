"""Running and training the CNN on 10 split dataset"""

from keras.utils.np_utils import to_categorical
from cnn import Cnn
from tensorflow.keras import datasets
import numpy as np
import Plotter
import json

r"""splitting the dataset to test and train"""
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

r"""setting the classes in each split dataset."""
selections = [[0, 1],
              [2, 3],
              [4, 5],
              [6, 7],
              [8, 9],
              [0, 4],
              [5, 8],
              [9, 1],
              [2, 6],
              [3, 7]]

r""" normalizing the data to categorical and to between 0 to 1 """
train_labels_full = to_categorical(train_labels)
test_labels_full = to_categorical(test_labels)
test_images_full = test_images / 255.0

r""" getting the dimension of the input data """
input_dim = train_images.shape[1] * train_images.shape[2]
class_num = train_labels_full.shape[1]

r""" initializing the CNN model """
cnn = Cnn(input_dim, class_num)

scores = []
img_test = []
lbl_test = []
num_of_classes = []
for i in range(10):
    r"""splitting the data according to the selections."""
    subset_img_train = train_images[np.isin(train_labels, selections[i]).flatten()]
    subset_label_train = train_labels[np.isin(train_labels, selections[i]).flatten()]
    subset_img_test = test_images[np.isin(test_labels, selections[i]).flatten()]
    subset_label_test = test_labels[np.isin(test_labels, selections[i]).flatten()]

    r"""converting the labels to categorical"""
    subset_label_train = to_categorical(subset_label_train, num_classes=10)
    subset_label_test = to_categorical(subset_label_test, num_classes=10)
    # Normalize pixel values to be between 0 and 1
    subset_img_train, subset_img_test = subset_img_train / 255.0, subset_img_test / 255.0

    print(input_dim, class_num)

    new_labels = np.unique(np.argmax(subset_label_train, axis=1))

    for j in new_labels:
        if j not in num_of_classes:
            num_of_classes.append(j)

    r"""training the cnn"""
    loss, acc, metrics = cnn.train(subset_img_train, subset_label_train, subset_img_test, subset_label_test, 10, 64)

    if len(num_of_classes) < 10:
        score = acc * (len(num_of_classes)/10)
    else:
        score = acc
    scores.append(score)

print('the cnn scores are: {}'.format(scores))

r""" saving the retraining CNN results """
json.dump(scores, open("retrainingCNN.json", 'w'))

r""" plotting the retraining CNN results """
Plotter.plot_graph(scores, "results of retraining of CNN", 'ro--')
