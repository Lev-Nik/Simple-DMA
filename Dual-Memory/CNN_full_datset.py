"""Running and training the CNN model on the entire data set"""

from keras.utils.np_utils import to_categorical
from cnn import Cnn
from tensorflow.keras import datasets

r"""splitting the dataset to test and train"""
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

r""" normalizing the data to categorical and to between 0 to 1 """
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
train_images, test_images = train_images / 255.0, test_images / 255.0

r""" getting the dimension of the input data """
input_dim = train_images.shape[1] * train_images.shape[2]
class_num = train_labels.shape[1]

r""" initializing the CNN model """
cnn = Cnn(input_dim, class_num)

r"""training the cnn"""
loss, acc, metrics = cnn.train(train_images, train_labels, test_images, test_labels, 50, 64)
print('the cnn accuracy on the full data set is: {}'.format(acc))
