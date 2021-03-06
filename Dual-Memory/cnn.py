"""CNN implementation"""

import copy
import logging
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, LeakyReLU
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DNN")


class Cnn:
    def __init__(self, input_size, class_num):
        r""" initializer for the CNN

             Args:
                 input_size: the size of the input data.
                 class_num: the number of classes that are needed to be classified to.
            """

        self.class_num = class_num
        self.input_size = input_size

        # creating the CNN model
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                              input_shape=(32, 32, 3)))
        self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.3))
        self.model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.4))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))

        # the hyper parameters where chosen based on Dual-Memory Deep Learning Architectures for
        # Lifelong Learning of Everyday Human Behaviors paper.
        self.model.compile(optimizer=SGD(learning_rate=0.025, momentum=0.9, decay=5e-4),
                           loss=categorical_crossentropy,
                           metrics=["accuracy"])

    def train(self, samples, labels, test_samples, test_labels,
              epoch=1, batch_size=1, cm=False):
        r""" training function for the training of the CNN

             Args:
                 samples: the input samples.
                 labels: the labels of all the input samples.
                 test_samples: the samples we are going to test the CNN on.
                 test_labels: the labels we are going to test the CNN on.
                 epoch: the number of epoch we want for our CNN, default value is 1.
                 batch_size: the batch size we want for each run, default value is 1.
                 cm: if we want to lot a confusion matrix, default value is False.
            """

        matrices = []
        history = TrainHistory(self.model, epoch)
        print(samples.shape)
        print(labels.shape)
        print(test_samples.shape)
        print(test_labels.shape)
        self.model.fit(samples, labels, batch_size=batch_size, epochs=epoch,
                       verbose='auto', callbacks=[history],
                       validation_data=(test_samples, test_labels))

        if cm:
            cm_model = copy.copy(self.model)
            cm_labels = np.argmax(test_labels, axis=1)
            for weight in history.weights_list:
                confusion_matrix = np.zeros([self.class_num, self.class_num]).astype("int32")
                cm_model.set_weights(weight)
                predicts = np.argmax(
                    cm_model.predict(test_samples), axis=1
                ).astype("int32")
                for p in range(predicts.shape[0]):
                    confusion_matrix[predicts[p], cm_labels[p]] += 1
                matrices.append(confusion_matrix)
        return history.loss, history.acc, matrices

    def evaluate(self, xt, yt, batch_size=1, verbose=0):
        r""" evaluating the CNN

             Args:
                 xt: the input samples.
                 yt: the wanted output labels.
                 batch_size: the batch size we want for each run, default value is 1.
                 verbose: if we want to see the training process on the output window, default value is 0
                 meaning we want to see it.
            """
        history = EvaluationHistory()
        self.model.evaluate(xt, yt, verbose=0, batch_size=batch_size, callbacks=[history])
        if verbose == 1:
            logger.info(f"\rTest Accuracy={np.mean(np.array(history.acc))}")
            logger.info(f"\rTest Loss={np.mean(np.array(history.loss))}")
        return history.loss, history.acc

    def get_last_layer_output(self, x):
        r""" getting the features of the CNNs last layer

             Args:
                 x: input data
            """
        layer_output = self.model.predict(x)
        return layer_output


class TrainHistory(Callback):
    r""" class for retrieving the training history of the model
        """
    def __init__(self, model, epochs):
        super().__init__()
        self.model_list = model
        self.weights_list = []
        self.loss = 0.0
        self.acc = 0.0
        self.epochs = epochs

    def on_train_begin(self, logs=None):
        self.weights_list = []
        self.loss = 0.0
        self.acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 == self.epochs:
            self.weights_list.append(self.model.get_weights())
        self.loss = logs.get('loss')
        self.acc = logs.get('accuracy')


class EvaluationHistory(Callback):
    r""" class for retrieving the evaluation history of the model
            """
    def __init__(self):
        super().__init__()
        self.loss = []
        self.acc = []

    def on_test_begin(self, logs=None):
        self.loss = []
        self.acc = []

    def on_test_batch_end(self, batch, logs=None):
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))

