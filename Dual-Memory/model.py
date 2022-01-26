"""Dual Memory Architecture implementation"""

import logging
import shutil
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from cnn import Cnn
from tqdm import trange
from Bqueue import Bqueue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Model")
terminal_columns = shutil.get_terminal_size().columns // 2


class Model:
    def __init__(self, input_dim, batch_size, img_train, lbl_train,
                 class_num, img_test, lbl_test, limit, ltm):
        r""" initializer for the DMA model

             Args:
                 input_dim: the dimension of the input data.
                 batch_size: the batch size if the training data at each run.
                 img_train: training input data.
                 lbl_train: training input labels data.
                 class_num: the number of classes to classify.
                 img_test: testing input data.
                 lbl_test: testing input label data.
                 limit: the limit of data that can be inserted to the long term memory at once.
                 ltm: the size of the memory of the long term memory.
            """
        self.input_dim = input_dim
        self.im_train = img_train
        self.lbl_train = lbl_train
        self.class_num = class_num
        self.batch_size = batch_size
        self.reg = LinearRegression()
        self.cnn = Cnn(self.input_dim, self.class_num)
        self.x_test = img_test
        self.y_test = lbl_test
        self.ltm = Bqueue(max_size=ltm)
        self.limit = limit
        self.scalar = StandardScaler()
        self.max = 1.0
        self.num_of_classes = []

    def reply(self):
        r""" reply the data that is stored in the long term memory."""
        samples = None
        labels = None
        """ pulling the data from the long term memory"""
        ltm_samples = np.array([s[0] for s in self.ltm.get_list()]).astype("float32")
        ltm_labels = np.array([s[1] for s in self.ltm.get_list()]).astype("float32")
        if ltm_samples.shape[0] > 0:
            for i in trange(self.class_num, desc="Replaying Data"):
                class_stm_idx = np.argwhere(np.argmax(ltm_labels, axis=1) == i).ravel()
                if class_stm_idx.shape[0] == 0:
                    break
                class_prototypes = ltm_samples[class_stm_idx]
                ll = ltm_labels[class_stm_idx]
                g_samples = np.repeat(
                    class_prototypes, self.limit // class_prototypes.shape[0], axis=0
                )
                g_labels = np.repeat(ll, self.limit // class_prototypes.shape[0], axis=0)
                """ if the long term memory is empty"""
                if i == 0:
                    samples = g_samples
                    labels = g_labels
                else:
                    samples = np.concatenate((samples, g_samples))
                    labels = np.concatenate((labels, g_labels))
            return samples, labels

    def fill_ltm(self, samples, labels):
        r""" filling the long term memory with new data

             Args:
                 samples: the features from the last layer of the CNN.
                 labels: the true labels for each feature.
            """
        logger.info("\rFilling LTM")
        stm_idx = np.arange(0, len(samples))
        for s in range(self.class_num):
            class_idx = np.argwhere(np.argmax(labels[stm_idx], axis=1) == s).ravel()
            np.random.shuffle(class_idx)
            class_samples = samples[class_idx]
            class_labels = labels[class_idx]
            class_samples, class_labels = shuffle(class_samples, class_labels)
            loop_iter = min(self.ltm.max_size // self.class_num, class_idx.shape[0])
            for i in range(loop_iter):
                self.ltm.push(
                    (class_samples[i], class_labels[i])
                )

    def train(self, samples, labels, test_samp, test_lbl, cnn_iter, sub_task):
        r""" training the DMA model

             Args:
                 samples: training input data.
                 labels: training input labels data.
                 test_samp: testing input data.
                 test_lbl: testing input labels data.
                 cnn_iter: number of epochs of the CNN
                 sub_task: the number of the dataset.
            """
        samples, labels = shuffle(samples, labels)
        logger.info("\r".center(terminal_columns, "="))
        logger.info(f"\r Sub-Task D{sub_task}")
        logger.info("\r".center(terminal_columns, "="))
        r_samples = None
        r_labels = None

        """checking if there any new labels."""
        new_labels = np.unique(np.argmax(labels, axis=1))
        for j in new_labels:
            if j not in self.num_of_classes:
                self.num_of_classes.append(j)

        print(new_labels)
        """ creating a new CNN."""
        self.cnn = Cnn(self.input_dim, self.class_num)
        cnn_loss, cnn_acc, cnn_CM = self.cnn.train(samples, labels, test_samp, test_lbl, cnn_iter, 64, True)
        cnn_out = self.cnn.get_last_layer_output(samples)
        if sub_task > 1 and self.ltm.max_size > 0:
            """ replaying od data from the long term memory"""
            m_samples, m_labels = self.reply()
            if m_samples is not None:
                r_samples = np.concatenate((cnn_out, m_samples))
                r_labels = np.concatenate((labels, m_labels))
                r_samples, r_labels = shuffle(r_samples, r_labels)
        else:
            r_samples = cnn_out
            r_labels = labels
        """ splitting the old a new data to test and train sets."""
        r_samples_train, r_samples_test, r_labels_train,  r_labels_test = \
            train_test_split(r_samples, r_labels, test_size=0.2, random_state=42)
        """ training the new kernel with new and old data"""
        self.reg = LinearRegression()
        self.reg.fit(r_samples_train, r_labels_train)
        score = self.reg.score(r_samples_test, r_labels_test)
        print(score)
        if self.ltm.max_size > 0:
            self.fill_ltm(cnn_out, labels)
        if len(self.num_of_classes) < 10:
            final_score = score*(len(self.num_of_classes)/10)
        else:
            final_score = score
        print(final_score)
        return final_score
