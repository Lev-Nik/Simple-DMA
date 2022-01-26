"""Plotting class for the graphs"""
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1, 11)


def plot_graph(res1, title, color, res2=0):
    r""" plotting function for the scores of a chosen model.

         Args:
             res1: results of one model
             title: the graph title
             color: the graph color
             res2: results of the second model, default value is 0
        """
    if res2 != 0:
        plt.plot(x, res2, 'ro--')
        plt.plot(x, res1, color)
        plt.axis([0.5, 10.5, 0, 1])
        plt.xlabel('number of split dataset')
        plt.ylabel('accuracy')
        plt.title('DMA vs CNN')
        plt.grid()
    else:
        plt.plot(x, res1, color)
        plt.axis([1, 10, 0, 1])
        plt.xlabel('number of split dataset')
        plt.ylabel('accuracy')
        plt.title(title)
        plt.grid()
    plt.show()
