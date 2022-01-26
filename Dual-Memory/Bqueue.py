"""A simple class for managing the long term memory"""


class Bqueue(object):
    def __init__(self, max_size):
        r""" initializer the long term memory

         Args:
             max_size: the limit of the long term memory size.
            """
        self.max_size = max_size
        self.lst = []

    def push(self, st):
        r""" adding new data to the log term memory

             Args:
                 st: the data
            """
        self.lst.append(st)

    def get_list(self):
        r""" returning the data in the long term memory
            """
        return self.lst

    def is_empty(self):
        r""" checking if the memory is empty
            """
        return len(self.lst) == 0

    def is_full(self):
        r""" checking if the memory is full
                    """
        return len(self.lst) == self.max_size
