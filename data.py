import numpy as np

class DataSet(object):

    def __init__(self,
                images,
                labels):
        self.images = images
        self.labels = labels
        if images.shape[0] != labels.shape[0]:
            print("图像和标签行数不匹配")
        else:
            print("图像和标签行数匹配")
        self.num_examples = images.shape[0]
        self.epochs_completed = 0
        self.index_in_epoch = 0

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        if start + batch_size > self.num_examples or self.epochs_completed == 0:
            self.epochs_completed += 1
            shuffle_array = np.arange(self.num_examples)
            np.random.shuffle(shuffle_array)
            self.images = self.images[shuffle_array]
            self.labels = self.labels[shuffle_array]
            start = 0
            self.index_in_epoch = 0
        self.index_in_epoch += batch_size
        end = self.index_in_epoch
        return self.images[start:end], self.labels[start:end]
