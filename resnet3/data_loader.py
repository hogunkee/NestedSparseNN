import pickle
import random
import os
import os.path
import numpy as np

def unpickle(file):
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'labels'], dict[b'data']

def one_hot(label, num_labels):
    zero = [0 for _ in range(num_labels)]
    for i in range(len(label)):
        tmp = zero.copy()
        tmp[label[i]] = 1
        label[i] = tmp

class Dataset():
    def __init__(self, dataset, path, num_classes):
        self.dataset = dataset
        self.path = path
        self.num_labels = num_classes

    def __call__(self, validation):
        train_labels = []
        train_data = []
        test_labels = []
        test_data = []

        if self.dataset == 'mnist':
            from tensorflow.examples.tutorials.mnist import input_data
            mnist = input_data.read_data_sets('./sample/MNIST_data', one_hot = True)
            train_labels = mnist.train.labels
            train_data = mnist.train.images
            test_labels = mnist.test.labels
            test_data = mnist.test.images

        else:
            for fname in os.listdir(self.path):
                fpath = os.path.join(self.path, fname)
                _label, _data = unpickle(fpath)
                print('load', fname)
                if fname == 'test_batch': 
                    test_labels = _label
                    test_data = _data
                else:
                    if train_labels==[]:
                        train_labels = _label
                        train_data = _data
                    else:
                        train_labels = train_labels + _label
                        train_data = np.concatenate((train_data, _data))

        tmp = list(zip(train_data, train_labels))
        random.shuffle(tmp)
        train_data, train_labels = zip(*tmp)

        data_train = list(train_data)
        labels_train = list(train_labels)
        data_test = test_data
        labels_test = test_labels

        if self.dataset != 'mnist':
            one_hot(labels_train, self.num_labels)
            one_hot(labels_test, self.num_labels)
        print('train data length: %d' %(len(labels_train)))
        print('test data length: %d' %(len(labels_test)))
        
        #return [data_train, data_val, data_test], [labels_train, labels_val, labels_test]
        return [data_train, labels_train], [data_test, labels_test]
        #return [data_train, labels_train], [data_val, labels_val], [data_test, labels_test]

