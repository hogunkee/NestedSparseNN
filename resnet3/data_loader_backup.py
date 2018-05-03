import pickle
import random
import os
import os.path
import numpy as np

def unpickle(file):
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'labels'], dict[b'data']

def unpickle2(file):
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'fine_labels'], dict[b'data']
    #return dict[b'coarse_labels'], dict[b'data']

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
                if self.dataset == 'cifar10':
                    _label, _data = unpickle(fpath)
                elif self.dataset == 'cifar100':
                    _label, _data = unpickle2(fpath)
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

        train = list(zip(train_data, train_labels))
        test = list(zip(test_data, test_labels))

        train = self.slice(train)
        test = self.slice(test)

        train_data, train_labels = zip(*train)
        test_data, test_labels = zip(*test)
        data_train = list(train_data)
        labels_train = list(train_labels)
        data_test = list(test_data)
        labels_test = list(test_labels)

        if self.dataset != 'mnist':
            one_hot(labels_train, self.num_labels)
            one_hot(labels_test, self.num_labels)
        print('train data length: %d' %(len(labels_train)))
        print('test data length: %d' %(len(labels_test)))
        
        return [data_train, labels_train], [data_test, labels_test]

    def slice(self, data_list):
        data_list = sorted(data_list, key=lambda k: k[1])
        
        end = 0
        for i in range(len(data_list)):
            if data_list[i][1] >= self.num_labels:
                end = i
                break
        if end==0:
            out = data_list
        else:
            out = data_list[:end]
        random.shuffle(out)
        return out
