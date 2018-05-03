import pickle
import random
import os
import os.path
import numpy as np

def unpickle(file):
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

        for fname in os.listdir(self.path):
            fpath = os.path.join(self.path, fname)
            _label, _data = unpickle(fpath)
            print('load', fname)
            if train_labels==[]:
                train_labels = _label
                train_data = _data
            else:
                train_labels = train_labels + _label
                train_data = np.concatenate((train_data, _data))

        data = list(zip(train_data, train_labels))
        train, test = self.slice(data, validation)

        '''
        train = list(data)[:int(-validation * len(data))]
        test = list(data)[int(-validation * len(data)):]
        '''

        train_data, train_labels = zip(*train)
        test_data, test_labels = zip(*test)

        data_train = list(train_data)
        labels_train = list(train_labels)
        data_test = list(test_data)
        labels_test = list(test_labels)

        one_hot(labels_train, self.num_labels)
        one_hot(labels_test, self.num_labels)
        print('train data length: %d' %(len(labels_train)))
        print('test data length: %d' %(len(labels_test)))
        
        return [data_train, labels_train], [data_test, labels_test]


    def slice(self, data_list, validation):
        data_list = sorted(data_list, key=lambda k: k[1])
        
        c = 0
        c_start = 0
        c_end = 0
        tmp = [[] for i in range(self.num_labels)]

        end = len(data_list)
        for i in range(len(data_list)):
            if data_list[i][1] > c:
                c_end = i
                tmp[c] = data_list[c_start:c_end]
                c_start = c_end
                c += 1

            if data_list[i][1] >= self.num_labels:
                end = i
                break

        if c_end != end:
            print('c_end:',c_end,'end:',end)
            tmp[c] = data_list[c_start:]
            c += 1
        assert c==self.num_labels

        train, test = [], []
        for i in range(len(tmp)):
            random.shuffle(tmp[i])
            train += (tmp[i][:int(-validation * len(tmp[i]))])
            test += (tmp[i][int(-validation * len(tmp[i])):])

        random.shuffle(train)
        random.shuffle(test)

        return train, test

    '''
    def slice(self, data_list):
        data_list = sorted(data_list, key=lambda k: k[1])
        
        for i in range(len(data_list)):
            if data_list[i][1] >= self.num_labels:
                end = i
                break
        out = data_list[:end]
        random.shuffle(out)
        return out
    '''
