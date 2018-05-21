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
    def __init__(self, dataset, path, level, num_classes1=10, num_classes2=15, num_classes3=20):
        self.dataset = dataset
        self.path = path
        self.level = level
        self.num_labels1 = num_classes1
        self.num_labels2 = num_classes2
        self.num_labels3 = num_classes3

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
        data = self.slice(data, validation)
        #train, test = self.slice(data, validation)

        dataset_train = []
        dataset_test = []

        train_data, train_label = zip(*(data[0]))
        test_data, test_label = zip(*(data[1]))

        train_data = list(train_data)
        train_label = list(train_label)
        test_data = list(test_data)
        test_label = list(test_label)
        if self.level==1:
            one_hot(train_label, self.num_labels1)
            one_hot(test_label, self.num_labels1)
        elif self.level==2:
            train_label = list(np.array(train_label) - self.num_labels1)
            test_label = list(np.array(test_label) - self.num_labels1)
            one_hot(train_label, self.num_labels2 - self.num_labels1)
            one_hot(test_label, self.num_labels2 - self.num_labels1)
        elif self.level==3:
            train_label = list(np.array(train_label) - self.num_labels2)
            test_label = list(np.array(test_label) - self.num_labels2)
            one_hot(train_label, self.num_labels3 - self.num_labels2)
            one_hot(test_label, self.num_labels3 - self.num_labels2)

        print('train data length: %d' %(len(train_label)))
        print('test data length: %d' %(len(test_label)))

        return [train_data, train_label], [test_data, test_label]


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
        '''


    def slice(self, data_list, validation):
        data_list = sorted(data_list, key=lambda k: k[1])
        
        if self.level==1:
            max_label = self.num_labels1
        elif self.level==2:
            max_label = self.num_labels2
        elif self.level==3:
            max_label = self.num_labels3

        c = 0
        c_start = 0
        c_end = 0
        tmp = [[] for i in range(max_label)]

        end = len(data_list)
        for i in range(len(data_list)):
            if data_list[i][1] > c:
                c_end = i
                tmp[c] = data_list[c_start:c_end]
                c_start = c_end
                c += 1

            if data_list[i][1] >= max_label:
                end = i
                break

        if c_end != end:
            print('c_end:',c_end,'end:',end)
            tmp[c] = data_list[c_start:]
            c += 1
        assert c == max_label

        train1, test1 = [], []
        train2, test2 = [], []
        train3, test3 = [], []
        for i in range(len(tmp)):
            random.shuffle(tmp[i])
            if i < self.num_labels1:
                train1 += (tmp[i][:int(-validation * len(tmp[i]))])
                test1 += (tmp[i][int(-validation * len(tmp[i])):])
            elif self.num_labels1 <= i < self.num_labels2:
                train2 += (tmp[i][:int(-validation * len(tmp[i]))])
                test2 += (tmp[i][int(-validation * len(tmp[i])):])
            elif self.num_labels2 <= i < self.num_labels3:
                train3 += (tmp[i][:int(-validation * len(tmp[i]))])
                test3 += (tmp[i][int(-validation * len(tmp[i])):])

        if self.level==1:
            random.shuffle(train1)
            random.shuffle(test1)
            return train1, test1
        elif self.level==2:
            random.shuffle(train2)
            random.shuffle(test2)
            return train2, test2
        elif self.level==3:
            random.shuffle(train3)
            random.shuffle(test3)
            return train3, test3
