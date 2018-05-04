from config import get_config
from data_loader import * 
from sparseresnet import *
from trainer import run_epoch

def main(config):
    if config.outf is None:
        config.outf = 'result'
    os.system('mkdir {0}'.format(config.outf))

    DataLoader = Dataset(config.dataset, config.datapath, config.num_classes)
    DataLoader2 = Dataset(config.dataset, config.datapath, config.num_classes2)
    DataLoader3 = Dataset(config.dataset, config.datapath, config.num_classes3)
    Input_train, Input_test = DataLoader(config.validation)
    if config.train_mode != 1:
        Input_train2, Input_test2 = DataLoader2(config.validation)
        Input_train3, Input_test3 = DataLoader3(config.validation)
    else:
        Input_train2 = Input_train3 = Input_train
        Input_test2 = Input_test3 = Input_test

    ### writing results ###
    filename = config.savename #+'_ver:'+str(config.version)+'_pad:'+str(config.padding)+'_norm:'+str(config.norm)
    savepath = os.path.join(config.outf, filename)
    pfile = open(savepath, 'w+')
    pfile.write('dataset: '+str(config.dataset)+'\n')
    pfile.write('num epoch: '+str(config.num_epoch)+'\n')
    pfile.write('batch size: '+str(config.batch_size)+'\n')
    pfile.write('initial learning rate: '+str(config.learning_rate)+'\n')
    pfile.write('validation split: '+str(config.validation)+'\n')
    pfile.write('regularization rate: '+str(config.beta)+'\n')
    pfile.write('n: '+str(config.num_layers)+'\n')
    pfile.close()

    with tf.Graph().as_default():
        Model = SparseResNet
        trainModel = Model(config, is_training = True)
        testModel = Model(config, is_training = False)


        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            init = tf.global_variables_initializer()
            #print(init.node_def)
            sess.run(init)
            print("initialize all variables")

            max1 = 0
            max2 = 0
            max3 = 0
            pre_val = 0
            pre_val2 = 0
            pre_val3 = 0
            count = 0 
            num_change = 0
            count_epoch = 0
            for i in range(config.num_epoch):
                ### data shuffle ###
                train_data, train_labels = Input_train[0], Input_train[1]
                train_data2, train_labels2 = Input_train2[0], Input_train2[1]
                train_data3, train_labels3 = Input_train3[0], Input_train3[1]

                tmp = list(zip(train_data, train_labels))
                tmp2 = list(zip(train_data2, train_labels2))
                tmp3 = list(zip(train_data3, train_labels3))
                random.shuffle(tmp)
                random.shuffle(tmp2)
                random.shuffle(tmp3)
                train_data, train_labels = zip(*tmp)
                train_data2, train_labels2 = zip(*tmp2)
                train_data3, train_labels3 = zip(*tmp3)

                Input_train = [train_data, train_labels]
                Input_train2 = [train_data2, train_labels2]
                Input_train3 = [train_data3, train_labels3]

                train_accur1, train_accur2, train_accur3 = run_epoch(sess, trainModel, 
                        Input_train, Input_train2, Input_train3, config.train_mode, True)

                test_accur1, test_accur2, test_accur3 = run_epoch(sess, testModel, 
                        Input_test, Input_test2, Input_test3, config.train_mode)

                print("\nEpoch: %d/%d" %(i+1, config.num_epoch))
                print("lv1 - train accur: %.3f" %train_accur1)
                print("lv2 - train accur: %.3f" %train_accur2)
                print("lv3 - train accur: %.3f" %train_accur3)
                pfile = open(savepath, 'a+')
                pfile.write("\nEpoch: %d/%d\n" %(i+1, config.num_epoch))
                pfile.write("lv1 - train: %.3f\n" %train_accur1)
                pfile.write("lv2 - train: %.3f\n" %train_accur2)
                pfile.write("lv3 - train: %.3f\n" %train_accur3)
                pfile.close()

                ### if validation accuracy decreased, decrease learning rate ###
                count_epoch += 1
                if (test_accur1<pre_val and test_accur2<pre_val2 and test_accur3<pre_val3):
                    count += 1
                if count >= 3 and num_change < 4 and count_epoch > 30:
                    trainModel.lr /= 10
                    trainModel.lr2 /= 10
                    trainModel.lr3 /= 10
                    print('lv-1 learning rate %g:' %(trainModel.lr))
                    print('lv-2 learning rate %g:' %(trainModel.lr2))
                    print('lv-3 learning rate %g:' %(trainModel.lr3))
                    pfile = open(savepath, 'a+')
                    pfile.write('\nChange Learning Rate')
                    pfile.write('\nlv-1 learning rate: %g\n' %trainModel.lr)
                    pfile.write('\nlv-2 learning rate: %g\n' %trainModel.lr2)
                    pfile.write('\nlv-3 learning rate: %g\n' %trainModel.lr3)
                    pfile.close()
                    num_change += 1
                    count = 0
                    count_epoch = 0
                pre_val = test_accur1
                pre_val2 = test_accur2
                pre_val3 = test_accur3


                print("lv1 - test accur: %.3f" %test_accur1)
                print("lv2 - test accur: %.3f" %test_accur2)
                print("lv3 - test accur: %.3f" %test_accur3)
                pfile = open(savepath, 'a+')
                pfile.write("lv1 - test accur: %.3f\n" %test_accur1)
                pfile.write("lv2 - test accur: %.3f\n" %test_accur2)
                pfile.write("lv3 - test accur: %.3f\n" %test_accur3)
                pfile.close()
                
                if (test_accur1 > max1):
                    max1 = test_accur1
                if (test_accur2 > max2):
                    max2 = test_accur2
                if (test_accur3 > max3):
                    max3 = test_accur3
                print("max: %.3f / %.3f / %.3f\n" %(max1, max2, max3))

if __name__ == "__main__":
	config = get_config()
	main(config)
