from config import get_config
from data_loader import * 
from resnet import *
from trainer import run_epoch

def main(config):
    if config.outf is None:
        config.outf = 'sample'
    os.system('mkdir {0}'.format(config.outf))

    CIFAR10_PATH = '../data/cifar-10'
    CIFAR100_PATH = '../data/cifar-100'
    if config.dataset=='cifar10':
        DataLoader = Dataset(config.dataset, CIFAR10_PATH, config.num_classes)
    elif config.dataset=='cifar100':
        DataLoader = Dataset(config.dataset, CIFAR100_PATH, config.num_classes)

    Input_train, Input_test = DataLoader(config.validation)

    ### writing results ###
    filename = config.savename #+'_pad:'+str(config.padding)+'_norm:'+str(config.norm)
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
        trainModel = ResNet(config, is_training = True)
        config.batch_size = 100
        testModel = ResNet(config, is_training = False)

        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            init = tf.global_variables_initializer()
            #print(init.node_def)
            sess.run(init)
            print("initialize all variables")

            pre_val = 0
            count = 0 
            count_epoch = 0
            num_change = 0
            max_accur = 0
            for i in range(config.num_epoch):
                ### data shuffle ###
                train_data, train_labels = Input_train[0], Input_train[1]
                tmp = list(zip(train_data, train_labels))
                random.shuffle(tmp)
                train_data, train_labels = zip(*tmp)
                Input_train = [train_data, train_labels]

                train_accur = run_epoch(sess, trainModel, Input_train, printOn = True)
                #val_accur = run_epoch(sess, testModel, Input_val)
                print("Epoch: %d/%d" %(i+1, config.num_epoch))
                print("train accur: %.3f" %train_accur)
                #print("val accur: %.3f" %val_accur)
                pfile = open(savepath, 'a+')
                pfile.write("\nEpoch: %d/%d\n" %(i+1, config.num_epoch))
                pfile.write("train accur: %.3f\n" %train_accur)
                #pfile.write("val accur: %.3f\n" %val_accur)
                pfile.close()

                test_accur = run_epoch(sess, testModel, Input_test)

                if test_accur > max_accur:
                    max_accur = test_accur

                print("test accur: %.3f\t max accur: %.3f" %(test_accur,max_accur))
                pfile = open(savepath, 'a+')
                pfile.write("test accur: %.3f\t max accur: %.3f\n" %(test_accur,max_accur))
                pfile.close()

                ### if validation accuracy decreased, decrease learning rate ###
                if (i>=100 and i<150):
                    trainModel.lr = config.learning_rate / 10
                elif (i>=150):
                    trainModel.lr = config.learning_rate / 100
                '''
                count_epoch += 1
                if (test_accur < pre_val):
                    count += 1
                if count == 4 and num_change < 4 and count_epoch > 20: # 10
                    trainModel.lr /= 10
                    print('change learning rate: %g' %(trainModel.lr))
                    pfile = open(savepath, 'a+')
                    pfile.write("\nchange learning rate: %g\n" %trainModel.lr)
                    pfile.close()
                    num_change += 1
                    count = 0
                    count_epoch = 0
                pre_val = test_accur 
                '''
            print('best accuracy:', max_accur)


if __name__ == "__main__":
    config = get_config()
    main(config)
