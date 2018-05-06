from config import get_config
from data_loader import * 
from sparseresnet import *
from trainer import run_epoch

def main(config):
    if config.outf is None:
        config.outf = 'result'
    os.system('mkdir {0}'.format(config.outf))

    if config.level==1:
            DataLoader = Dataset(config.dataset, config.datapath, config.num_classes)
    elif config.level==2:
            DataLoader = Dataset(config.dataset, config.datapath, config.num_classes2)
    elif config.level==3:
            DataLoader = Dataset(config.dataset, config.datapath, config.num_classes3)
    Input_train, Input_test = DataLoader(config.validation)

    ### writing results ###
    filename = config.savename #+'_ver:'+str(config.version)+'_pad:'+str(config.padding)+'_norm:'+str(config.norm)
    savepath = os.path.join(config.outf, filename)
    pfile = open(savepath, 'w+')
    pfile.write('dataset: '+str(config.dataset)+'\n')
    pfile.write('level: '+str(config.level)+'\n')
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

            max = 0
            pre_val = 0
            count = 0 
            num_change = 0
            count_epoch = 0
            for i in range(config.num_epoch):
                ### data shuffle ###
                train_data, train_labels = Input_train[0], Input_train[1]

                tmp = list(zip(train_data, train_labels))
                random.shuffle(tmp)
                train_data, train_labels = zip(*tmp)

                Input_train = [train_data, train_labels]

                print("\nEpoch: %d/%d" %(i+1, config.num_epoch))

                train_accur = run_epoch(sess, trainModel, 
                        Input_train, config.level, True)

                test_accur = run_epoch(sess, testModel, 
                        Input_test, config.level)

                if (test_accur > max):
                    max = test_accur

                print("lv%d - train accur: %.3f" %(config.level, train_accur))
                pfile = open(savepath, 'a+')
                pfile.write("\nEpoch: %d/%d\n" %(i+1, config.num_epoch))
                pfile.write("lv%d - train: %.3f\n" %(config.level, train_accur))
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
                if count >= 3 and num_change < 4 and count_epoch > 30:
                    trainModel.lr /= 10
                    print('learning rate %g:' %(trainModel.lr))
                    pfile = open(savepath, 'a+')
                    pfile.write('\nChange Learning Rate')
                    pfile.write('\nlearning rate: %g\n' %trainModel.lr)
                    pfile.close()
                    num_change += 1
                    count = 0
                    count_epoch = 0
                pre_val = test_accur
                '''

                print("lv%d - test accur: %.3f / max: %.3f" %(config.level, test_accur, max))
                pfile = open(savepath, 'a+')
                pfile.write("lv%d - test accur: %.3f / max: %.3f\n" %(config.level, test_accur, max))
                pfile.close()
                

if __name__ == "__main__":
	config = get_config()
	main(config)
