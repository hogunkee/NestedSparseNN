from config import get_config
from data_loader import * 
from sparseresnet import *
from trainer import run_epoch

def main(config):
    if config.outf is None:
        config.outf = 'result'
    os.system('mkdir {0}'.format(config.outf))

    DataLoader = Dataset(config.dataset, config.datapath, config.num_classes1, config.num_classes2, config.num_classes3)
    dataset_train, dataset_test = DataLoader(config.validation)

    ### writing results ###
    filename = config.savename #+'_ver:'+str(config.version)+'_pad:'+str(config.padding)+'_norm:'+str(config.norm)
    savepath = os.path.join(config.outf, filename)
    pfile = open(savepath, 'w+')
    pfile.write('dataset: '+str(config.dataset)+'\n')
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

            max_accur = [0, 0, 0]
            for lv in range(1,4):
                max = 0
                pre_val = 0
                count = 0 
                num_change = 0
                count_epoch = 0
                for i in range(config.num_epoch):
                    ### data shuffle ###
                    train_data, train_labels = dataset_train[lv-1][0], dataset_train[lv-1][1]

                    tmp = list(zip(train_data, train_labels))
                    random.shuffle(tmp)
                    train_data, train_labels = zip(*tmp)

                    Input_train = [train_data, train_labels]
                    Input_test = dataset_test[lv-1]

                    print("\nEpoch: %d/%d" %(i+1, config.num_epoch))

                    train_accur = run_epoch(sess, trainModel, 
                                    Input_train, lv, True)

                    test_accur = run_epoch(sess, testModel, 
                                    Input_test, lv)

                    if (test_accur > max):
                            max = test_accur

                    print("lv%d - train accur: %.3f" %(lv, train_accur))
                    pfile = open(savepath, 'a+')
                    pfile.write("\nEpoch: %d/%d\n" %(i+1, config.num_epoch))
                    pfile.write("lv%d - train: %.3f\n" %(lv, train_accur))
                    pfile.close()

                    ### if validation accuracy decreased, decrease learning rate ###
                    if (i>=100 and i<150):
                            trainModel.lr = config.learning_rate / 10
                    elif (i>=150):
                            trainModel.lr = config.learning_rate / 100 
                    else:
                            trainModel.lr = config.learning_rate

                    print("lv%d - test accur: %.3f / max: %.3f" %(lv, test_accur, max))
                    pfile = open(savepath, 'a+')
                    pfile.write("lv%d - test accur: %.3f / max: %.3f\n" %(lv, test_accur, max))
                    pfile.close()
                max_accur[lv-1] = max
                print('lv%d - max accur: %.3f'%(lv, max))
            print('MAX lv1: %.3f lv2: %.3f lv3: %.3f'%(max_accur[0], max_accur[1], max_accur[2]))
                

if __name__ == "__main__":
	config = get_config()
	main(config)
