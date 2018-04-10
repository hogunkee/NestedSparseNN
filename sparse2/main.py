from config import get_config
from data_loader import * 
from sparsevgg import *
from trainer import run_epoch

def main(config):
    if config.outf is None:
        config.outf = 'result'
    os.system('mkdir {0}'.format(config.outf))

    DataLoader = Dataset(config.dataset, config.datapath, config.num_classes)
    DataLoader2 = Dataset(config.dataset, config.datapath, config.num_classes2)
    Input_train, Input_val, Input_test = DataLoader(config.validation)
    Input_train2, Input_val2, Input_test2 = DataLoader2(config.validation)

    ### writing results ###
    filename = config.savename+'_ver:'+str(config.version)+'_pad:'+str(config.padding)+'_norm:'+str(config.norm)
    savepath = os.path.join(config.outf, filename)
    pfile = open(savepath, 'w+')
    pfile.write('version: '+str(config.version)+'\n')
    pfile.write('dataset: '+str(config.dataset)+'\n')
    #pfile.write('image size: '+str(config.image_size)+'\n')
    pfile.write('padding: '+str(config.padding)+'\n')
    pfile.write('pixel norm: '+str(config.norm)+'\n\n')
    pfile.write('num epoch: '+str(config.num_epoch)+'\n')
    pfile.write('batch size: '+str(config.batch_size)+'\n')
    pfile.write('initial learning rate: '+str(config.learning_rate)+'\n')
    pfile.write('validation split: '+str(config.validation)+'\n')
    pfile.write('regularization rate: '+str(config.beta)+'\n')
    pfile.write('drop out: '+str(config.dropout)+'\n')
    pfile.close()

    with tf.Graph().as_default():
        Model = SparseVGG
        trainModel = Model(config, is_training = True)
        testModel = Model(config, is_training = False)


        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            init = tf.global_variables_initializer()
            #print(init.node_def)
            sess.run(init)
            print("initialize all variables")

            pre_val = 0
            pre_val2 = 0
            count = 0 
            num_change = 0
            count_epoch = 0
            for i in range(config.num_epoch):
                train_accur1, train_accur2 = run_epoch(sess, trainModel, Input_train, Input_train2, config.train_mode, printOn = True)
                val_accur1, val_accur2 = run_epoch(sess, testModel, Input_val, Input_val2, config.train_mode)

                print("\nEpoch: %d/%d" %(i+1, config.num_epoch))
                print("lv1")
                print("train accur: %.3f\tval accur: %.3f" %(train_accur1, val_accur1))
                print("lv2")
                print("train accur: %.3f\tval accur: %.3f" %(train_accur2, val_accur2))
                pfile = open(savepath, 'a+')
                pfile.write("\nEpoch: %d/%d\n" %(i+1, config.num_epoch))
                pfile.write("lv1 - train: %.3f\tval: %.3f\n" %(train_accur1, val_accur1))
                pfile.write("lv2 - train: %.3f\tval: %.3f\n" %(train_accur2, val_accur2))
                pfile.close()

                ### if validation accuracy decreased, decrease learning rate ###
                count_epoch += 1
                if (val_accur1 < pre_val and val_accur2 < pre_val2):
                    count += 1
                if count == 3 and num_change < 4 and count_epoch > 10:
                    trainModel.lr /= 10
                    trainModel.lr2 /= 10
                    print('lv-1 learning rate %g:' %(trainModel.lr))
                    print('lv-2 learning rate %g:' %(trainModel.lr2))
                    pfile = open(savepath, 'a+')
                    pfile.write('\nlv-1 leaerning rate: %g\n' %trainModel.lr)
                    pfile.write('\nlv-2 leaerning rate: %g\n' %trainModel.lr2)
                    pfile.close()
                    num_change += 1
                    count = 0
                    count_epoch = 0
                pre_val = val_accur1
                pre_val2 = val_accur2

                test_accur1, test_accur2 = run_epoch(sess, testModel, Input_test, Input_test2, config.train_mode)

                print("lv1 - test accur: %.3f" %test_accur1)
                print("lv2 - test accur: %.3f\n" %test_accur2)
                pfile = open(savepath, 'a+')
                pfile.write("lv1 - test accur: %.3f\n" %test_accur1)
                pfile.write("lv2 - test accur: %.3f\n" %test_accur2)
                pfile.close()

if __name__ == "__main__":
	config = get_config()
	main(config)
