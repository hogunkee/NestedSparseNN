from config import get_config
from data_loader import * 
from vgg import *
from trainer import run_epoch

def main(config):
    if config.outf is None:
        config.outf = 'result'
    os.system('mkdir {0}'.format(config.outf))

    DataLoader = Dataset(config.datapath, config.num_classes)
    Input_train, Input_val, Input_test = DataLoader(config.validation)

	### writing results ###
    savepath = os.path.join(config.outf, config.savename)
    pfile = open(savepath, 'w+')
    pfile.write('dataset: '+str(config.dataset)+'\n')
    pfile.write('image size: '+str(config.image_size)+'\n')
    pfile.write('num epoch: '+str(config.num_epoch)+'\n')
    pfile.write('batch size: '+str(config.batch_size)+'\n')
    pfile.write('learning rate: '+str(config.learning_rate)+'\n')
    pfile.write('validation split: '+str(config.validation)+'\n')
    pfile.write('regularization rate: '+str(config.beta)+'\n')
    pfile.write('drop out: '+str(config.dropout)+'\n')
    pfile.close()

    with tf.Graph().as_default():
        trainModel = VGG(config, is_training = True)
        testModel = VGG(config, is_training = False)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            #print(init.node_def)
            sess.run(init)
            print("initialize all variables")

            for i in range(config.num_epoch):
                train_accur = run_epoch(sess, trainModel, Input_train, printOn = True)
                val_accur = run_epoch(sess, testModel, Input_val)
                print("Epoch: %d/%d" %(i+1, config.num_epoch))
                print("train accur: %.3f" %train_accur)
                print("val accur: %.3f" %val_accur)
                pfile = open(savepath, 'a+')
                pfile.write("\nEpoch: %d/%d\n" %(i+1, config.num_epoch))
                pfile.write("train accur: %.3f\n" %train_accur)
                pfile.write("val accur: %.3f\n" %val_accur)
                pfile.close()

            test_accur = run_epoch(sess, testModel, Input_test)
            print("test accur: %.3f" %test_accur)
            pfile = open(savepath, 'a+')
            pfile.write("\ntest accur: %.3f\n" %test_accur)
            pfile.close()

if __name__ == "__main__":
    config = get_config()
    main(config)
