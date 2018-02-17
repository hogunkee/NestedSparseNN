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

    with tf.Graph().as_default():
        trainModel = VGG(config, is_training = True)
        testModel = VGG(config, is_training = False)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            #print(init.node_def)
            sess.run(init)
            print("initialize all variables")

            for i in range(config.num_epoch):
                print("Epoch: %d/%d" %(i+1, config.num_epoch))
                train_accur = run_epoch(sess, trainModel, Input_train, printOn = True)
                print("train accur: %.3f" %train_accur)
                val_accur = run_epoch(sess, testModel, Input_val)
                print("val accur: %.3f" %val_accur)

            test_accur = run_epoch(sess, testModel, Input_test)
            print("test accur: %.3f" %test_accur)

if __name__ == "__main__":
    config = get_config()
    main(config)
