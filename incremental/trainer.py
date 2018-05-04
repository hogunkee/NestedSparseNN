import numpy as np
import tensorflow as tf

def normalizer(X):
    mean = np.mean(X, axis=(0,1))
    std = np.std(X, axis=(0,1))
    normed_X = (X-mean)/(std+1e-7)
    return normed_X

def run_epoch(session, model, data, data2, data3, train_mode, printOn = False):
    sum_loss1 = 0
    sum_loss2 = 0
    sum_loss3 = 0
    sum_regul_loss1 = 0
    sum_regul_loss2 = 0
    sum_regul_loss3 = 0
    sum_accur1 = 0
    sum_accur2 = 0
    sum_accur3 = 0

    num_steps = len(data[0]) // model.batch_size
    num_steps2 = len(data2[0]) // model.batch_size
    num_steps3 = len(data3[0]) // model.batch_size

    fetches1 = {
            'loss' : model.loss1,
            'accur' : model.accur1,
            'regul_loss' : model.regularizer1
            }
    fetches2 = {
            'loss' : model.loss2,
            'accur' : model.accur2,
            'regul_loss' : model.regularizer2
            }
    fetches3 = {
            'loss' : model.loss3,
            'accur' : model.accur3,
            'regul_loss' : model.regularizer3
            }
    if model.is_training:
        fetches1['train_step'] = model.train_step1
        fetches2['train_step'] = model.train_step2
        fetches3['train_step'] = model.train_step3

    # batch 돌아가며 train
    if train_mode == 1:
        num_steps2 = num_steps
        for iter in range(num_steps):
            fetch = {
                    'loss1' : model.loss1,
                    'loss2' : model.loss2,
                    'loss3' : model.loss3,
                    'accur1' : model.accur1,
                    'accur2' : model.accur2,
                    'accur3' : model.accur2,
                    'regul_loss1' : model.regularizer1,
                    'regul_loss2' : model.regularizer2,
                    'regul_loss3' : model.regularizer3
                    }
            if model.is_training:
                fetch['train_step1'] = model.train_step1
                fetch['train_step2'] = model.train_step2
                fetch['train_step3'] = model.train_step3


            x = data[0][iter*model.batch_size : (iter+1)*model.batch_size]
            y = data[1][iter*model.batch_size : (iter+1)*model.batch_size]
            x = normalizer(x)

            vals = session.run(fetch, feed_dict = {
                model.learning_rate: model.lr, 
                model.learning_rate2: model.lr2, 
                model.learning_rate3: model.lr3, 
                model.X: x, model.Y1: y, model.Y2: y, model.Y3: y})

            loss1 = vals['loss1']
            regul_loss1 = vals['regul_loss1']
            accur1 = vals['accur1']

            loss2 = vals['loss2']
            regul_loss2 = vals['regul_loss2']
            accur2 = vals['accur2']

            loss3 = vals['loss3']
            regul_loss3 = vals['regul_loss3']
            accur3 = vals['accur3']

            sum_loss1 += loss1
            sum_regul_loss1 += regul_loss1
            sum_accur1 += accur1

            sum_loss2 += loss2
            sum_regul_loss2 += regul_loss2
            sum_accur2 += accur2

            sum_loss3 += loss3
            sum_regul_loss3 += regul_loss3
            sum_accur3 += accur3
            
            if printOn==True and model.is_training==True and (iter+1)%model.print_step==0:
                print("%d/%d steps.\nlv1 - loss: %.3f, regul loss: %.3f, accur: %.3f" %
                        (iter+1, num_steps, loss1, regul_loss1, accur1))
                print("lv2 - loss: %.3f, regul loss: %.3f, accur: %.3f" %
                        (loss2, regul_loss2, accur2))
                print("lv3 - loss: %.3f, regul loss: %.3f, accur: %.3f" %
                        (loss3, regul_loss3, accur3))

    # epoch씩 train
    elif train_mode == 2:
        for iter in range(num_steps):
            x1 = data[0][iter*model.batch_size : (iter+1)*model.batch_size]
            y1 = data[1][iter*model.batch_size : (iter+1)*model.batch_size]
            x1 = normalizer(x1)

            vals1 = session.run(fetches1, feed_dict = {
                model.learning_rate: model.lr, 
                model.X: x1, model.Y1: y1})

            loss1 = vals1['loss']
            regul_loss1 = vals1['regul_loss']
            accur1 = vals1['accur']

            sum_loss1 += loss1
            sum_regul_loss1 += regul_loss1
            sum_accur1 += accur1
            
            if printOn==True and model.is_training==True and (iter+1)%model.print_step==0:
                print("%d/%d steps.\nlv1 - loss: %.3f, regul loss: %.3f, accur: %.3f" %
                        (iter+1, num_steps, loss1, regul_loss1, accur1))
                
        for iter in range(num_steps2):
            x2 = data2[0][iter*model.batch_size : (iter+1)*model.batch_size]
            y2 = data2[1][iter*model.batch_size : (iter+1)*model.batch_size]
            x2 = normalizer(x2)

            vals2 = session.run(fetches2, feed_dict = {
                model.learning_rate2: model.lr2, 
                model.X : x2, model.Y2: y2})

            loss2 = vals2['loss']
            regul_loss2 = vals2['regul_loss']
            accur2 = vals2['accur']

            sum_loss2 += loss2
            sum_regul_loss2 += regul_loss2
            sum_accur2 += accur2
            
            if printOn==True and model.is_training==True and (iter+1)%model.print_step==0:
                print("%d/%d steps.\nlv2 - loss: %.3f, regul loss: %.3f, accur: %.3f" %
                        (iter+1, num_steps2, loss2, regul_loss2, accur2))

        for iter in range(num_steps3):
            x3 = data3[0][iter*model.batch_size : (iter+1)*model.batch_size]
            y3 = data3[1][iter*model.batch_size : (iter+1)*model.batch_size]
            x3 = normalizer(x3)

            vals3 = session.run(fetches3, feed_dict = {
                model.learning_rate3: model.lr3, 
                model.X : x3, model.Y3: y3})

            loss3 = vals3['loss']
            regul_loss3 = vals3['regul_loss']
            accur3 = vals3['accur']

            sum_loss3 += loss3
            sum_regul_loss3 += regul_loss3
            sum_accur3 += accur3
            
            if printOn==True and model.is_training==True and (iter+1)%model.print_step==0:
                print("%d/%d steps.\nlv3 - loss: %.3f, regul loss: %.3f, accur: %.3f" %
                        (iter+1, num_steps3, loss3, regul_loss3, accur3))

    return (sum_accur1/num_steps), (sum_accur2/num_steps2), (sum_accur3/num_steps3)
