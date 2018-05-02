import numpy as np
import tensorflow as tf

def normalizer(X):
    mean = np.mean(X, axis=(0,1))
    std = np.std(X, axis=(0,1))
    normed_X = (X-mean)/(std+1e-7)
    return normed_X

def run_epoch(session, model, data, data2, train_mode, printOn = False):
    sum_loss1 = 0
    sum_regul_loss1 = 0
    sum_accur1 = 0
    sum_loss2 = 0
    sum_regul_loss2 = 0
    sum_accur2 = 0

    num_steps = len(data[0]) // model.batch_size
    num_steps2 = len(data2[0]) // model.batch_size

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
    if model.is_training and train_mode!=3:
        fetches1['train_step'] = model.train_step1
        fetches2['train_step'] = model.train_step2

    # batch 돌아가며 train
    if train_mode == 1:
        num_steps2 = num_steps
        for iter in range(num_steps):
            x1 = data[0][iter*model.batch_size : (iter+1)*model.batch_size]
            y1 = data[1][iter*model.batch_size : (iter+1)*model.batch_size]
            x2 = data2[0][iter*model.batch_size : (iter+1)*model.batch_size]
            y2 = data2[1][iter*model.batch_size : (iter+1)*model.batch_size]
            x1 = normalizer(x1)
            x2 = normalizer(x2)

            vals1 = session.run(fetches1, feed_dict = {
                model.learning_rate: model.lr, 
                model.X: x1, model.Y1: y1})
            vals2 = session.run(fetches2, feed_dict = {
                model.learning_rate2: model.lr2, 
                model.X: x2, model.Y2: y2})

            loss1 = vals1['loss']
            regul_loss1 = vals1['regul_loss']
            accur1 = vals1['accur']

            loss2 = vals2['loss']
            regul_loss2 = vals2['regul_loss']
            accur2 = vals2['accur']

            sum_loss1 += loss1
            sum_regul_loss1 += regul_loss1
            sum_accur1 += accur1

            sum_loss2 += loss2
            sum_regul_loss2 += regul_loss2
            sum_accur2 += accur2
            
            if printOn==True and model.is_training==True and (iter+1)%model.print_step==0:
                print("%d/%d steps.\nlv1 - loss: %.3f, regul loss: %.3f, accur: %.3f" %
                        (iter+1, num_steps, loss1, regul_loss1, accur1))
                print("lv2 - loss: %.3f, regul loss: %.3f, accur: %.3f" %
                        (loss2, regul_loss2, accur2))

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

    # loss_t = loss1 + loss2
    elif train_mode == 3:
        num_steps2 = num_step
        for iter in range(num_steps):
            vals1 = session.run(fetches1, feed_dict = {
                model.learning_rate: model.lr, 
                model.X: data[0][iter*model.batch_size : (iter+1)*model.batch_size], 
                model.Y1: data[1][iter*model.batch_size : (iter+1)*model.batch_size]
                })
            vals2 = session.run(fetches2, feed_dict = {
                model.learning_rate2: model.lr2, 
                model.X: data2[0][iter*model.batch_size : (iter+1)*model.batch_size], 
                model.Y2: data2[1][iter*model.batch_size : (iter+1)*model.batch_size]
                })
            if model.is_training:
                loss_t, _ = session.run([model.loss_t, model.train_step_t], feed_dict = {
                    model.learning_rate: model.lr, 
                    model.learning_rate2: model.lr2, 
                    model.X: data[0][iter*model.batch_size : (iter+1)*model.batch_size], 
                    model.Y: data[1][iter*model.batch_size : (iter+1)*model.batch_size]
                    })

            accur1 = vals1['accur']
            accur2 = vals2['accur']

            sum_accur1 += accur1
            sum_accur2 += accur2
            
            if printOn==True and model.is_training==True and (iter+1)%model.print_step==0:
                print("%d/%d steps.\nloss: %.3f, lv1-accur: %.3f, lv2-accur: %.3f" %
                        (iter+1, num_steps, loss_t, accur1, accur2))

    return (sum_accur1/num_steps), (sum_accur2/num_steps2)
