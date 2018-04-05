import tensorflow as tf

def run_epoch(session, model, data, printOn = False):
    sum_loss = 0
    sum_regul_loss = 0
    sum_accur = 0

    num_steps = len(data[0]) // model.batch_size
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
    if model.is_training:
        fetches1['train_step'] = model.train_step1
        fetches2['train_step'] = model.train_step2

    for iter in range(num_steps):
        vals1 = session.run(fetches1, feed_dict = {
            model.learning_rate: model.lr, 
            model.X: data[0][iter*model.batch_size : (iter+1)*model.batch_size], 
            model.Y: data[1][iter*model.batch_size : (iter+1)*model.batch_size]
            })
        vals2 = session.run(fetches2, feed_dict = {
            model.learning_rate: model.lr, 
            model.X: data[0][iter*model.batch_size : (iter+1)*model.batch_size], 
            model.Y: data[1][iter*model.batch_size : (iter+1)*model.batch_size]
            })

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
            
    return (sum_accur1/num_steps), (sum_accur2/num_steps)
