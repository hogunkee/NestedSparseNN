import numpy as np

def normalizer(X):
    mean = np.mean(X, axis=(0,1))
    std = np.std(X, axis=(0,1))
    normed_X = (X-mean)/(std+1e-7)
    return normed_X

def run_epoch(session, model, data, level, printOn = False):
    sum_loss = 0
    sum_regul_loss = 0
    sum_accur = 0

    num_steps = len(data[0]) // model.batch_size

    if level==1:
        fetches = {
            'loss' : model.loss1,
            'accur' : model.accur1,
            'regul_loss' : model.regularizer1
            }
    elif level==2:
        fetches = {
            'loss' : model.loss2,
            'accur' : model.accur2,
            'regul_loss' : model.regularizer2
            }
    elif level==3:
        fetches = {
            'loss' : model.loss3,
            'accur' : model.accur3,
            'regul_loss' : model.regularizer3
            }
		
    if model.is_training:
        if level==1:
            fetches['train_step'] = model.train_step1
        elif level==2:
            fetches['train_step'] = model.train_step2
        elif level==3:
            fetches['train_step'] = model.train_step3

    for iter in range(num_steps):
        x_ = data[0][iter*model.batch_size : (iter+1)*model.batch_size]
        y_ = data[1][iter*model.batch_size : (iter+1)*model.batch_size]
        x_norm = normalizer(x_)

        vals = session.run(fetches, feed_dict = {
            model.learning_rate: model.lr, 
            model.X: x_norm, model.Y: y_})

        loss = vals['loss']
        regul_loss = vals['regul_loss']
        accur = vals['accur']

        sum_loss += loss
        sum_regul_loss += regul_loss
        sum_accur += accur
        
        if printOn and model.is_training and (iter+1)%model.print_step==0:
            print("%d/%d steps. loss: %.3f, regul loss: %.3f, accur: %.3f" %
                    (iter+1, num_steps, loss, regul_loss, accur))

    return (sum_accur/num_steps)
