def run_epoch(session, model, data, printOn = False):
    sum_loss = 0
    sum_regul_loss = 0
    sum_accur = 0

    num_steps = len(data[0]) // model.batch_size
    fetches = {
            'loss' : model.loss,
            'accur' : model.accur,
            'regul_loss' : model.regularizer,
            'summary' : model.merged
            }
    if model.is_training:
        fetches['train_step'] = model.train_step
    for iter in range(num_steps):
        vals = session.run(fetches, feed_dict = {
            model.learning_rate: model.lr, 
            model.X: data[0][iter*model.batch_size : (iter+1)*model.batch_size], 
            model.Y: data[1][iter*model.batch_size : (iter+1)*model.batch_size]
            })
        loss = vals['loss']
        regul_loss = vals['regul_loss']
        accur = vals['accur']
        summary = vals['summary']

        sum_loss += loss
        sum_regul_loss += regul_loss
        sum_accur += accur
        
        if printOn and model.is_training and (iter+1)%model.print_step==0:
            print("%d/%d steps. loss: %.3f, regul loss: %.3f, accur: %.3f" %
                    (iter+1, num_steps, loss, regul_loss, accur))
    return (sum_accur/num_steps), summary
    #return (sum_loss/num_steps), (sum_regul_loss/num_steps), (sum_accur/num_steps)
