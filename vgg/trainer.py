from vgg import VGG
from data_loader import Dataset

class Trainer(object):
    def __init__(self, config):
        self.num_epoch = config.num_epoch
        self.batch_size = config.batch_size

        DataLoader = Dataset(config.datapath, config.num_classes)
        self.data, self.labels = DataLoader(self.validation)

        vggModel = VGG(config)
        self.trainModel = vggModel(is_training = True)
        self.testModel = vggModel(is_training = False)

    def run_epoch(session, model, data, labels, eval_op=None, printOn = False):
        costs = 0.0
        num_steps = len(data) // self.batch_size


