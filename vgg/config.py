import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default = 'cifar10', help = 'dataset')
parser.add_argument('--dataroot', required = True, heko = 'path to dataset')

parser.add_argument('--batch_size', default  = 50, help = 'batch size')
parser.add_argument('--num_epoch', default  = 300, help = 'number of epoch')
parser.add_argument('--num_classes', default  = 10, help = 'number of classes')
parser.add_argument('--learning_rate', default = 1e-5, help = 'learning rate')
parser.add_argument('--validation', default = 0.1, help = 'rate of validation split')
parser.add_argument('--beta', default = 5e-4, help = 'regularization rate')
parser.add_argument('--dropout', default = 0.5, help = 'probability of drop out on fc layers')

parser.add_argument('--outf', default = None, help = 'directory to save result')

def get_config():
    return parser.parse_args()
