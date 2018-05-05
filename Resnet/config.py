import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default = 'cifar10', help = 'dataset')

parser.add_argument('--batch_size', type = int, default  = 128, help = 'batch size')
parser.add_argument('--num_epoch', type = int, default  = 200, help = 'number of epoch')
parser.add_argument('--num_classes', type = int, default  = 10, help = 'number of classes')
parser.add_argument('--learning_rate', type = float, default = 0.1, help = 'learning rate')
parser.add_argument('--validation', type = float, default = 0.1, help = 'rate of validation split')
parser.add_argument('--beta', type = float, default = 2e-4, help = 'regularization rate')
parser.add_argument('--num_layers', type = int, default = 5, help = 'number of blocks in each level')

parser.add_argument('--outf', default = None, help = 'directory to save result')
parser.add_argument('--savename', required = True, help = 'save result')
parser.add_argument('--print_step', type = int, default = 130, help = 'print out training steps')
parser.add_argument('--shuffle', type = int, default = 1, help = 'shuffle training data')

def get_config():
    return parser.parse_args()
