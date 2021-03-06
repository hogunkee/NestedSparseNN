import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default = 'cifar100', help = 'dataset')
parser.add_argument('--datapath', default = '../data/cifar-100/', help = 'path to dataset')
parser.add_argument('--num_epoch', type = int, default  = 300, help = 'number of epoch')

parser.add_argument('--batch_size', type = int, default  = 100, help = 'batch size')
parser.add_argument('--num_layers', type = int, default = 9, help = 'number of blocks in each level')
parser.add_argument('--beta', type = float, default = 1e-4, help = 'regularization rate')

parser.add_argument('--num_classes', type = int, default  = 10, help = 'number of classes')
parser.add_argument('--num_classes2', type = int, default  = 10, help = 'number of classes')
parser.add_argument('--num_classes3', type = int, default  = 10, help = 'number of classes')
parser.add_argument('--learning_rate', type = float, default = 0.1, help = 'learning rate')
parser.add_argument('--learning_rate2', type = float, default = 0.1, help = 'learning rate')
parser.add_argument('--learning_rate3', type = float, default = 0.1, help = 'learning rate')


parser.add_argument('--validation', type = float, default = 0.1, help = 'rate of validation split')
parser.add_argument('--train_mode', type = int, default  = 1, help = 'train mode(1 or 2)')

parser.add_argument('--outf', default = None, help = 'directory to save result')
parser.add_argument('--savename', required = True, help = 'save result')
parser.add_argument('--print_step', type = int, default = 20, help = 'print out training steps')

def get_config():
    return parser.parse_args()
