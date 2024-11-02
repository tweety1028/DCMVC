from network import Network
from metric import valid
from model import *
import numpy as np
import argparse
import random
import os
from loss import *

Dataname = 'ALOI-100'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname, help = '[CCV, RGB-D, Cora, ALOI-100, Hdigit, Digit-Product]')
parser.add_argument('--save_model', default=True, help='Saving the model after training.')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=0.5)
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--con_epochs", default=100)
parser.add_argument("--feature_dim", default=256)
parser.add_argument("--large_datasets", default=False, type=str)
parser.add_argument("--k", default=5)
parser.add_argument('--gpu', default='0', type=str, help='GPU device idx.')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    if args.dataset == "CCV":
        args.seed = 10
        args.k = 10
        alpha = 0.0001
        beta = 0.001

    elif args.dataset == "Digit-Product":
        args.large_datasets = True
        args.seed = 10
        args.k = 4
        alpha = 0.01
        beta = 0.1

    elif args.dataset == "RGB-D":
        args.seed = 5
        args.k = 10
        alpha = 0.01
        beta = 1

    elif args.dataset == 'Cora':
        args.seed = 10
        args.con_epochs = 100
        args.k = 10
        alpha = 0.01
        beta = 0.1

    elif args.dataset == 'ALOI-100':
        args.large_datasets = True
        args.batch_size = 256
        args.seed = 5
        args.con_epochs = 100
        args.k = 5
        alpha = 0.1
        beta = 0.001

    elif args.dataset == 'Hdigit':
        args.large_datasets = True
        args.seed = 10
        args.k = 5
        alpha = 1
        beta = 0.1

    print("==================================\nArgs:{}\n==================================".format(args))
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mv_data = MultiviewData(args.dataset, device)
    num_views = len(mv_data.data_views)
    num_samples = mv_data.labels.size
    num_clusters = np.unique(mv_data.labels).size
    input_sizes = np.zeros(num_views, dtype=int)
    for idx in range(num_views):
        input_sizes[idx] = mv_data.data_views[idx].shape[1]

    network = Network(num_views, num_samples, num_clusters, input_sizes, args.feature_dim)
    network = network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    mvc_loss = Loss(args.batch_size, num_clusters, args.temperature_l, args.temperature_f).to(device)

    epoch_list = []
    totalloss_list = []
    pre_train_loss_values = pre_train(network, mv_data, args.batch_size, args.mse_epochs, optimizer)

    if args.large_datasets == False:
        W = get_W(mv_data, k=args.k)
        for epoch in range(1, args.con_epochs + 1):
            total_loss = contrastive_train(network, mv_data, mvc_loss, args.batch_size, epoch, W, alpha, beta, optimizer)
            epoch_list.append(epoch)
            totalloss_list.append(total_loss)
        valid(network, mv_data, num_samples)

    else:
        for epoch in range(1, args.con_epochs + 1):
            total_loss = contrastive_largedatasetstrain(network, mv_data, mvc_loss, args.batch_size, epoch, args.k,
                                                        alpha, beta, optimizer)
            epoch_list.append(epoch)
            totalloss_list.append(total_loss)
        valid(network, mv_data, num_samples)

    if args.save_model:
        state = network.state_dict()
        torch.save(state, './models/%s.pth' % args.dataset)