from network import Network
from metric import valid
from model import *
import numpy as np
import argparse

Dataname = 'RGB-D'
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dataset', default=Dataname, help = '[CCV, RGB-D, Cora, ALOI-100, Hdigit, Digit-Product]')
parser.add_argument('--load_model', default=False, help='Testing if True or training.')
parser.add_argument("--feature_dim", default=256)

args = parser.parse_args()

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
checkpoint = torch.load('./models/%s.pth' % args.dataset)

network.load_state_dict(checkpoint)
print("Dataset:{}".format(args.dataset))
print("Loading models...")
valid(network, mv_data, num_samples)