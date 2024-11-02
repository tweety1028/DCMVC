import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_clustering


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Network(nn.Module):
    def __init__(self, num_views, num_samples, num_clusters, input_size, feature_dim):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for i in range(num_views):
            self.encoders.append(Encoder(input_size[i], feature_dim))
            self.decoders.append(Decoder(input_size[i], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.num_views = num_views
        self.num_samples = num_samples
        self.num_clusters = num_clusters
        self.psedo_labels = torch.zeros((self.num_samples,)).long().cuda()
        self.weights = nn.Parameter(torch.full((self.num_views,), 1 / self.num_views), requires_grad=True)

    def forward(self, xs):
        xrs = []
        zs = []
        for i in range(self.num_views):
            z = self.encoders[i](xs[i])
            xr = self.decoders[i](z)
            xrs.append(xr)
            zs.append(z)

        return xrs, zs

    def get_weights(self):
        softmax_weights = torch.softmax(self.weights, dim=0)
        weights = softmax_weights / torch.sum(softmax_weights)

        return weights


    def fusion(self, zs):
        weights = self.get_weights()
        weighted_zs = [z * weight for z, weight in zip(zs, weights)]
        stacked_zs = torch.stack(weighted_zs)
        common_z = torch.sum(stacked_zs, dim=0)

        return common_z

    def compute_centers(self, x, psedo_labels):
        n_samples = x.size(0)
        if len(psedo_labels.size()) > 1:
            weight = psedo_labels.T
        else:
            weight = torch.zeros(self.num_clusters, n_samples).to(x)  # L, N
            weight[psedo_labels, torch.arange(n_samples)] = 1
        weight = F.normalize(weight, p=1, dim=1)  # l1 normalization
        centers = torch.mm(weight, x)
        centers = F.normalize(centers, dim=1)

        return centers

    def clustering(self, features):
        kwargs = {
            'metric': 'cosine',
            'distributed': False,
            'random_state': 0,
            'n_clusters': self.num_clusters,
            'verbose': False
        }
        clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
        psedo_labels = clustering_model.fit_predict(features.to(dtype=torch.float64))

        return psedo_labels