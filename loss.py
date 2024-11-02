import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, batch_size, num_clusters, temperature_l, temperature_f):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.num_clusters = num_clusters
        self.temperature_l = temperature_l
        self.temperature_f = temperature_f
        self.similarity = nn.CosineSimilarity(dim=2)


    def feature_loss(self, zi, z, w, y_pse):
        cross_view_distance = self.similarity(zi.unsqueeze(1), z.unsqueeze(0)) / self.temperature_f
        N = z.size(0)
        w = w + torch.eye(N, dtype=int).to(w.device)
        positive_loss = (w & y_pse) * cross_view_distance
        inter_view_distance = self.similarity(zi.unsqueeze(1), zi.unsqueeze(0)) / self.temperature_f
        positive_loss = -torch.sum(positive_loss)
        negated_w = w ^ True
        negated_y = y_pse ^ True
        SMALL_NUM = torch.log(torch.tensor(1e-45)).to(zi.device)
        negtive_cross = (negated_w & negated_y) * cross_view_distance
        negtive_cross[negtive_cross == 0.] = SMALL_NUM
        negtive_inter = (negated_w & negated_y) * inter_view_distance
        negtive_inter[negtive_inter == 0.] = SMALL_NUM
        negtive_similarity = torch.cat((negtive_inter, negtive_cross), dim=1) / self.temperature_f
        negtive_loss = torch.logsumexp(negtive_similarity, dim=1, keepdim=False)
        negtive_loss = torch.sum(negtive_loss)
        return (positive_loss + negtive_loss) / N


    def compute_cluster_loss(self, q_centers, k_centers, psedo_labels):
        d_q = q_centers.mm(q_centers.T) / self.temperature_l
        d_k = (q_centers * k_centers).sum(dim=1) / self.temperature_l
        d_q = d_q.float()
        d_q[torch.arange(self.num_clusters), torch.arange(self.num_clusters)] = d_k

        # q -> k
        # d_q = q_centers.mm(k_centers.T) / temperature

        zero_classes = torch.arange(self.num_clusters).cuda()[torch.sum(F.one_hot(torch.unique(psedo_labels),
                                                                                 self.num_clusters), dim=0) == 0]
        mask = torch.zeros((self.num_clusters, self.num_clusters), dtype=torch.bool, device=d_q.device)
        mask[:, zero_classes] = 1
        d_q.masked_fill_(mask, -10)
        pos = d_q.diag(0)
        mask = torch.ones((self.num_clusters, self.num_clusters))
        mask = mask.fill_diagonal_(0).bool()
        neg = d_q[mask].reshape(-1, self.num_clusters - 1)
        loss = - pos + torch.logsumexp(torch.cat([pos.reshape(self.num_clusters, 1), neg], dim=1), dim=1)
        loss[zero_classes] = 0.
        loss = loss.sum() / (self.num_clusters - len(zero_classes))

        return loss