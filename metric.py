from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score, fowlkes_mallows_score
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def evaluate(label, pred):
    nmi = v_measure_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    f_score = fowlkes_mallows_score(label, pred)
    return nmi, ari, acc, pur, f_score

def inference(loader, model, data_size):
    model.eval()
    commonZ_list = []
    labels_vector = []
    for batch_idx, (xs, y, _) in enumerate(loader):
        with torch.no_grad():
            xrs, zs = model(xs)
            commonz = model.fusion(zs)
            commonZ_list.append(commonz)
        labels_vector.extend(y)
    commonZ = torch.cat(commonZ_list, dim=0)
    labels_vector = np.array(labels_vector).reshape(data_size)
    return labels_vector, commonZ

def valid(model, dataset, data_size):

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        drop_last=False,
    )
    labels_vector, commonZ = inference(test_loader, model, data_size)
    print('Clustering results:')
    y_pred = model.clustering(commonZ)
    y_pred = y_pred.cpu().numpy()
    nmi, ari, acc, pur, f_score = evaluate(labels_vector, y_pred)
    print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f} ARI = {:.4f} f_score = {:.4f}'.format(acc, nmi, pur, ari, f_score))