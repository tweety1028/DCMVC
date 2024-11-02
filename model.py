from dataloader import *

def get_knn_graph(data, k):
    num_samples = data.size(0)
    graph = torch.zeros(num_samples, num_samples, dtype=torch.int32, device=data.device)

    for i in range(num_samples):
        distance = torch.sum((data - data[i])**2, dim=1)
        _, small_indices = torch.topk(distance, k, largest=False)  # +1 to exclude self from neighbors
        # Fill 1 in the graph for the k nearest neighbors
        graph[i, small_indices[1:]] = 1

    # Ensure the graph is symmetric
    result_graph = torch.max(graph, graph.t())

    return result_graph

def get_W(mv_data, k):
    W = []
    mv_data_loader, num_views, num_samples, _ = get_all_multiview_data(mv_data)
    for _, (sub_data_views, _, _) in enumerate(mv_data_loader):
        for i in range(num_views):
            result_graph = get_knn_graph(sub_data_views[i], k)
            W.append(result_graph)
    return W

def psedo_labeling(model, dataset, batch_size):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    commonZ_list = []
    for batch_idx, (xs, y, _) in enumerate(loader):
        with torch.no_grad():
            xrs, zs = model(xs)
            commonz = model.fusion(zs)
            commonZ_list.append(commonz)
    commonZ = torch.cat(commonZ_list, dim=0)
    psedo_labels = model.clustering(commonZ)
    model.psedo_labels = psedo_labels

def pre_train(model, mv_data, batch_size, epochs, optimizer):

    mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)

    pre_train_loss_values = np.zeros(epochs+1, dtype=np.float64)

    criterion = torch.nn.MSELoss()
    for epoch in range(1, epochs+1):
        total_loss = 0.
        for batch_idx, (sub_data_views, _, _) in enumerate(mv_data_loader):
            xrs, _ = model(sub_data_views)
            loss_list = list()
            for idx in range(num_views):
                loss_list.append(criterion(sub_data_views[idx], xrs[idx]))
            loss = sum(loss_list)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        pre_train_loss_values[epoch] = total_loss
        if epoch % 10 == 0 or epoch == epochs:
            print('Pre-training, epoch {}, Loss:{:.7f}'.format(epoch, total_loss))

    return pre_train_loss_values

def contrastive_train(model, mv_data, mvc_loss, batch_size, epoch, W, alpha, beta, optimizer):
    model.train()
    mv_data_loader, num_views, num_samples, num_clusters = get_multiview_data(mv_data, batch_size)
    criterion = torch.nn.MSELoss()
    total_loss = 0.
    psedo_labeling(model, mv_data, batch_size)
    for batch_idx, (sub_data_views, _, sample_idx) in enumerate(mv_data_loader):
        batch_psedo_label = model.psedo_labels[sample_idx]
        y_matrix = (batch_psedo_label.view(-1, 1) == batch_psedo_label.view(1, -1)).int()
        xrs, zs = model(sub_data_views)
        common_z = model.fusion(zs)
        q_centers = model.compute_centers(common_z, batch_psedo_label)
        loss_list = list()
        for i in range(num_views):
            w = W[i][sample_idx][:,sample_idx]
            k_centers = model.compute_centers(zs[i], batch_psedo_label)
            loss_list.append(criterion(sub_data_views[i], xrs[i]))
            loss_list.append(alpha*mvc_loss.compute_cluster_loss(q_centers, k_centers, batch_psedo_label))
            loss_list.append(beta*mvc_loss.feature_loss(zs[i], common_z, w, y_matrix))
        loss = sum(loss_list)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        print('Contrastive_train, epoch {} loss:{:.7f}'.format(epoch, total_loss))

    return total_loss


def contrastive_largedatasetstrain(model, mv_data, mvc_loss, batch_size, epoch, k, alpha, beta, optimizer):
    model.train()
    mv_data_lodaer, num_views, num_samples, num_clusters = get_multiview_data(mv_data, batch_size)
    criterion = torch.nn.MSELoss()
    total_loss = 0.
    psedo_labeling(model, mv_data, batch_size)
    for batch_idx, (sub_data_views, _, sample_idx) in enumerate(mv_data_lodaer):
        batch_psedo_label = model.psedo_labels[sample_idx]
        y_matrix = (batch_psedo_label.view(-1, 1) == batch_psedo_label.view(1, -1)).int()
        xrs, zs = model(sub_data_views)
        common_z = model.fusion(zs)
        q_centers = model.compute_centers(common_z, batch_psedo_label)
        loss_list = list()
        for i in range(num_views):
            w = get_knn_graph(sub_data_views[i], k)
            k_centers = model.compute_centers(zs[i], batch_psedo_label)
            loss_list.append(criterion(sub_data_views[i], xrs[i]))

            loss_list.append(alpha*mvc_loss.compute_cluster_loss(q_centers, k_centers, batch_psedo_label))
            loss_list.append(beta*mvc_loss.feature_loss(zs[i], common_z, w, y_matrix))
        loss = sum(loss_list)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 5 == 0:
        print('Contrastive_train, epoch {} loss:{:.7f}'.format(epoch, total_loss))

    return total_loss