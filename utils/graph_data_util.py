import dgl
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg
import sklearn.preprocessing as preprocessing
from collections import namedtuple

from utils.data_load import Data as recData


GraphData = namedtuple("Data", ["x", "edge_index", "y", "num_nodes"])


class DatasetInfo(object):
    def __init__(self, dataset):
        edge_index = self._preprocess(dataset)
        self.data = GraphData(x=None, edge_index=edge_index, y=None, num_nodes=self.n_users+self.n_items)

    def get(self, idx):
        assert idx == 0
        return self.data

    def _preprocess(self, dataset):
        path_data = './data/' + dataset + '/raw/raw.data'

        dataset_class = recData(path_data, header=['user_id', 'item_id', 'rating', 'timestamp'],
                                test_bool=False, sep='\t', type='pretrain')
        _, self.n_users, self.n_items = dataset_class.load_file_as_dataFrame()

        with open(path_data) as f:
            edge_list = []
            for line in f:
                x, y, r = list(line.strip().split('\t'))
                x, y = int(x), int(y)
                y = y + self.n_users
                edge_list.append([x, y])

        return torch.LongTensor(edge_list).t()


def get_traces(graph, seeds, size, num_hop=1):
    if num_hop == 1:
        lists = []
        for seed in seeds:
            lists.append(torch.tensor(graph.out_edges(torch.tensor(seed))[1].tolist()))
        if size < len(lists) and size > 0:
            lists = np.random.choice(a=lists, size=size, replace=False).tolist()
        return lists

    if num_hop == 2:
        lists = []
        for seed in seeds:
            list = graph.out_edges(torch.tensor(seed))[1].tolist()
            cnt = 0
            for node_1hop in graph.out_edges(torch.tensor(seed))[1]:
                for node_2hop in graph.out_edges(node_1hop)[1].tolist():
                    if node_2hop not in list:
                        cnt += 1
                        if cnt > size and size > 0:
                            break
                        list.append(node_2hop)

            lists.append(torch.tensor(list))
        if size < len(lists) and size > 0:
            lists = np.random.choice(a=lists, size=size, replace=False).tolist()
        return lists

    if num_hop == 3:
        lists = []
        for seed in seeds:
            list = graph.out_edges(torch.tensor(seed))[1].tolist()
            cnt = 0
            for node_1hop in graph.out_edges(torch.tensor(seed))[1]:
                for node_2hop in graph.out_edges(node_1hop)[1].tolist():
                    for node_3hop in graph.out_edges(node_2hop)[1].tolist():
                        if node_3hop not in list:
                            cnt += 1
                            if cnt > size and size > 0:
                                break
                            list.append(node_3hop)
            lists.append(torch.tensor(list))
        if size < len(lists) and size > 0:
            lists = np.random.choice(a=lists, size=size, replace=False).tolist()
        return lists


def _rwr_trace_to_dgl_graph(g, seed, trace, positional_embedding_size, entire_graph=False):
    subv = torch.unique(torch.cat(trace)).tolist()
    try:
        subv.remove(seed)
    except ValueError:
        pass
    subv = [seed] + subv
    if entire_graph:
        subg = g.subgraph(g.nodes())
    else:
        subg = g.subgraph(subv)

    subg = _add_undirected_graph_positional_embedding(subg, positional_embedding_size)

    subg.ndata["seed"] = torch.zeros(subg.number_of_nodes(), dtype=torch.long)
    if entire_graph:
        subg.ndata["seed"][seed] = 1
    else:
        subg.ndata["seed"][0] = 1
    return subg


def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype("float64")
    ncv = min(n, max(2 * k + 1, 20))
    # follows https://stackoverflow.com/questions/52386942/scipy-sparse-linalg-eigsh-with-fixed-seed
    v0 = np.random.rand(n).astype("float64")
    for i in range(retry):
        try:
            s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
        except sparse.linalg.eigen.arpack.ArpackError:
            # print("arpack error, retry=", i)
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm="l2")
    x = torch.from_numpy(x.astype("float32"))
    x = F.pad(x, (0, hidden_size - k), "constant", 0)
    return x


def _add_undirected_graph_positional_embedding(g, hidden_size, retry=10):
    # We use eigenvectors of normalized graph laplacian as vertex features.
    # It could be viewed as a generalization of positional embedding in the
    # attention is all you need paper.
    # Recall that the eignvectors of normalized laplacian of a line graph are cos/sin functions.
    # See section 2.4 of http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
    n = g.number_of_nodes()
    adj = g.adjacency_matrix_scipy(transpose=False, return_edge_ids=False).astype(float)
    norm = sparse.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    laplacian = norm * adj * norm
    k = min(n - 2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    g.ndata["pos_undirected"] = x.float()
    return g


def batcher():
    def batcher_dev(batch):
        graph_q, graph_k = zip(*batch)
        graph_q, graph_k = dgl.batch(graph_q), dgl.batch(graph_k)
        return graph_q, graph_k

    return batcher_dev


def labeled_batcher():
    def batcher_dev(batch):
        graph_q, label = zip(*batch)
        graph_q = dgl.batch(graph_q)
        return graph_q, torch.LongTensor(label)

    return batcher_dev


