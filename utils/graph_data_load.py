import math
import operator

import dgl
import torch
import numpy as np

from utils import graph_data_util


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.graphs, _ = dgl.data.utils.load_graphs(dataset.dgl_graphs_file, dataset.jobs[worker_id])
    dataset.length = sum([g.number_of_nodes() for g in dataset.graphs])
    np.random.seed(worker_info.seed % (2 ** 32))


class LoadBalanceGraphDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 dgl_graphs_file="./data/small.bin",
                 rw_hops=64,
                 restart_prob=0.8,
                 positional_embedding_size=32,
                 num_workers=12,
                 num_samples=2000,
                 num_copies=6,
                 ):
        super(LoadBalanceGraphDataset).__init__()
        self.rw_hops = rw_hops
        self.restart_prob = restart_prob
        self.step_dist = [1.0, 0.0, 0.0]
        self.positional_embedding_size = positional_embedding_size

        self.num_samples = num_samples

        self.dgl_graphs_file = dgl_graphs_file
        assert sum(self.step_dist) == 1.0
        assert positional_embedding_size % 2 == 0

        graph_sizes = dgl.data.utils.load_labels(dgl_graphs_file)["graph_sizes"].tolist()
        print("\n[LOAD DATA in train pretrain model] load data from %s ..." % self.dgl_graphs_file, flush=True)

        assert num_workers % num_copies == 0
        jobs = [list() for i in range(num_workers // num_copies)]
        workloads = [0] * (num_workers // num_copies)

        graph_sizes = sorted(enumerate(graph_sizes), key=operator.itemgetter(1), reverse=True)
        for idx, size in graph_sizes:
            argmin = workloads.index(min(workloads))
            workloads[argmin] += size
            jobs[argmin].append(idx)
        self.jobs = jobs * num_copies
        self.total = self.num_samples * num_workers

    def __len__(self):
        return self.num_samples * self.num_workers

    def __iter__(self):
        degrees = torch.cat([g.in_degrees().double() ** 0.75 for g in self.graphs])
        prob = degrees / torch.sum(degrees)
        samples = np.random.choice(self.length, size=self.num_samples, replace=True, p=prob.numpy())
        for idx in samples:
            yield self.__getitem__(idx)

    def _convert_idx(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()
        return graph_idx, node_idx

    def __getitem__(self, idx):
        graph_idx, node_idx = self._convert_idx(idx)

        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        if step == 0:
            other_node_idx = node_idx
        else:
            other_node_idx = dgl.contrib.sampling.random_walk(g=self.graphs[graph_idx], seeds=[node_idx],
                                                              num_traces=1, num_hops=step)[0][0][-1].item()

        max_nodes_per_seed = max(
            self.rw_hops,
            int(
                (
                        (self.graphs[graph_idx].in_degree(node_idx) ** 0.75)
                        * math.e
                        / (math.e - 1)
                        / self.restart_prob
                )
                + 0.5
            ),
        )
        traces = dgl.contrib.sampling.random_walk_with_restart(
            self.graphs[graph_idx],
            seeds=[node_idx, other_node_idx],
            restart_prob=self.restart_prob,
            max_nodes_per_seed=max_nodes_per_seed,
        )

        graph_q = graph_data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
        )
        graph_k = graph_data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=other_node_idx,
            trace=traces[1],
            positional_embedding_size=self.positional_embedding_size,
        )
        return graph_q, graph_k


class RecDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset,
                 rw_hops=64,
                 restart_prob=0.8,
                 positional_embedding_size=32
                 ):
        super(RecDataset).__init__()
        self.rw_hops = rw_hops
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = [1.0, 0.0, 0.0]
        assert sum(self.step_dist) == 1
        assert positional_embedding_size > 1

        dataset_class = graph_data_util.DatasetInfo(dataset)
        self.data = dataset_class.data
        self.graphs = [self._create_dgl_graph(self.data)]
        self.length = sum([g.number_of_nodes() for g in self.graphs])
        self.total = self.length

    def __len__(self):
        return self.length

    def _create_dgl_graph(self, data):
        graph = dgl.DGLGraph()
        src, dst = data.edge_index.tolist()
        num_nodes = data.num_nodes
        graph.add_nodes(num_nodes)
        graph.add_edges(src, dst)
        graph.add_edges(dst, src)
        graph.readonly()
        return graph

    def _convert_idx(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()
        return graph_idx, node_idx

    def __getitem__(self, idx):
        graph_idx, node_idx = self._convert_idx(idx)

        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        if step == 0:
            other_node_idx = node_idx
        else:
            other_node_idx = dgl.contrib.sampling.random_walk(g=self.graphs[graph_idx], seeds=[node_idx], num_traces=1,
                                                              num_hops=step)[0][0][-1].item()
        max_nodes_per_seed = max(
            self.rw_hops,
            int(
                (
                    self.graphs[graph_idx].out_degree(node_idx)
                    * math.e  # 2.718...
                    / (math.e - 1)
                    / self.restart_prob
                )
                + 0.5
            ),
        )
        traces = dgl.contrib.sampling.random_walk_with_restart(
            self.graphs[graph_idx],
            seeds=[node_idx, other_node_idx],
            restart_prob=self.restart_prob,
            max_nodes_per_seed=max_nodes_per_seed,
        )

        graph_q = graph_data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
        )
        graph_k = graph_data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=other_node_idx,
            trace=traces[1],
            positional_embedding_size=self.positional_embedding_size,
        )
        return graph_q, graph_k
