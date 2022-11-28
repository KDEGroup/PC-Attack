import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SeqEncoder(nn.Module):
    def __init__(
        self,
        positional_embedding_size=32,
        max_degree=512,
        degree_embedding_size=16,
        hidden_size=64,
        num_layers=5,
        degree_input=True,
        # seq_model="lstm",
    ):
        super(SeqEncoder, self).__init__()
        self.max_degree = max_degree
        self.degree_input = degree_input
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        if degree_input:
            input_size = positional_embedding_size + degree_embedding_size + 1
            self.degree_embedding = nn.Embedding(num_embeddings=max_degree + 1, embedding_dim=degree_embedding_size)
        else:
            input_size = positional_embedding_size + 1

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, g):
        if self.degree_input:
            device = g.ndata["seed"].device
            degrees = g.in_degrees()
            if device != torch.device("cpu"):
                degrees = degrees.cuda(device)

            n_feat = torch.cat(
                (
                    g.ndata["pos_undirected"],
                    self.degree_embedding(degrees.clamp(0, self.max_degree)),
                    g.ndata["seed"].unsqueeze(1).float(),
                ),
                dim=-1,
            )
        else:
            n_feat = torch.cat(
                (
                    g.ndata["pos_undirected"],
                    g.ndata["seed"].unsqueeze(1).float(),
                ),
                dim=-1,
            )

        _seeds = g.ndata['seed'].cpu().numpy()
        seeds = np.where(_seeds == 1)[0]

        seq_list = []
        for seed in seeds:
            nodes_1hop = g.out_edges(torch.tensor(seed))[1].tolist()

            nodes_2hop = []
            for node_1hop in nodes_1hop:
                for node_2hop in g.out_edges(node_1hop)[1].tolist():
                    if node_2hop not in nodes_2hop:
                        nodes_2hop.append(node_2hop)

            nodes_seq = nodes_1hop + nodes_2hop

            seq = n_feat[nodes_seq].unsqueeze(0)
            h0 = torch.zeros(self.num_layers, seq.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, seq.size(0), self.hidden_size).to(device)

            out, _ = self.lstm(seq, (h0, c0))
            seq_list.append(out[:, -1, :])
        seq_list = torch.cat(seq_list, dim=0)

        return seq_list
