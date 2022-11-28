import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pretrain.gin import UnsupervisedGIN


class GraphEncoder(nn.Module):
    def __init__(
        self,
        positional_embedding_size=32,
        max_degree=512,
        degree_embedding_size=16,
        output_dim=64,
        node_hidden_dim=64,
        num_layers=5,
        norm=False,
        gnn_model="gin",
        degree_input=True,
    ):
        super(GraphEncoder, self).__init__()

        if degree_input:
            node_input_dim = positional_embedding_size + degree_embedding_size + 1
        else:
            node_input_dim = positional_embedding_size + 1

        if gnn_model == "gin":
            self.gnn = UnsupervisedGIN(
                num_layers=num_layers,
                num_mlp_layers=2,
                input_dim=node_input_dim,
                hidden_dim=node_hidden_dim,
                output_dim=output_dim,
                final_dropout=0.5,
                learn_eps=False,
                graph_pooling_type="sum",
                neighbor_pooling_type="sum",
                use_selayer=False,
            )
        self.gnn_model = gnn_model
        self.norm = norm
        self.max_degree = max_degree
        self.degree_input = degree_input

        if degree_input:
            self.degree_embedding = nn.Embedding(num_embeddings=max_degree + 1, embedding_dim=degree_embedding_size)

    def forward(self, g, return_all_outputs=False):
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

        if self.gnn_model == "gin":
            x, all_outputs = self.gnn(g, n_feat)
        if self.norm:
            x = F.normalize(x, p=2, dim=-1, eps=1e-5)
        if return_all_outputs:
            return x, all_outputs
        else:
            return x
