import math
import torch
from torch import nn


class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, inputSize=64, K=256, T=0.07, use_softmax=True):
        super(MemoryMoCo, self).__init__()
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax

        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.register_buffer("memory_graph", torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer("memory_seq", torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, graph_q, graph_k, seq_q, seq_k):

        batchSize = graph_q.shape[0]
        graph_k = graph_k.detach()
        seq_k = seq_k.detach()

        l_pos_g = torch.bmm(graph_q.view(batchSize, 1, -1), seq_k.view(batchSize, -1, 1))
        l_pos_g = l_pos_g.view(batchSize, 1)

        queue_seq = self.memory_seq.clone()
        l_neg_g = torch.mm(queue_seq.detach(), graph_q.transpose(1, 0))
        l_neg_g = l_neg_g.transpose(0, 1)

        out_g = torch.cat((l_pos_g, l_neg_g), dim=1)

        if self.use_softmax:
            out_g = torch.div(out_g, self.T)
            out_g = out_g.squeeze().contiguous()

        l_pos_s = torch.bmm(seq_q.view(batchSize, 1, -1), graph_k.view(batchSize, -1, 1))
        l_pos_s = l_pos_s.view(batchSize, 1)

        queue_graph = self.memory_graph.clone()
        l_neg_s = torch.mm(queue_graph.detach(), seq_q.transpose(1, 0))
        l_neg_s = l_neg_s.transpose(0, 1)

        out_s = torch.cat((l_pos_s, l_neg_s), dim=1)

        if self.use_softmax:
            out_s = torch.div(out_s, self.T)
            out_s = out_s.squeeze().contiguous()

        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory_graph.index_copy_(0, out_ids, graph_k)
            self.memory_seq.index_copy_(0, out_ids, seq_k)
            self.index = (self.index + batchSize) % self.queueSize

        return out_g, out_s
