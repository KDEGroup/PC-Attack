import os
import time
from functools import partial
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import utils
from utils.data_load import Data
from utils.ranking_metrics import *
from utils.criterions import *
from models.recommender.recommender import Recommender, build_result_graph


class ItemAE(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(ItemAE, self).__init__()
        self.q_dims = [input_dim] + hidden_dims
        self.p_dims = self.q_dims[::-1]

        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out
                                       in zip(self.q_dims[:-1], self.q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out
                                       in zip(self.p_dims[:-1], self.p_dims[1:])])

        self.recon_loss = partial(mse_loss, weight=1)

    def encode(self, input):
        h = input
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            h = torch.tanh(h)
        return h

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def forward(self, input):
        z = self.encode(input)
        return self.decode(z)

    def loss(self, data, outputs):
        return self.recon_loss(data, outputs)


class ItemAETrainer(Recommender):
    def __init__(self, hidden_dims='[256, 128]', num_epoch=50, batch_size=128, lr=0.001, *args, **kwargs):
        super(ItemAETrainer, self).__init__(*args, **kwargs)

        self.hidden_dims = eval(hidden_dims)
        self.num_epoch = num_epoch
        self.save_feq = num_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.l2 = 1e-6

    def build_network(self):
        self.net = ItemAE(self.n_users, self.hidden_dims).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.l2)

    def train(self):
        best_perf = 0.0
        # Transpose the data first for ItemVAE
        data = self.train_matrix.transpose()

        n_rows = data.shape[0]
        n_cols = data.shape[1]

        # Set model to training mode
        model = self.net.to(self.device)
        model.train()
        for epoch in range(1, self.num_epoch + 1):
            time_st = time.time()

            idx_list = np.arange(n_rows)
            np.random.shuffle(idx_list)

            epoch_loss = 0.0
            # batch_size = (self.batch_size if self.batch_size > 0 else len(idx_list))

            for batch_idx in utils.minibatch(idx_list, batch_size=self.batch_size):
                batch_tensor = utils.sparse2tensor(data[batch_idx]).to(self.device)

                # Compute loss
                outputs = model(batch_tensor)
                loss = model.loss(data=batch_tensor, outputs=outputs).sum()
                epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print("[TRIAN recommender ItemAE] [{:.1f} s], epoch: {}, loss: {:.4f}".format(time.time() - time_st, epoch, epoch_loss))

            if epoch % self.save_feq == 0:
                result = self.evaluate(verbose=False)
                # Save model checkpoint if it has better performance.
                if result[self.golden_metric] > best_perf:
                    str_metric = "{}={:.4f}".format(self.golden_metric, result[self.golden_metric])
                    # print("Having better model checkpoint with performance {}(epoch:{})".format(str_metric, epoch))
                    self.path_save = os.path.join(self.dir_model, 'ItemAE')
                    utils.save_checkpoint(self.path_save, self.net, self.optimizer, epoch)
                    best_perf = result[self.golden_metric]

    def recommend(self, data, top_k, return_preds=False, allow_repeat=False):
        # Set model to eval mode.
        model = self.net.to(self.device)
        model.eval()

        # Transpose the data first for ItemVAE.
        data = data.transpose()

        n_rows = data.shape[0]
        n_cols = data.shape[1]

        recommendations = np.empty([n_cols, top_k], dtype=np.int64)

        # Make predictions first, and then sort for top-k.

        with torch.no_grad():
            data_tensor = utils.sparse2tensor(data).to(self.device)
            preds = model(data_tensor)

        preds = preds.t()
        data = data.transpose()
        data_array = data.toarray()

        if return_preds:
            all_preds = preds
        if not allow_repeat:
            preds[data_array.nonzero()] = -np.inf
        if top_k > 0:
            _, recs = preds.topk(k=top_k, dim=1)
            recommendations = recs.cpu().numpy()
        if return_preds:
            return recommendations, all_preds.cpu()
        else:
            return recommendations