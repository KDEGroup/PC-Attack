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


class ItemCF(nn.Module):
    def __init__(self, n_users, n_items, knn=50):
        super(ItemCF, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.knn = knn

        self.sims = nn.Parameter(
            torch.zeros([self.n_items, self.n_items]),
            requires_grad=False)

        self.top_nns = None
        self.top_sims = None

    def forward(self, item_id):
        if self.top_nns is None:
            self.top_sims, self.top_nns = self.sims.topk(k=self.knn, dim=1)
        return self.top_sims[item_id], self.top_nns[item_id]


class ItemCFTrainer(Recommender):
    def __init__(self, num_epoch=1, knn=50, *args, **kwargs):
        super(ItemCFTrainer, self).__init__(*args, **kwargs)

        self.num_epoch = num_epoch
        self.knn = knn

    def build_network(self):
        self.net = ItemCF(self.n_users, self.n_items, self.knn)

    def train(self):
        for epoch in range(self.num_epoch):
            self.net.sims.fill_(0.0)
            self.net.top_nns = None
            self.net.top_sims = None

            data_t = self.train_matrix.transpose()
            self.net.sims.data = torch.FloatTensor(utils._pairwise_jaccard(data_t))

            # Save model checkpoint
            self.path_save = os.path.join(self.dir_model, 'ItemCF')
            utils.save_checkpoint(self.path_save, self.net, None, 1)

    def fit_withPQ(self, P, Q):
        # get self.train_matrix(real+fake) and self.test_matrix
        self.prepare_data()

        self.build_network()
        # self.train()
        self.net.sims.fill_(0.0)
        self.net.top_nns = None
        self.net.top_sims = None

        self.net.sims.data = utils.matrix_cos_similar(torch.FloatTensor(Q))

        # Save model checkpoint
        self.path_save = os.path.join(self.dir_model, 'ItemCF')
        utils.save_checkpoint(self.path_save, self.net, None, 1)
        # Load best model and evaluate on test data
        print("Loading best model checkpoint.")
        self.restore(self.path_save)

        # Evaluate model performance on test data.
        self.evaluate()

        # Evaluate attack performance on target item.
        self.validate()

    def recommend(self, data, top_k, return_preds=False, allow_repeat=False):
        model = self.net
        n_rows = data.shape[0]
        n_cols = data.shape[1]

        nns_sims = torch.zeros([n_cols, n_cols])
        for item in range(n_cols):
            topk_sims, topk_nns = model(item_id=item)
            nns_sims[item].put_(topk_nns, topk_sims)

        recommendations = np.empty([n_rows, top_k], dtype=np.int64)
        all_preds = list()
        # with torch.no_grad():
        # for batch_idx in utils.minibatch(idx_list, batch_size=self.batch_size):
        data_tensor = utils.sparse2tensor(data)
        preds = torch.mm(data_tensor, nns_sims)
        if return_preds:
            all_preds = preds
        if not allow_repeat:
            preds[data.nonzero()] = -np.inf
        if top_k > 0:
            _, recs = preds.topk(k=top_k, dim=1)
            recommendations = recs.cpu().numpy()
        if return_preds:
            return recommendations, all_preds
        else:
            return recommendations