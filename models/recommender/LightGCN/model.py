import os
import scipy.sparse as sp
import dgl
import dgl.function as fn
import torch
from torch import nn
import random
import time
from scipy import sparse
import shutil
from collections import OrderedDict
from utils.ranking_metrics import *

from utils.data_load import Data as recData
from models.recommender.LightGCN import rec_utils as lightgcn_utils
from utils import utils


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, g, u_f, v_f):
        with g.local_scope():
            node_f = torch.cat([u_f, v_f], dim=0)
            degrees = g.out_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degrees, -0.5).view(-1, 1)

            node_f = node_f * norm

            g.ndata['n_f'] = node_f
            g.update_all(message_func=fn.copy_src(src='n_f', out='m'),
                         reduce_func=fn.sum(msg='m', out='n_f'))

            result = g.ndata['n_f']

            degrees = g.in_degrees().to(u_f.device).float().clamp(min=1)
            norm = torch.pow(degrees, -0.5).view(-1, 1)
            result = result * norm

            return result


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, hidden_dim, n_layers=1):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        initializer = nn.init.xavier_uniform_
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(n_users, hidden_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(n_items, hidden_dim))),
        })

        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(GCNLayer())

    def forward(self, graph):
        res_user_embedding = self.embedding_dict['user_emb']
        res_item_embedding = self.embedding_dict['item_emb']

        for i, layer in enumerate(self.layers):
            if i == 0:
                embeddings = layer(graph, res_user_embedding, res_item_embedding)
            else:
                embeddings = layer(graph, embeddings[: self.n_users], embeddings[self.n_users:])

            res_user_embedding = res_user_embedding + embeddings[: self.n_users] * (1 / (i + 2))
            res_item_embedding = res_item_embedding + embeddings[self.n_users:] * (1 / (i + 2))

        user_embedding = res_user_embedding

        item_embedding = res_item_embedding

        return user_embedding, item_embedding


class LightGCNTrainer:
    def __init__(self,
                 dataset='filmtrust',
                 target_item=5,
                 path_fake_data=None,
                 path_fake_matrix=None,
                 path_fake_array=None,
                 topk='1,5,10,20,50,100',
                 device=0,
                 batch_size=1024,
                 epochs=30,
                 hidden_dim=64,
                 n_layers=1,
                 reg=0.001,
                 decay=0.98,
                 lr=0.01,
                 minlr=0.0001,
                 ):
        self.dataset = dataset
        self.target_item = target_item
        self.path_train = './data/' + dataset + '/preprocess/train.data'
        self.path_test = './data/' + dataset + '/preprocess/test.data'
        self.path_fake_data = path_fake_data
        self.path_fake_array = path_fake_array
        self.path_fake_matrix = path_fake_matrix

        self.device = 'cuda:{}'.format(device) if device >= 0 and torch.cuda.is_available() else 'cpu'
        self.seed = 1234
        utils.set_seed(1234)

        self.topk = list(map(int, topk.split(',')))
        self.metrics = [PrecisionRecall(k=self.topk), NormalizedDCG(k=self.topk)]

        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lr = lr
        self.reg = reg
        self.decay = decay
        self.minlr = minlr

    def prepare_data(self):
        # 获得ori_n_users
        print("[load data in LightGCN] original dataset info")
        dataset_class = recData(self.path_train, self.path_test, test_bool=True, type='recommender', header=['user_id', 'item_id', 'rating', 'timestamp'], sep='\t')
        _, _, self.ori_n_users, self.ori_n_items = dataset_class.load_file_as_dataFrame()

        if self.path_fake_matrix or self.path_fake_array:
            self.injected_dir = './results/fake_data/'
            self.path_train = './data/' + self.dataset + '/preprocess' + '/train.data'

            self.generate_injectedFile(self.path_fake_array, self.path_fake_matrix)
            self.path_fake_data = self.injected_dir + self.dataset + '/%s_attacker_%d.data' % (self.dataset, self.target_item)

        if self.path_fake_data:
            self.path_train = self.path_fake_data

        dataset_class = recData(self.path_train, self.path_test, test_bool=True, type='recommender', header=['user_id', 'item_id', 'rating', 'timestamp'], sep='\t')

        train_df, test_df, self.n_users, self.n_items = dataset_class.load_file_as_dataFrame()
        _, self.train_matrix = dataset_class.dataFrame_to_matrix(train_df, self.n_users, self.n_items)
        _, self.test_matrix = dataset_class.dataFrame_to_matrix(test_df, self.n_users, self.n_items)

        u_i_adj = (self.train_matrix != 0) * 1
        i_u_adj = u_i_adj.T
        u_u_adj = sp.csr_matrix((self.n_users, self.n_users))
        i_i_adj = sp.csr_matrix((self.n_items, self.n_items))

        adj = sp.vstack([sp.hstack([u_u_adj, u_i_adj]), sp.hstack([i_u_adj, i_i_adj])]).tocsr()

        edge_src, edge_dst = adj.nonzero()
        self.graph = dgl.graph(data=(edge_src, edge_dst), idtype=torch.int32, num_nodes=adj.shape[0], device=self.device)

        train_u, train_v = self.train_matrix.nonzero()
        train_data = np.hstack((train_u.reshape(-1, 1), train_v.reshape(-1, 1))).tolist()
        train_dataset = lightgcn_utils.BPRData(train_data, self.train_matrix, True)
        self.train_loader = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def build_network(self):
        # model
        self.model = LightGCN(self.n_users, self.n_items, self.hidden_dim, self.n_layers).to(self.device)

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0)

    def adjust_learning_rate(self, opt):
        for param_group in opt.param_groups:
            param_group['lr'] = max(param_group['lr'] * self.decay, self.minlr)

    def generate_injectedFile(self, path_fake_array=None, path_fake_matrix=None):
        injected_path = self.injected_dir + self.dataset + '/%s_attacker_%d.data' % (self.dataset, self.target_item)
        if not os.path.exists(os.path.join(self.injected_dir, self.dataset)):
            os.makedirs(os.path.join(self.injected_dir, self.dataset))
        if os.path.exists(injected_path):
            os.remove(injected_path)
        shutil.copyfile(self.path_train, injected_path)

        if path_fake_matrix:
            fake_matrix = sparse.load_npz(path_fake_matrix)
            fake_array = fake_matrix.toarray()
        if path_fake_array:
            fake_array = np.load(path_fake_array)
        uids = np.where(fake_array > 0)[0] + self.ori_n_users
        iids = np.where(fake_array > 0)[1]
        values = fake_array[fake_array > 0]

        data_to_write = np.concatenate([np.expand_dims(x, 1) for x in [uids, iids, values]], 1)
        F_tuple_encode = lambda x: '\t'.join(map(str, [int(x[0]), int(x[1]), x[2]]))
        data_to_write = '\n'.join([F_tuple_encode(tuple_i) for tuple_i in data_to_write])
        with open(injected_path, 'a+')as fout:
            fout.write(data_to_write)
        return

    def recommend(self, model, g, data, top_k, return_preds=False, allow_repeat=False):
        # Set model to eval mode.
        model = model.to(self.device)
        model.eval()

        n_rows = data.shape[0]
        recommendations = np.empty([n_rows, top_k], dtype=np.int64)

        with torch.no_grad():
            user_embed, item_embed = model(g)
            preds = torch.matmul(user_embed, item_embed.t())

        if return_preds:
            all_preds = preds
        if not allow_repeat:
            preds[data.nonzero()] = -np.inf
        if top_k > 0:
            _, recs = preds.topk(k=top_k, dim=1)
            recommendations = recs.cpu().numpy()
        if return_preds:
            return recommendations, all_preds.cpu()
        else:
            return recommendations

    def evaluate(self, model, g, verbose=True):
        """Evaluate model performance on test data."""
        t1 = time.time()

        n_rows = self.train_matrix.shape[0]
        n_evaluate_users = self.test_matrix.shape[0]

        total_metrics_len = sum(len(x) for x in self.metrics)
        total_val_metrics = np.zeros([n_rows, total_metrics_len], dtype=np.float32)

        recommendations = self.recommend(model, g, self.train_matrix, top_k=100)

        valid_rows = list()
        for i in range(n_rows):
            if i >= n_evaluate_users:
                continue
            targets = self.test_matrix[i].indices
            if targets.size <= 0:
                continue

            recs = recommendations[i].tolist()

            metric_results = list()
            for metric in self.metrics:
                result = metric(targets, recs)
                metric_results.append(result)
            total_val_metrics[i, :] = np.concatenate(metric_results)
            valid_rows.append(i)

        total_val_metrics = total_val_metrics[valid_rows]
        avg_val_metrics = (total_val_metrics.mean(axis=0)).tolist()

        ind, result = 0, OrderedDict()
        for metric in self.metrics:
            values = avg_val_metrics[ind: ind + len(metric)]
            if len(values) <= 1:
                result[str(metric)] = round(values[0], 3)
            else:
                for name, value in zip(str(metric).split(','), values):
                    result[name] = round(value, 3)
            ind += len(metric)
        if verbose:
            result_pre, result_recall, result_ndcg = [], [], []
            for k, v in result.items():
                if 'Precision' in k:
                    result_pre.append(round(v, 2))
                elif 'Recall' in k:
                    result_recall.append(round(v, 2))
                elif 'NDCG' in k:
                    result_ndcg.append(round(v, 2))
            top_k_str = ', '.join(list(map(str, self.topk)))
            pre_str = ', '.join(list(map(str, result_pre)))
            recall_str = ', '.join(list(map(str, result_recall)))
            ndcg_str = ', '.join(list(map(str, result_ndcg)))
            print('[Evaluation recommender][{:.1f} s] topk=[{}]\nprecison=[{}], recall=[{}], ndcg=[{}]'.
                  format(time.time() - t1, top_k_str, pre_str, recall_str, ndcg_str))
        return result

    def validate(self, model, g):
        """Evaluate attack performance on target item."""
        t1 = time.time()

        n_rows = self.train_matrix.shape[0]
        n_evaluate_users = self.test_matrix.shape[0]

        target_item_position = np.zeros([n_rows], dtype=np.int64)
        recommendations = self.recommend(model, g, self.train_matrix, top_k=100)

        valid_rows = list()
        target_item_num = 0
        for i in range(n_rows):
            if i >= n_evaluate_users:
                continue
            targets = self.test_matrix[i].indices
            if targets.size <= 0:
                continue
            if self.target_item in self.train_matrix[i].indices:
                continue

            recs = recommendations[i].tolist()

            if self.target_item in recs:
                target_item_position[i] = recs.index(self.target_item)
                target_item_num += 1
            else:
                target_item_position[i] = -1

            valid_rows.append(i)

        target_item_position = target_item_position[valid_rows]
        target_item_position = target_item_position[target_item_position >= 0]

        result = OrderedDict()
        result['HitUserNum'] = target_item_num
        result['TargetAvgRank'] = round(target_item_position.mean(), 1)

        for cutoff in self.topk:
            result['TargetHR@%d_' % cutoff] = round((target_item_position < cutoff).sum() / len(valid_rows), 3)

            _target_item_position = target_item_position[target_item_position < cutoff]
            result['TargetNDCG@%d_' % cutoff] = round(
                (np.log(2) / np.log(np.array(_target_item_position) + 2)).sum() / len(valid_rows), 3)

        hr_list, ndcg_list = [], []
        for k, v in result.items():
            if 'TargetHR' in k:
                hr_list.append(v)
            if 'TargetNDCG' in k:
                ndcg_list.append(v)
        top_k_str = ', '.join(list(map(str, self.topk)))
        hr_str = ', '.join(list(map(str, hr_list)))
        ndcg_str = ', '.join(list(map(str, ndcg_list)))
        print(
            '[Evaluation recommender after attack][{:.1f} s] topk=[{}]\nHitUserNum=[{}], TargetAvgRank=[{}], TargetHR=[{}], TargetNDCG=[{}]'.
            format(time.time() - t1, top_k_str, result['HitUserNum'], result['TargetAvgRank'], hr_str, ndcg_str))
        return result

    def fit(self):
        # prepare data
        self.prepare_data()

        # build network
        self.build_network()

        for epoch in range(self.epochs):
            epoch_loss, epoch_bprloss, epoch_regloss = 0, 0, 0
            for user, item_pos, item_neg in self.train_loader:
                user = user.long().to(self.device)
                item_pos = item_pos.long().to(self.device)
                item_neg = item_neg.long().to(self.device)

                user_embed, item_embed = self.model(self.graph)

                batch_user_embed = user_embed[user]
                batch_item_pos_embed = item_embed[item_pos]
                batch_item_neg_embed = item_embed[item_neg]

                pred_pos = torch.sum(torch.mul(batch_user_embed, batch_item_pos_embed), dim=1)
                pred_neg = torch.sum(torch.mul(batch_user_embed, batch_item_neg_embed), dim=1)

                bprloss = - (pred_pos.view(-1) - pred_neg.view(-1)).sigmoid().log().sum()
                regloss = (torch.norm(batch_user_embed) ** 2 + torch.norm(batch_item_pos_embed) ** 2 + torch.norm(batch_item_neg_embed) ** 2)

                loss = 0.5 * (bprloss + self.reg * regloss) / self.batch_size
                epoch_bprloss += bprloss.item() / self.batch_size
                epoch_regloss += self.reg * regloss.item() / self.batch_size
                epoch_loss += loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print('Epoch %d : train==[%.3f=(%.3f + %.3f)/2]' % (
                epoch, epoch_loss, epoch_bprloss, epoch_regloss))

            self.adjust_learning_rate(self.optimizer)

        torch.save(self.model.state_dict(), './saved/recommender/' + 'LightGCN.pkl')
        results1 = self.evaluate(self.model, self.graph)
        results2 = self.validate(self.model, self.graph)
        return results1, results2




