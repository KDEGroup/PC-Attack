import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dgl.function as fn
import numpy as np
# import multiprocessing
import heapq
import time
import shutil
from collections import OrderedDict
from scipy import sparse

import os
from models.recommender.NGCF import rec_utils
from utils.ranking_metrics import *
from utils import utils


class NGCFLayer(nn.Module):
    def __init__(self, in_size, out_size, norm_dict, dropout):
        super(NGCFLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.W1 = nn.Linear(in_size, out_size, bias = True)
        self.W2 = nn.Linear(in_size, out_size, bias = True)

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.dropout = nn.Dropout(dropout)

        torch.nn.init.xavier_uniform_(self.W1.weight)
        torch.nn.init.constant_(self.W1.bias, 0)
        torch.nn.init.xavier_uniform_(self.W2.weight)
        torch.nn.init.constant_(self.W2.bias, 0)

        self.norm_dict = norm_dict

    def forward(self, g, feat_dict):

        funcs = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            if srctype == dsttype:
                messages = self.W1(feat_dict[srctype])
                g.nodes[srctype].data[etype] = messages
                funcs[(srctype, etype, dsttype)] = (fn.copy_u(etype, 'm'), fn.sum('m', 'h'))
            else:
                src, dst = g.edges(etype=(srctype, etype, dsttype))
                norm = self.norm_dict[(srctype, etype, dsttype)]
                messages = norm * (self.W1(feat_dict[srctype][src]) + self.W2(feat_dict[srctype][src]*feat_dict[dsttype][dst])) #compute messages
                g.edges[(srctype, etype, dsttype)].data[etype] = messages
                funcs[(srctype, etype, dsttype)] = (fn.copy_e(etype, 'm'), fn.sum('m', 'h'))

        g.multi_update_all(funcs, 'sum')
        feature_dict={}
        for ntype in g.ntypes:
            h = self.leaky_relu(g.nodes[ntype].data['h'])
            h = self.dropout(h)
            h = F.normalize(h, dim=1, p=2)
            feature_dict[ntype] = h
        return feature_dict


class NGCF(nn.Module):
    def __init__(self, g, in_size, layer_size, dropout, lmbd=1e-5):
        super(NGCF, self).__init__()
        self.lmbd = lmbd
        self.norm_dict = dict()
        for srctype, etype, dsttype in g.canonical_etypes:
            src, dst = g.edges(etype=(srctype, etype, dsttype))
            dst_degree = g.in_degrees(dst, etype=(srctype, etype, dsttype)).float()
            src_degree = g.out_degrees(src, etype=(srctype, etype, dsttype)).float()
            norm = torch.pow(src_degree * dst_degree, -0.5).unsqueeze(1)
            self.norm_dict[(srctype, etype, dsttype)] = norm

        self.layers = nn.ModuleList()
        self.layers.append(
            NGCFLayer(in_size, layer_size[0], self.norm_dict, dropout[0])
        )
        self.num_layers = len(layer_size)
        for i in range(self.num_layers-1):
            self.layers.append(
                NGCFLayer(layer_size[i], layer_size[i+1], self.norm_dict, dropout[i+1])
            )
        self.initializer = nn.init.xavier_uniform_

        self.feature_dict = nn.ParameterDict({
            ntype: nn.Parameter(self.initializer(torch.empty(g.num_nodes(ntype), in_size))) for ntype in g.ntypes
        })

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = (users * pos_items).sum(1)
        neg_scores = (users * neg_items).sum(1)

        mf_loss = nn.LogSigmoid()(pos_scores - neg_scores).mean()
        mf_loss = -1 * mf_loss

        regularizer = (torch.norm(users) ** 2 + torch.norm(pos_items) ** 2 + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.lmbd * regularizer / users.shape[0]

        return mf_loss + emb_loss, mf_loss, emb_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def forward(self, g, user_key, item_key, users, pos_items, neg_items):
        h_dict = {ntype : self.feature_dict[ntype] for ntype in g.ntypes}
        user_embeds = []
        item_embeds = []
        user_embeds.append(h_dict[user_key])
        item_embeds.append(h_dict[item_key])
        for layer in self.layers:
            h_dict = layer(g, h_dict)
            user_embeds.append(h_dict[user_key])
            item_embeds.append(h_dict[item_key])
        user_embd = torch.cat(user_embeds, 1)
        item_embd = torch.cat(item_embeds, 1)

        u_g_embeddings = user_embd[users, :]
        pos_i_g_embeddings = item_embd[pos_items, :]
        neg_i_g_embeddings = item_embd[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings


class NGCFTrainer:
    def __init__(self,
                 dataset='filmtrust',
                 target_item=5,
                 path_fake_data=None,
                 path_fake_matrix=None,
                 path_fake_array=None,
                 epoch=10,
                 batch_size=1024,
                 embed_size=64,
                 layer_size='[64,64,64]',
                 lr=0.0001,
                 regs='[1e-5]',
                 mess_dropout='[0.1,0.1,0.1]',
                 Ks='[1, 5, 10, 20, 50, 100]',
                 test_flag='part',
                 device=0,
                 verbose=1,
                 ):
        self.model_path = './saved/recommender/'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.Ks = eval(Ks)
        self.metrics = [PrecisionRecall(k=self.Ks), NormalizedDCG(k=self.Ks)]
        self.mess_dropout = eval(mess_dropout)
        self.layer_size = eval(layer_size)
        self.regs = eval(regs)

        self.dataset = dataset
        self.target_item = target_item
        self.batch_size = batch_size
        self.epoch = epoch
        self.embed_size = embed_size
        self.lr = lr
        self.test_flag = test_flag
        self.verbose = verbose

        print("[load data in NGCF] original dataset info")
        self.data_generator = rec_utils.Data(path=path_fake_data, dataset=dataset, batch_size=batch_size)
        self.ori_n_users, _ = self.data_generator.n_users, self.data_generator.n_items

        if path_fake_matrix or path_fake_array:
            self.injected_dir = './results/fake_data/'
            self.path_train = './data/' + dataset + '/preprocess' + '/train.data'

            self.generate_injectedFile(path_fake_array, path_fake_matrix)
            path_fake_data = self.injected_dir + self.dataset + '/%s_attacker_%d.data' % (self.dataset, self.target_item)

        print("[load data in NGCF] dataset info after injected fake data")
        self.data_generator = rec_utils.Data(path=path_fake_data, dataset=dataset, batch_size=batch_size)
        self.train_matrix, self.test_matrix = self.data_generator.train_matrix, self.data_generator.test_matrix
        self.n_users, self.n_items = self.data_generator.n_users, self.data_generator.n_items
        self.N_TRAIN, self.N_TEST = self.data_generator.n_train, self.data_generator.n_test

        if device >= 0 and torch.cuda.is_available():
            self.device = 'cuda:{}'.format(device)
        else:
            self.device = 'cpu'
        utils.set_seed(1234)


    def ranklist_by_heapq(self, user_pos_test, test_items, rating, Ks):
        item_score = {}
        for i in test_items:
            item_score[i] = rating[i]

        K_max = max(Ks)
        K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

        r = []
        for i in K_max_item_score:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = 0.
        return r, auc

    def get_auc(self, item_score, user_pos_test):
        item_score = sorted(item_score.items(), key=lambda kv: kv[1])
        item_score.reverse()
        item_sort = [x[0] for x in item_score]
        posterior = [x[1] for x in item_score]

        r = []
        for i in item_sort:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = rec_utils.auc(ground_truth=r, prediction=posterior)
        return auc

    def ranklist_by_sorted(self, user_pos_test, test_items, rating, Ks):
        item_score = {}
        for i in test_items:
            item_score[i] = rating[i]

        K_max = max(Ks)
        K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

        r = []
        for i in K_max_item_score:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = self.get_auc(item_score, user_pos_test)

        return r, auc

    def recommend(self, model, g, data, top_k, return_preds=False, allow_repeat=False):
        # Set model to eval mode.
        model = model.to(self.device)
        model.eval()

        n_rows = data.shape[0]
        recommendations = np.empty([n_rows, top_k], dtype=np.int64)

        # Make predictions first, and then sort for top-k.
        with torch.no_grad():
            u_g_embeddings, pos_i_g_embeddings, _ = model(g, 'user', 'item', range(self.n_users), range(self.n_items), [])
            preds = model.rating(u_g_embeddings, pos_i_g_embeddings)

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

        # TODO:topk?
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

        # Average evaluation results by user.
        total_val_metrics = total_val_metrics[valid_rows]
        avg_val_metrics = (total_val_metrics.mean(axis=0)).tolist()

        # Summary evaluation results into a dict.
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
            top_k_str = ', '.join(list(map(str, self.Ks)))
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

        # Init evaluation results.
        target_item_position = np.zeros([n_rows], dtype=np.int64)
        recommendations = self.recommend(model, g, self.train_matrix, top_k=100)

        valid_rows = list()
        target_item_num = 0
        for i in range(n_rows):
            # Ignore augmented users, evaluate only on real users.
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
                target_item_position[i] = -1  # self.train_matrix.shape[1]

            valid_rows.append(i)

        target_item_position = target_item_position[valid_rows]
        target_item_position = target_item_position[target_item_position >= 0]

        # Summary evaluation results into a dict.
        result = OrderedDict()
        result['HitUserNum'] = target_item_num
        result['TargetAvgRank'] = round(target_item_position.mean(), 1)

        for cutoff in self.Ks:
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
        top_k_str = ', '.join(list(map(str, self.Ks)))
        hr_str = ', '.join(list(map(str, hr_list)))
        ndcg_str = ', '.join(list(map(str, ndcg_list)))
        print(
            '[Evaluation recommender after attack][{:.1f} s] topk=[{}]\nHitUserNum=[{}], TargetAvgRank=[{}], TargetHR=[{}], TargetNDCG=[{}]'.
            format(time.time() - t1, top_k_str, result['HitUserNum'], result['TargetAvgRank'], hr_str, ndcg_str))
        return result

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

    def get_performance(self, user_pos_test, r, auc, Ks, test_items, rating):
        precision, recall, ndcg, hit_ratio = [], [], [], []

        for K in Ks:
            precision.append(rec_utils.precision_at_k(r, K))
            recall.append(rec_utils.recall_at_k(r, K, len(user_pos_test)))
            ndcg.append(rec_utils.ndcg_at_k(r, K))
            # hit_ratio.append(utils.hit_at_k(r, K))
            hit_ratio.append(rec_utils.hit_at_k(self.target_item, test_items, rating, K))

        return {'recall': np.array(recall), 'precision': np.array(precision),
                'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}

    def test_one_user(self, x):
        rating = x[0]  # user u's ratings for user
        u = x[1]  # uid
        try:
            training_items = self.data_generator.train_items[u]  # user u's items in the training set
        except Exception:
            training_items = []
        user_pos_test = self.data_generator.test_set[u]  # user u's items in the test set

        all_items = set(range(self.n_items))

        test_items = list(all_items - set(training_items))

        if self.test_flag == 'part':
            r, auc = self.ranklist_by_heapq(user_pos_test, test_items, rating, self.Ks)
        else:
            r, auc = self.ranklist_by_sorted(user_pos_test, test_items, rating, self.Ks)

        return self.get_performance(user_pos_test, r, auc, self.Ks, test_items, rating)

    def test(self, model, g, users_to_test, batch_test_flag=False):
        result = {'precision': np.zeros(len(self.Ks)), 'recall': np.zeros(len(self.Ks)),
                  'ndcg': np.zeros(len(self.Ks)), 'hit_ratio': np.zeros(len(self.Ks)), 'auc': 0.}

        # pool = multiprocessing.Pool(cores)

        u_batch_size = 5000
        i_batch_size = self.batch_size

        test_users = users_to_test
        n_test_users = len(test_users)
        n_user_batchs = n_test_users // u_batch_size + 1

        count = 0

        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size

            user_batch = test_users[start: end]

            if batch_test_flag:
                # batch-item test
                n_item_batchs = self.n_items // i_batch_size + 1
                rate_batch = np.zeros(shape=(len(user_batch), self.n_items))

                i_count = 0
                for i_batch_id in range(n_item_batchs):
                    i_start = i_batch_id * i_batch_size
                    i_end = min((i_batch_id + 1) * i_batch_size, self.n_items)

                    item_batch = range(i_start, i_end)

                    u_g_embeddings, pos_i_g_embeddings, _ = model(g, 'user', 'item', user_batch, item_batch, [])
                    i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

                    rate_batch[:, i_start: i_end] = i_rate_batch
                    i_count += i_rate_batch.shape[1]

                assert i_count == self.n_items

            else:
                # all-item test
                item_batch = range(self.n_items)
                u_g_embeddings, pos_i_g_embeddings, _ = model(g, 'user', 'item', user_batch, item_batch, [])
                rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

            user_batch_rating_uid = zip(rate_batch.numpy(), user_batch)
            # batch_result = pool.map(test_one_user, user_batch_rating_uid)
            batch_result = []
            for idx1, idx2 in user_batch_rating_uid:
                batch_result.append(self.test_one_user((idx1, idx2)))

            count += len(batch_result)

            for re in batch_result:
                result['precision'] += re['precision'] / n_test_users
                result['recall'] += re['recall'] / n_test_users
                result['ndcg'] += re['ndcg'] / n_test_users
                result['hit_ratio'] += re['hit_ratio'] / n_test_users
                result['auc'] += re['auc'] / n_test_users

        assert count == n_test_users
        # pool.close()
        return result

    def fit(self):
        # Step 1: Prepare graph data and device ================================================================= #
        g = self.data_generator.g
        g = g.to(self.device)

        # Step 2: Create model and training components=========================================================== #
        model = NGCF(g, self.embed_size, self.layer_size, self.mess_dropout, self.regs[0]).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # Step 3: training epoches ============================================================================== #
        n_batch = self.data_generator.n_train // self.batch_size + 1
        t0 = time.time()
        cur_best_pre_0, stopping_step = 0, 0
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
        for epoch in range(self.epoch):
            t1 = time.time()
            loss, mf_loss, emb_loss = 0., 0., 0.
            for idx in range(n_batch):
                users, pos_items, neg_items = self.data_generator.sample()
                u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(g, 'user', 'item', users,
                                                                               pos_items,
                                                                               neg_items)

                batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                                  pos_i_g_embeddings,
                                                                                  neg_i_g_embeddings)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss
                mf_loss += batch_mf_loss
                emb_loss += batch_emb_loss

            if (epoch + 1) % 10 != 0:
                if self.verbose > 0 and epoch % self.verbose == 0:
                    perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                        epoch, time.time() - t1, loss, mf_loss, emb_loss)
                    print(perf_str)
                continue  # end the current epoch and move to the next epoch, let the following evaluation run every 10 epoches

            # evaluate the model every 10 epoches
            t2 = time.time()
            users_to_test = list(self.data_generator.test_set.keys())
            ret = self.test(model, g, users_to_test)
            t3 = time.time()

            loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])

            if self.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                           'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                           (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1])
                print(perf_str)

            cur_best_pre_0, stopping_step, should_stop = rec_utils.early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                                  stopping_step, expected_order='acc',
                                                                                  flag_step=5)
            # early stop
            if should_stop == True:
                break

            if ret['recall'][0] == cur_best_pre_0:
                torch.save(model.state_dict(), self.model_path + 'NGCF.pkl')
                # print('save the weights in path: ', self.model_path + 'NGCF.pkl')

        recs = np.array(rec_loger)
        pres = np.array(pre_loger)
        ndcgs = np.array(ndcg_loger)
        hit = np.array(hit_loger)

        results1 = self.evaluate(model, g)
        results2 = self.validate(model, g)
        return results1, results2


