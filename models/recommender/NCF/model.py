import os
import time
import torch.backends.cudnn as cudnn
from torch import nn
import torch.utils.data
import numpy as np
from collections import OrderedDict
import shutil
from scipy import sparse

from models.recommender.NCF import rec_utils as ncf_utils
from utils.ranking_metrics import *
from utils import utils


class NCF(nn.Module):
    def __init__(self, n_users, n_items, n_factor, num_layers, dropout,
                 model_type, GMF_model=None, MLP_model=None):
        super(NCF, self).__init__()
        self.dropout = dropout
        self.model_type = model_type
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model

        self.embed_user_GMF = nn.Embedding(n_users, n_factor)
        self.embed_item_GMF = nn.Embedding(n_items, n_factor)
        self.embed_user_MLP = nn.Embedding(
            n_users, n_factor * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(
            n_items, n_factor * (2 ** (num_layers - 1)))

        MLP_modules = []
        for i in range(num_layers):
            input_size = n_factor * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model_type in ['MLP', 'GMF']:
            predict_size = n_factor
        else:
            predict_size = n_factor * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()

    def _init_weight_(self):
        if not self.model_type == 'NeuMF-pre':
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(
                self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(
                self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(
                self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(
                self.MLP_model.embed_item_MLP.weight)

            # mlp layers
            for (m1, m2) in zip(
                    self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            # predict layers
            predict_weight = torch.cat([
                self.GMF_model.predict_layer.weight,
                self.MLP_model.predict_layer.weight], dim=1)
            precit_bias = self.GMF_model.predict_layer.bias + \
                          self.MLP_model.predict_layer.bias

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * precit_bias)

    def forward(self, user, item):
        if not self.model_type == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model_type == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
            output_MLP = self.MLP_layers(interaction)

        if self.model_type == 'GMF':
            concat = output_GMF
        elif self.model_type == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)


class NCFTrainer:
    def __init__(self,
                 dataset='filmtrust',
                 target_item=5,
                 path_fake_data=None,
                 path_fake_matrix=None,
                 path_fake_array=None,
                 model_type='NeuMF-end',  # ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']
                 epochs=10,
                 batch_size=1024,
                 n_factor=32,
                 num_layers=3,
                 num_ng=4,
                 lr=0.001,
                 dropout=0.0,
                 topk='[1, 5, 10, 20, 50, 100]',
                 device=0,
                 ):
        self.dataset = dataset
        self.target_item = target_item
        self.path_fake_data = path_fake_data
        self.path_fake_matrix = path_fake_matrix
        self.path_fake_array = path_fake_array

        self.model_type = model_type
        self.GMF_model_path = None
        self.MLP_model_path = None
        self.lr = lr
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.topk = eval(topk)
        self.metrics = [PrecisionRecall(k=self.topk), NormalizedDCG(k=self.topk)]

        self.n_factor = n_factor  # predictive factors numbers in the model
        self.num_layers = num_layers  # number of layers in MLP model
        self.num_ng = num_ng  # sample negative items for training
        self.device = device
        self.device = 'cuda:{}'.format(device) if device >= 0 and torch.cuda.is_available() else 'cpu'
        utils.set_seed(1234)

        # os.environ["CUDA_VISIBLE_DEVICES"] = self.device
        # cudnn.benchmark = True

    def prepare_data(self):
        # 获得ori_n_users
        print("[load data in NCF] original dataset info")
        self.ori_n_users, self.ori_n_items,_, _, _, _ = ncf_utils.load_data(self.dataset, self.path_fake_data)

        if self.path_fake_matrix or self.path_fake_array:
            self.injected_dir = './results/fake_data/'
            self.path_train = './data/' + self.dataset + '/preprocess' + '/train.data'

            self.generate_injectedFile(self.path_fake_array, self.path_fake_matrix)
            self.path_fake_data = self.injected_dir + self.dataset + '/%s_attacker_%d.data' % (self.dataset, self.target_item)

        print("[load data in NCF] dataset info after injected fake data")
        self.n_users, self.n_items, self.train_matrix, self.test_matrix, train_list, test_list = ncf_utils.load_data(self.dataset, self.path_fake_data)
        # construct the train and test datasets.
        train_dataset = ncf_utils.NCFData(train_list, self.train_matrix, self.num_ng, True)
        # test_dataset = ncf_utils.NCFData(test_list, None, 0, False)
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,)
        #self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False,)

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

    def build_network(self):
        if self.model_type == 'NeuMF-pre':
            assert os.path.exists(self.GMF_model_path), 'lack of GMF model'
            assert os.path.exists(self.MLP_model_path), 'lack of MLP model'
            GMF_model = torch.load(self.GMF_model_path)
            MLP_model = torch.load(self.MLP_model_path)
            # GMF_model: pre-trained GMF weights;
            # MLP_model: pre-trained MLP weights.
        else:
            GMF_model = None
            MLP_model = None

        self.model = NCF(self.n_users, self.n_items, self.n_factor, self.num_layers,
                         self.dropout, self.model_type, GMF_model, MLP_model)
        self.model.to(self.device)

        self.loss_function = nn.BCEWithLogitsLoss()

        if self.model_type == 'NueMF-pre':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def recommend(self, model, data, top_k=50, return_preds=False, allow_repeat=False):
        model = model.to(self.device)
        model.eval()

        idx_list = np.arange(self.n_users)
        recommendations = np.empty([self.n_users, top_k], dtype=np.int64)

        # Make predictions first, and then sort for top-k.
        all_preds = list()
        with torch.no_grad():
            for user_id in range(self.n_users):
                batch_idx = torch.LongTensor([user_id]).repeat(self.n_items)  # 把[user_id]复制n份，size=(self.n_users)
                preds = model(batch_idx.to(self.device), torch.LongTensor(list(range(self.n_items))).to(self.device))
                preds = preds.view(1, -1)
                all_preds.append(preds)
        all_preds = torch.cat(all_preds, dim=0)
        preds = all_preds
        if not allow_repeat:
            preds[data.nonzero()] = -np.inf
        if top_k > 0:
            _, recs = preds.topk(k=top_k, dim=1)
            recommendations = recs.cpu().numpy()
        if return_preds:
            return recommendations, all_preds.cpu()
        else:
            return recommendations

    def evaluate(self, model, verbose=True):
        t1 = time.time()

        n_rows = self.train_matrix.shape[0]
        n_evaluate_users = self.test_matrix.shape[0]

        total_metrics_len = sum(len(x) for x in self.metrics)
        total_val_metrics = np.zeros([n_rows, total_metrics_len], dtype=np.float32)

        recommendations = self.recommend(model, self.train_matrix, top_k=100, return_preds=False, allow_repeat=False)

        valid_rows = list()
        for i in range(n_rows):
            # Ignore augmented users, evaluate only on real users.
            if i >= n_evaluate_users:
                continue
            targets = self.test_matrix.tocsr()[i].indices
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
            top_k_str = ', '.join(list(map(str, self.topk)))
            pre_str = ', '.join(list(map(str, result_pre)))
            recall_str = ', '.join(list(map(str, result_recall)))
            ndcg_str = ', '.join(list(map(str, result_ndcg)))
            print('[Evaluation recommender][{:.1f} s] topk=[{}]\nprecison=[{}], recall=[{}], ndcg=[{}]'.
                  format(time.time() - t1, top_k_str, pre_str, recall_str, ndcg_str))
        return result

    def validate(self, model):
        t1 = time.time()

        n_rows = self.train_matrix.shape[0]
        n_evaluate_users = self.test_matrix.shape[0]

        # Init evaluation results.
        target_item_position = np.zeros([n_rows], dtype=np.int64)
        recommendations = self.recommend(model, self.train_matrix, top_k=100, return_preds=False, allow_repeat=False)

        valid_rows = list()
        target_item_num = 0
        for i in range(n_rows):
            # Ignore augmented users, evaluate only on real users.
            if i >= n_evaluate_users:
                continue
            targets = self.test_matrix.tocsr()[i].indices
            if targets.size <= 0:
                continue
            if self.target_item in self.train_matrix.tocsr()[i].indices:
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
        self.prepare_data()
        self.build_network()
        for epoch in range(self.epochs):
            self.model.train()  # enable dropout(if have).
            start_time = time.time()
            self.train_loader.dataset.ng_sample()
            loss_total = 0.0
            for user, item, label in self.train_loader:
                user = user.long().to(self.device)
                item = item.long().to(self.device)
                label = label.float().to(self.device)

                self.model.zero_grad()
                prediction = self.model(user, item)
                loss = self.loss_function(prediction, label)
                loss_total += loss
                loss.backward()
                self.optimizer.step()
            loss_total /= len(self.train_loader)
            print('[NCF] [%.1fs] Epoch %d : %.3f' % (time.time() - start_time, epoch, loss_total))

        results1 = self.evaluate(self.model)
        results2 = self.validate(self.model)
        return results1, results2
