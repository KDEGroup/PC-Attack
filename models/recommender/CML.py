import os
import time
from functools import partial
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import random
import matplotlib.pyplot as plt

from utils import utils
from utils.data_load import Data
from utils.ranking_metrics import *
from utils.criterions import *
from models.recommender.recommender import Recommender, build_result_graph


class CML(nn.Module):
    def __init__(self, n_users, n_items, num_factors, loss_func):
        super(CML, self).__init__()

        self.n_factor = num_factors

        self.n_users = n_users
        self.n_items = n_items

        self.emb_user = nn.Embedding(self.n_users, self.n_factor)
        self.emb_item = nn.Embedding(self.n_items, self.n_factor)

        self.l2 = 1e-4
        self.loss_func = loss_func

    def forward(self, user_ids, item_ids):
        emb_user = self.emb_user(user_ids)
        emb_item = self.emb_item(item_ids)

        res = -((emb_user - emb_item) ** 2)
        res = res.sum(2)
        return res

    def loss(self, outputs, user_ids, item_ids):
        return (self.loss_func(outputs).mean() +
                self.l2 * self._l2_loss(user_ids, item_ids))

    def _l2_loss(self, user_ids, item_ids):
        """Return the l2 loss for weight decay regularization."""
        all_params = []
        # Embedding for users and items.
        emb_user = self.emb_user(user_ids)
        emb_item = self.emb_item(item_ids)

        all_params += [emb_user] + [emb_item]
        res = 0.0
        for params in all_params:
            res += 0.5 * (params ** 2).sum()
        return res


class NeuralCF(nn.Module):
    def __init__(self, n_users, n_items, hidden_dims, num_factors, ac, loss_func):
        super(NeuralCF, self).__init__()

        self.n_dims = hidden_dims
        self.n_factor = num_factors
        _activations = {"sigm": torch.sigmoid, "tanh": torch.tanh, "relu": F.relu}
        self.ac = _activations[ac]

        self.n_users = n_users
        self.n_items = n_items

        self.emb_user_mf = nn.Embedding(self.n_users, self.n_factor)
        self.emb_item_mf = nn.Embedding(self.n_items, self.n_factor)
        self.emb_user_mlp = nn.Embedding(self.n_users, int(self.n_dims[0] / 2))
        self.emb_item_mlp = nn.Embedding(self.n_items, int(self.n_dims[0] / 2))

        self.mlp_layers = list()
        for i in range(len(self.n_dims) - 1):
            self.mlp_layers.append(
                ('fc%d' % i, nn.Linear(self.n_dims[i], self.n_dims[i + 1])))
        self.mlp_params = nn.Sequential(OrderedDict(self.mlp_layers))

        self.last_fc = nn.Linear(self.n_factor + self.n_dims[-1], 1)

        self.l2 = 1e-4
        self.loss_func = loss_func

    def forward(self, user_ids, item_ids):
        emb_user_mf = self.emb_user_mf(user_ids)
        emb_item_mf = self.emb_item_mf(item_ids)
        emb_user_mlp = self.emb_user_mlp(user_ids)
        emb_item_mlp = self.emb_item_mlp(item_ids)

        # Element-wise Product on MF side.
        mf_output = torch.mul(emb_user_mf, emb_item_mf)
        # Concatenation on MLP side.
        mlp_output = torch.cat([emb_user_mlp.expand_as(emb_item_mlp),
                                emb_item_mlp], dim=-1)
        # MLP layers with ReLU
        for name, layer in self.mlp_layers:
            mlp_input = mlp_output
            layer_output = layer(mlp_input)
            mlp_output = self.ac(layer_output)
            # mlp_output = mlp_input + layer_output
        output = self.last_fc(torch.cat((mf_output, mlp_output), dim=-1))
        return output.squeeze(2)

    def loss(self, outputs, user_ids, item_ids):
        return (self.loss_func(outputs).mean() +
                self.l2 * self._l2_loss(user_ids, item_ids))

    def _l2_loss(self, user_ids, item_ids):
        """Return the l2 loss for weight decay regularization."""
        all_params = []
        # Embedding for users and items.
        emb_user_mf = self.emb_user_mf(user_ids)
        emb_item_mf = self.emb_item_mf(item_ids)
        emb_user_mlp = self.emb_user_mlp(user_ids)
        emb_item_mlp = self.emb_item_mlp(item_ids)

        all_params += ([emb_user_mf] + [emb_item_mf] +
                       [emb_user_mlp] + [emb_item_mlp])
        res = 0.0
        for params in all_params:
            res += 0.5 * (params ** 2).sum()
        return res


class NCFTrainer(Recommender):
    def __init__(self, model_type='NeuralCF', hidden_dims='[128]', num_factors=256, num_epoch=32, batch_size=128, lr=3e-4, *args, **kwargs):
        super(NCFTrainer, self).__init__(*args, **kwargs)

        self.model_type = model_type
        if self.model_type == "NeuralCF":
            self.n_negatives = 5
        elif self.model_type == "CML":
            self.n_negatives = 10
            self.hinge_margin = 10

        self.hidden_dims = hidden_dims
        self.num_factors = num_factors
        self.ac = 'relu'

        self.num_epoch = num_epoch
        self.save_feq = self.num_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.l2 = 1e-4

    def build_network(self):
        if self.model_type == "NeuralCF":
            loss_func = partial(sampled_bce_loss, n_negatives=self.n_negatives)
            self.net = NeuralCF(self.n_users, self.n_items,
                                self.hidden_dims, self.num_factors,
                                self.ac, loss_func).to(self.device)
        elif self.model_type == "CML":
            loss_func = partial(sampled_cml_loss, n_negatives=self.n_negatives,
                margin=self.hinge_margin)
            self.net = CML(self.n_users, self.n_items,
                           self.num_factors, loss_func).to(self.device)
        else:
            raise ValueError("Unknown model type {}".format(self.model_type))

        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=self.lr)

        # For negative sampling.
        self.user_rated_items = None

    def _sample_negative(self, user_ids, pos_item_ids, n_negatives):
        samples = np.empty((user_ids.shape[0], n_negatives), np.int64)
        if self.user_rated_items is None:
            self.user_rated_items = dict()
            for user in range(self.n_users):
                self.user_rated_items[user] = pos_item_ids[user_ids == user]

        for i, u in enumerate(user_ids):
            j = 0
            rated_items = self.user_rated_items[u]
            while j < n_negatives:
                sample_item = int(random.random() * self.n_items)
                if sample_item not in rated_items:
                    samples[i, j] = sample_item
                    j += 1
        return samples

    def train(self):
        best_perf = 0.0
        for epoch in range(1, self.num_epoch + 1):
            time_st = time.time()
            data = self.train_matrix

            # Get training pairs and sample negatives.
            user_ids, pos_item_ids = (data > 0).nonzero()
            neg_item_ids = self._sample_negative(user_ids, pos_item_ids, self.n_negatives)
            user_ids = np.expand_dims(user_ids, 1)
            pos_item_ids = np.expand_dims(pos_item_ids, 1)
            combined_item_ids = np.concatenate([pos_item_ids, neg_item_ids], 1)

            idx_list = np.arange(user_ids.shape[0])

            # Set model to training mode.
            model = self.net.to(self.device)
            model.train()
            np.random.shuffle(idx_list)

            epoch_loss = 0.0
            counter = 0
            for batch_idx in utils.minibatch(
                    idx_list, batch_size=self.batch_size):
                batch_users = user_ids[batch_idx]
                batch_items = combined_item_ids[batch_idx]
                batch_users = torch.LongTensor(batch_users).to(self.device)
                batch_items = torch.LongTensor(batch_items).to(self.device)

                # Compute loss
                outputs = model(user_ids=batch_users, item_ids=batch_items)
                loss = model.loss(outputs=outputs,
                                  user_ids=batch_users,
                                  item_ids=batch_items)
                epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                counter += 1
            epoch_loss = epoch_loss / counter
            print("[TRIAN recommender {}] [{:.1f} s], epoch: {}, loss: {:.4f}".format(self.model_type, time.time() - time_st, epoch, epoch_loss))

            if epoch % self.save_feq == 0:
                result = self.evaluate(verbose=False)
                # Save model checkpoint if it has better performance.
                if result[self.golden_metric] > best_perf:
                    str_metric = "{}={:.4f}".format(self.golden_metric, result[self.golden_metric])
                    # print("Having better model checkpoint with performance {}(epoch:{})".format(str_metric, epoch))
                    self.path_save = os.path.join(self.dir_model, self.model_type)
                    utils.save_checkpoint(self.path_save, self.net, self.optimizer, epoch)
                    best_perf = result[self.golden_metric]

    def recommend(self, data, top_k, return_preds=False, allow_repeat=False):
        # Set model to eval mode
        model = self.net.to(self.device)
        model.eval()

        n_rows = data.shape[0]
        idx_list = np.arange(n_rows)
        recommendations = np.empty([n_rows, top_k], dtype=np.int64)
        all_preds = list()
        with torch.no_grad():
            for batch_idx in utils.minibatch(
                    idx_list, batch_size=self.batch_size):
                cur_batch_size = batch_idx.shape[0]
                batch_data = data[batch_idx].toarray()

                batch_users = np.expand_dims(batch_idx, 1)
                batch_users = torch.LongTensor(batch_users).to(self.device)

                all_items = np.arange(self.n_items)[None, :]
                all_items = np.tile(all_items, (cur_batch_size, 1))
                all_items = torch.LongTensor(all_items).to(self.device)

                preds = model(user_ids=batch_users, item_ids=all_items)
                if return_preds:
                    all_preds.append(preds)
                if not allow_repeat:
                    preds[batch_data.nonzero()] = -np.inf
                if top_k > 0:
                    _, recs = preds.topk(k=top_k, dim=1)
                    recommendations[batch_idx] = recs.cpu().numpy()

        if return_preds:
            return recommendations, torch.cat(all_preds, dim=0).cpu()
        else:
            return recommendations

    def train_with_print(self):
        self.save_feq = 1
        f = open('./HRandRec.txt', mode='w')
        best_perf = 0.0
        for epoch in range(1, self.num_epoch + 1):
            time_st = time.time()
            data = self.train_matrix

            # Get training pairs and sample negatives.
            user_ids, pos_item_ids = (data > 0).nonzero()
            neg_item_ids = self._sample_negative(user_ids, pos_item_ids, self.n_negatives)
            user_ids = np.expand_dims(user_ids, 1)
            pos_item_ids = np.expand_dims(pos_item_ids, 1)
            combined_item_ids = np.concatenate([pos_item_ids, neg_item_ids], 1)

            idx_list = np.arange(user_ids.shape[0])

            # Set model to training mode.
            model = self.net.to(self.device)
            model.train()
            np.random.shuffle(idx_list)

            epoch_loss = 0.0
            counter = 0
            for batch_idx in utils.minibatch(
                    idx_list, batch_size=self.batch_size):
                batch_users = user_ids[batch_idx]
                batch_items = combined_item_ids[batch_idx]
                batch_users = torch.LongTensor(batch_users).to(self.device)
                batch_items = torch.LongTensor(batch_items).to(self.device)

                # Compute loss
                outputs = model(user_ids=batch_users, item_ids=batch_items)
                loss = model.loss(outputs=outputs,
                                  user_ids=batch_users,
                                  item_ids=batch_items)
                epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                counter += 1
            epoch_loss = epoch_loss / counter
            print("[TRIAN recommender {}}] [{:.1f} s], epoch: {}, loss: {:.4f}".format(self.model_type, time.time() - time_st, epoch, epoch_loss))

            if epoch % self.save_feq == 0:
                result = self.evaluate(verbose=False)
                # Save model checkpoint if it has better performance.
                if result[self.golden_metric] > best_perf:
                    str_metric = "{}={:.4f}".format(self.golden_metric, result[self.golden_metric])
                    print("Having better model checkpoint with performance {}(epoch:{})".format(str_metric, epoch))
                    self.path_save = os.path.join(self.dir_model, self.model_type)
                    utils.save_checkpoint(self.path_save, self.net, self.optimizer, epoch)
                    best_perf = result[self.golden_metric]
                result2 = self.validate()

                rec_result_5, rec_result_10, rec_result_20, rec_result_50 = [], [], [], []
                for k, v in result.items():
                    if '5' in k:
                        rec_result_5.append(v)
                    if '10' in k:
                        rec_result_10.append(v)
                    if '20' in k:
                        rec_result_20.append(v)
                    if '50' in k:
                        rec_result_50.append(v)
                for k, v in result2.items():
                    if 'HR' in k:
                        atk_result = v
                line = '\t'.join([str(epoch), str(atk_result), str(rec_result_50[1])]) + '\n'
                f.write(line)
        f.close()
        build_result_graph('./HRandRec.txt', '/', 50, 3, name='rec_atk')

    def fit_withPQ(self, P, Q):
        # get self.train_matrix(real+fake) and self.test_matrix
        self.prepare_data()

        self.build_network()
        self.net.emb_user_mf.train_data = torch.tensor(P)
        self.net.emb_item_mf.train_data = torch.tensor(Q)
        self.net.emb_user_mlp.train_data = torch.tensor(P)
        self.net.emb_item_mlp.train_data = torch.tensor(Q)
        self.train_with_print()

        # Evaluate model performance on test data.
        result1 = self.evaluate()

        # Evaluate attack performance on target item.
        result2 = self.validate()
        return result1, result2