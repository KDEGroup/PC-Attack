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


class WeightedMF(nn.Module):
    def __init__(self, n_users, n_items, hidden_dim):
        super(WeightedMF, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.dim = hidden_dim  # WMF can only have one latent dimension.

        self.Q = nn.Parameter(torch.zeros([self.n_items, self.dim]).normal_(mean=0, std=0.1))
        self.P = nn.Parameter(torch.randn([self.n_users, self.dim]).normal_(mean=0, std=0.1))
        self.params = nn.ParameterList([self.Q, self.P])

    def forward(self, user_id=None, item_id=None):
        if user_id is None and item_id is None:
            return torch.mm(self.P, self.Q.t())
        if user_id is not None:
            return torch.mm(self.P[[user_id]], self.Q.t())
        if item_id is not None:
            return torch.mm(self.P, self.Q[[item_id]].t())

    def get_norm(self, user_id=None, item_id=None):
        l2_reg = torch.norm(self.P[[user_id]], p=2, dim=-1).sum() + torch.norm(self.Q[[item_id]], p=2, dim=-1).sum()
        return l2_reg


class MFTrainer(Recommender):
    def __init__(self,
                 dim=64,
                 num_epoch=50,
                 batch_size=128,
                 lr=0.005,
                 weight_alpha=None,
                 *args,
                 **kwargs):
        super(MFTrainer, self).__init__(*args, **kwargs)

        self.dim = dim
        self.num_epoch = num_epoch
        self.save_feq = self.num_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.l2 = 1e-5
        self.weight_alpha = weight_alpha

    def build_network(self):
        self.net = WeightedMF(self.n_users, self.n_items, self.dim).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.l2)

    def train(self):
        best_perf = 0.0
        for epoch in range(1, self.num_epoch + 1):
            time_st = time.time()
            data = self.train_matrix
            n_rows = data.shape[0]
            idx_list = np.arange(n_rows)

            # Set model to training mode.
            model = self.net.to(self.device)
            model.train()
            np.random.shuffle(idx_list)

            epoch_loss = 0.0
            for batch_idx in utils.minibatch(idx_list, batch_size=self.batch_size):
                batch_tensor = utils.sparse2tensor(data[batch_idx]).to(self.device)

                # Compute loss
                outputs = model(user_id=batch_idx)
                if self.weight_alpha:
                    loss = mse_loss(data=batch_tensor,
                                    logits=outputs,
                                    weight=self.weight_alpha).sum()
                else:
                    loss = torch.nn.MSELoss()(batch_tensor, outputs)
                epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print("[TRIAN recommender WMF] [{:.1f} s], epoch: {}, loss: {:.4f}".format(time.time() - time_st, epoch, epoch_loss))

            if epoch % self.save_feq == 0:
                result = self.evaluate(verbose=False)
                # Save model checkpoint if it has better performance.
                if result[self.golden_metric] > best_perf:
                    str_metric = "{}={:.4f}".format(self.golden_metric, result[self.golden_metric])
                    # print("Having better model checkpoint with performance {}(epoch:{})".format(str_metric, epoch))
                    self.path_save = os.path.join(self.dir_model, 'WMF_sgd')
                    utils.save_checkpoint(self.path_save, self.net, self.optimizer, epoch)
                    best_perf = result[self.golden_metric]

    def train_with_print(self):
        self.save_feq = 1
        f = open('./HRandRec.txt', mode='w')
        best_perf = 0.0
        for epoch in range(1, self.num_epoch + 1):
            time_st = time.time()
            data = self.train_matrix
            n_rows = data.shape[0]
            idx_list = np.arange(n_rows)

            # Set model to training mode.
            model = self.net.to(self.device)
            model.train()
            np.random.shuffle(idx_list)

            epoch_loss = 0.0
            for batch_idx in utils.minibatch(idx_list, batch_size=self.batch_size):
                batch_tensor = utils.sparse2tensor(data[batch_idx]).to(self.device)

                # Compute loss
                outputs = model(user_id=batch_idx)
                # loss = mse_loss(data=batch_tensor,
                #                 logits=outputs,
                #                 weight=self.weight_alpha).sum()
                loss = torch.nn.MSELoss()(batch_tensor, outputs)
                epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print("[TRIAN recommender WMF] [{:.1f} s], epoch: {}, loss: {:.4f}".format(time.time() - time_st, epoch, epoch_loss))

            if epoch % self.save_feq == 0:
                result = self.evaluate(verbose=False)
                # Save model checkpoint if it has better performance.
                if result[self.golden_metric] > best_perf:
                    str_metric = "{}={:.4f}".format(self.golden_metric, result[self.golden_metric])
                    # print("Having better model checkpoint with performance {}(epoch:{})".format(str_metric, epoch))
                    self.path_save = os.path.join(self.dir_model, 'WMF_sgd')
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

    def recommend(self, data, top_k, return_preds=False, allow_repeat=False):
        # Set model to eval mode.
        model = self.net.to(self.device)
        model.eval()

        n_rows = data.shape[0]
        recommendations = np.empty([n_rows, top_k], dtype=np.int64)

        # Make predictions first, and then sort for top-k.
        with torch.no_grad():
            preds = model()

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

    def fit_withPQ(self, P, Q):
        self.prepare_data()
        self.build_network()
        self.net.P.data = torch.tensor(P)
        self.net.Q.data = torch.tensor(Q)
        # self.train()
        # self.train_with_print()

        # Evaluate model performance on test data.
        result1 = self.evaluate()

        # Evaluate attack performance on target item.
        result2 = self.validate()
        return result1, result2


class WMFTrainer(Recommender):
    def __init__(self, optim_method='sgd', dim=64, num_epoch=50, weight_alpha=None,*args, **kwargs):
        super(WMFTrainer, self).__init__(*args, **kwargs)

        self.dim = dim

        if optim_method == 'sgd':
            self.num_epoch = num_epoch
            self.save_feq = num_epoch
        elif optim_method == 'als':
            self.num_epoch = 20
            self.save_feq = 20
        self.batch_size = 128
        self.lr = 0.005
        self.l2 = 1e-5

        self.weight_alpha = 20
        self.optim_method = optim_method

    def build_network(self):
        self.net = WeightedMF(self.n_users, self.n_items, self.dim).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.l2)

    def train(self):
        if self.optim_method not in ('sgd', 'als'):
            raise ValueError("Unknown optim_method {} for WMF.".format(self.optim_method))

        if self.optim_method == 'sgd':
            self.train_sgd()
        if self.optim_method == 'als':
            self.train_als()

    def train_als(self):
        best_perf = 0.0
        for epoch in range(1, self.num_epoch + 1):
            time_st = time.time()
            data = self.train_matrix

            model = self.net
            P = model.P.detach()
            Q = model.Q.detach()

            weight_alpha = self.weight_alpha - 1
            # Using Pytorch for ALS optimization
            # Update P
            lambda_eye = torch.eye(self.dim).to(self.device) * self.l2
            # residual = Q^tQ + lambda*I
            residual = torch.mm(Q.t(), Q) + lambda_eye
            for user, batch_data in enumerate(data):
                # x_u: N * 1
                x_u = utils.sparse2tensor(batch_data).to(self.device).t()
                cu = batch_data.toarray().squeeze() * weight_alpha + 1
                Cu = utils._array2sparsediag(cu).to(self.device)
                Cu_minusI = utils._array2sparsediag(cu - 1).to(self.device)
                # Q^tCuQ + lambda*I = Q^tQ + lambda*I + Q^t(Cu-I)Q
                # left hand side
                lhs = torch.mm(Q.t(), Cu_minusI.mm(Q)) + residual
                # right hand side
                rhs = torch.mm(Q.t(), Cu.mm(x_u))

                new_p_u = torch.mm(lhs.inverse(), rhs)
                model.P.data[user] = new_p_u.t()

            # Update Q
            data = data.transpose()
            # residual = P^tP + lambda*I
            residual = torch.mm(P.t(), P) + lambda_eye
            for item, batch_data in enumerate(data):
                # x_v: M x 1
                x_v = utils.sparse2tensor(batch_data).to(self.device).t()
                # Cv = diagMat(alpha * rating + 1)
                cv = batch_data.toarray().squeeze() * weight_alpha + 1
                Cv = utils._array2sparsediag(cv).to(self.device)
                Cv_minusI = utils._array2sparsediag(cv - 1).to(self.device)
                # left hand side
                lhs = torch.mm(P.t(), Cv_minusI.mm(P)) + residual
                # right hand side
                rhs = torch.mm(P.t(), Cv.mm(x_v))

                new_q_v = torch.mm(lhs.inverse(), rhs)
                model.Q.data[item] = new_q_v.t()

            print("[TRIAN recommender WMF_als] [{:.1f} s], epoch: {}".format(time.time() - time_st, epoch))

            if epoch % self.save_feq == 0:
                result = self.evaluate(verbose=False)
                # Save model checkpoint if it has better performance.
                if result[self.golden_metric] > best_perf:
                    str_metric = "{}={:.4f}".format(self.golden_metric, result[self.golden_metric])
                    # print("Having better model checkpoint with performance {}(epoch:{})".format(str_metric, epoch))
                    self.path_save = os.path.join(self.dir_model, 'WMF_als')
                    utils.save_checkpoint(self.path_save, self.net, self.optimizer, epoch)
                    best_perf = result[self.golden_metric]

    def train_sgd(self):
        best_perf = 0.0
        for epoch in range(1, self.num_epoch + 1):
            time_st = time.time()
            data = self.train_matrix
            n_rows = data.shape[0]
            n_cols = data.shape[1]
            idx_list = np.arange(n_rows)

            # Set model to training mode.
            model = self.net.to(self.device)
            model.train()
            np.random.shuffle(idx_list)

            epoch_loss = 0.0
            for batch_idx in utils.minibatch(idx_list, batch_size=self.batch_size):
                batch_tensor = utils.sparse2tensor(data[batch_idx]).to(self.device)

                # Compute loss
                outputs = model(user_id=batch_idx)
                l2_norm = model.get_norm(user_id=batch_idx)
                loss = mse_loss(data=batch_tensor,
                                logits=outputs,
                                weight=self.weight_alpha).sum()
                # loss += l2_norm * 10
                epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print("[TRIAN recommender WMF_sgd] [{:.1f} s], epoch: {}, loss: {:.4f}".format(time.time() - time_st, epoch, epoch_loss))

            if epoch % self.save_feq == 0:
                result = self.evaluate(verbose=False)
                # Save model checkpoint if it has better performance.
                if result[self.golden_metric] > best_perf:
                    str_metric = "{}={:.4f}".format(self.golden_metric, result[self.golden_metric])
                    # print("Having better model checkpoint with performance {}(epoch:{})".format(str_metric, epoch))
                    self.path_save = os.path.join(self.dir_model, 'WMF_sgd')
                    utils.save_checkpoint(self.path_save, self.net, self.optimizer, epoch)
                    best_perf = result[self.golden_metric]

    def recommend(self, data, top_k, return_preds=False, allow_repeat=False):
        # Set model to eval mode.
        model = self.net.to(self.device)
        model.eval()

        n_rows = data.shape[0]
        idx_list = np.arange(n_rows)
        recommendations = np.empty([n_rows, top_k], dtype=np.int64)

        # Make predictions first, and then sort for top-k.
        with torch.no_grad():
            data_tensor = utils.sparse2tensor(data).to(self.device)
            preds = model()

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

    def fit_withPQ(self, P, Q):
        # get self.train_matrix(real+fake) and self.test_matrix
        self.prepare_data()

        self.build_network()
        self.net.P.data = torch.tensor(P)
        self.net.Q.data = torch.tensor(Q)
        # self.train()
        # self.train_with_print()

        # Evaluate model performance on test data.
        result1 = self.evaluate()

        # Evaluate attack performance on target item.
        result2 = self.validate()
        return result1, result2