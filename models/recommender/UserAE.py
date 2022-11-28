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


# Variational Auto-Encoder
class UserVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, betas):
        super(UserVAE, self).__init__()
        self.q_dims = [input_dim] + hidden_dims
        self.p_dims = self.q_dims[::-1]
        # Double the letent code dimension for both mu and log_var in VAE.
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out
                                       in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out
                                       in zip(self.p_dims[:-1], self.p_dims[1:])])

        beta_init, self.beta_acc, self.beta_final = betas
        self.beta = beta_init

        self.recon_loss = mult_ce_loss
        self.kld_loss = kld_loss

    def encode(self, input):
        h = input
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if 1 != len(self.q_layers) - 1:
                h = torch.tanh(h)
            else:
                mu, log_var = torch.split(h, self.q_dims[-1], dim=1)
                return mu, log_var

    # 重参数
    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def forward(self, input, predict=False, **kwargs):
        # Annealing beta while training.
        if self.training:
            if self.beta_acc >= 0 and self.beta < self.beta_final:
                self.beta += self.beta_acc
            elif self.beta_acc < 0 and self.beta > self.beta_final:
                self.beta += self.beta_acc

        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        if predict:
            return self.decode(z)
        return self.decode(z), mu, log_var

    def loss(self, data, outputs):
        logits, mu, log_var = outputs
        return self.recon_loss(data, logits) + self.beta * self.kld_loss(mu, log_var)


# Collaborative Denoising Auto-Encoders
class CDAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_users, drop_rate):
        super(CDAE, self).__init__()
        self.q_dims = [input_dim] + hidden_dims
        self.p_dims = self.q_dims[::-1]

        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out
                                       in zip(self.q_dims[:-1], self.q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out
                                       in zip(self.p_dims[:-1], self.p_dims[1:])])
        self.user_node = nn.Parameter(torch.randn([n_users, self.q_dims[-1]]),
                                      requires_grad=True)

        self.drop_input = torch.nn.Dropout(drop_rate)
        self.recon_loss = mult_ce_loss

    def encode(self, input):
        h = input
        h = self.drop_input(h)
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            h = h * F.sigmoid(h)
        return h

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.selu(h)
        return h

    def forward(self, input, batch_user, **kwargs):
        z = self.encode(input) + self.user_node[batch_user]
        return self.decode(z)

    def loss(self, data, outputs):
        logits = outputs
        return self.recon_loss(data, logits)


class UserAETrainer(Recommender):
    def __init__(self, model_type='UserVAE', hidden_dims='[600, 300]', num_epoch=50, batch_size=128, lr=1e-3, *args, **kwargs):
        super(UserAETrainer, self).__init__(*args, **kwargs)
        self.model_type = model_type

        self.hidden_dims = eval(hidden_dims)
        if self.model_type == 'UserVAE':
            self.betas = [0.0, 1e-5, 1.0]
        elif self.model_type == 'CDAE':
            self.drop_rate = 0.5

        self.num_epoch = num_epoch
        self.save_feq = num_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.l2 = 1e-6

    def build_network(self):
        if self.model_type == 'UserVAE':
            self.net = UserVAE(self.n_items, self.hidden_dims, self.betas).to(self.device)
        elif self.model_type == 'CDAE':
            self.net = CDAE(self.n_items, self.hidden_dims, self.n_users, self.drop_rate).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.l2)

    def train(self):
        best_perf = 0.0
        for epoch in range(1, self.num_epoch + 1):
            time_st = time.time()
            data = self.train_matrix

            n_rows = data.shape[0]
            n_cols = data.shape[1]
            idx_list = np.arange(n_rows)

            # Set model to training mode
            model = self.net.to(self.device)
            model.train()
            np.random.shuffle(idx_list)

            epoch_loss = 0.0
            counter = 0
            for batch_idx in utils.minibatch(idx_list, batch_size=self.batch_size):
                batch_tensor = utils.sparse2tensor(data[batch_idx]).to(self.device)

                # Compute loss
                outputs = model(batch_tensor, batch_user=batch_idx)
                loss = model.loss(data=batch_tensor, outputs=outputs).mean()
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
        # Set model to eval mode.
        model = self.net.to(self.device)
        model.eval()

        n_rows = data.shape[0]
        idx_list = np.arange(n_rows)
        recommendations = np.empty([n_rows, top_k], dtype=np.int64)

        # Make predictions first, and then sort for top-k.
        with torch.no_grad():
            data_tensor = utils.sparse2tensor(data).to(self.device)
            preds = model(data_tensor, batch_user=idx_list, predict=True)

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