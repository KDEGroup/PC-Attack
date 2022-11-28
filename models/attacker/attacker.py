from torch import nn
import torch
import numpy as np
from scipy import sparse
from bunch import Bunch
import importlib
import argparse
import shutil
import time
import os

from utils import utils
from utils.data_load import Data


class Attacker:
    def __init__(self,
                 dataset,
                 target_item,
                 device=0,
                 n_factor=64,
                 vic_rec='wmf',
                 lr=0.5,
                 num_epochs=64,
                 batch_size=64,
                 num_fake_user=50,
                 num_filler_item=50,
                 num_filler_pop=30,
                 path_atk_emb=None,
                 popularity='pop',
                 lambda_item=0.5,
                 lambda_user=0.5):
        dataset = dataset.split('_')
        if len(dataset) == 1:
            self.dataset = dataset[0]
            self.data_integrity = True
        else:
            self.dataset = dataset[0]
            self.n_pop = int(dataset[1])
            self.n_neigh = int(dataset[2])
            self.percentage = float(dataset[3])
            self.data_integrity = False
        self.target_item = target_item
        self.vic_rec = vic_rec
        self.weight_alpha = 10 if self.dataset == 'automotive' else None

        self.n_factor = n_factor
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lambda_item = lambda_item
        self.lambda_user = lambda_user

        self.injected_dir = './results/fake_data/'
        self.path_atk_emb = path_atk_emb
        self.num_fake_user = num_fake_user
        self.num_filler_item = num_filler_item
        self.num_filler_pop = num_filler_pop
        self.popularity = popularity

        self.device = device
        utils.set_seed(1234)
        self.data_process()
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def data_process(self):
        self.path_train = './data/' + self.dataset + '/preprocess/train.data'
        self.path_test = './data/' + self.dataset + '/preprocess/test.data'
        sep = '\t'
        header = ['user_id', 'item_id', 'rating', 'timestamp']

        self.dataset_class = Data(self.path_train, self.path_test, test_bool=True, header=header, sep=sep, type='attacker')
        self.data_df, _, self.n_users, self.n_items = self.dataset_class.load_file_as_dataFrame()
        _, self.data_matrix = self.dataset_class.dataFrame_to_matrix(self.data_df, self.n_users, self.n_items)
        self.data_array_T = self.data_matrix.T.toarray()
        self.data_array = self.data_matrix.toarray()
        self.n_train = len(self.data_matrix.nonzero()[0])

        if self.data_integrity:
            self.new_path_train = None
            print('Keep data integrity and do no processing！')
        else:
            self.n_sample = int(self.percentage * self.n_train)
            print("n_pop={}, n_neigh={}, percent={}, n_sample={}".
                  format(self.n_pop, self.n_neigh, self.percentage, self.n_sample))
            self.new_path_train = './data/' + self.dataset + '/preprocess/subdata/sub_train_%d_%d_%.3f.data' % \
                                  (self.n_pop, self.n_neigh, self.percentage)
            if not os.path.exists('./data/' + self.dataset + '/preprocess/subdata/'):
                os.makedirs(self.new_path_train)
            num_interaction_users = []
            for item in range(self.n_items):
                users_list = np.where(self.data_array_T[item] > 0)[0]
                num_interaction_users.append(len(users_list))
            if self.popularity == 'pop':
                _, pop_items = torch.topk(torch.tensor(num_interaction_users), self.n_pop)
                selected_items = pop_items
            elif self.popularity == 'longtail':
                tmp = list(map(list, zip(range(len(num_interaction_users)), num_interaction_users)))
                large = sorted(tmp, key=lambda x: x[1], reverse=False)
                longtail_items = np.array(large[:int(0.4 * self.n_items)])[:, 0]
                selected_items = np.random.choice(longtail_items, self.n_pop)
            elif self.popularity == 'random':
                _, random_items = torch.topk(torch.tensor(num_interaction_users), int(0.3 * self.n_items))
                selected_items = np.random.choice(random_items, self.n_pop)

            row, col, rating = [], [], []
            edge_list = []
            cnt = 0
            stop = False
            for item0 in selected_items:
                # 1-hop
                users_list = np.where(self.data_array_T[item0] > 0)[0]
                for user1 in users_list:
                    if (user1, item0) not in edge_list:
                        cnt += 1
                        if cnt > self.n_sample:
                            print('cnt:', cnt)
                            stop = True
                            break
                        row.append(user1)
                        col.append(item0)
                        rating.append(1.0)
                        edge_list.append((user1, item0))

                    # 2-hop
                    items_list = np.where(self.data_array[user1] > 0)[0]
                    for item2 in items_list:
                        if (user1, item2) not in edge_list:
                            cnt += 1
                            if cnt > self.n_sample:
                                print('cnt:', cnt)
                                stop = True
                                break
                            row.append(user1)
                            col.append(item2)
                            rating.append(1.0)
                            edge_list.append((user1, item2))
                    if stop:
                        break
                if stop:
                    break
            self.data_matrix = sparse.csr_matrix((rating, (row, col)), shape=(self.n_users, self.n_items))
            self.data_array_T = self.data_matrix.T.toarray()
            self.data_array = self.data_matrix.toarray()
            print("sub_dataset size:", len(self.data_array.nonzero()[0]))

            data_to_write = np.concatenate([np.expand_dims(x, 1) for x in [np.array(row), np.array(col), np.array(rating)]], 1)
            F_tuple_encode = lambda x: '\t'.join(map(str, [int(x[0]), int(x[1]), x[2]]))
            data_to_write = '\n'.join([F_tuple_encode(tuple_i) for tuple_i in data_to_write])
            with open(self.new_path_train, 'w') as fout:
                fout.write(data_to_write)

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def get_num_interaction_users(self):
        num_interaction_users = []
        data = self.data_matrix.T.toarray()
        for item in range(self.n_items):
            users_list = np.where(data[item] > 0)[0]
            num_interaction_users.append(len(users_list))
        return num_interaction_users

    def build_network(self):
        self.item_embed = nn.Parameter(torch.zeros([self.n_items, self.n_factor]).normal_(mean=0, std=0.1))
        self.user_embed = nn.Parameter(torch.zeros([self.n_users, self.n_factor]).normal_(mean=0, std=0.1))
        self.params = nn.ParameterList([self.item_embed, self.user_embed])
        self.optimizer = torch.optim.SGD(self.params, lr=self.lr)

    def get_loss(self, embed, target, batch_idx, item_flag=True):
        num_batch = len(batch_idx)
        num = self.n_items if item_flag else self.n_users

        t_embed = embed[target].unsqueeze(0).repeat(num_batch, 1)
        l_pos = torch.bmm(embed[batch_idx].view(num_batch, 1, -1), t_embed.view(num_batch, -1, 1))
        l_pos = l_pos.view(num_batch, 1)

        idxs = list(range(num))
        del idxs[target]
        l_neg = torch.mm(embed[idxs], embed[batch_idx].transpose(1, 0))
        l_neg = l_neg.transpose(0, 1)

        out = torch.cat((l_pos, l_neg), dim=1)
        out = torch.div(out, 0.07)

        label = torch.zeros([out.shape[0]]).cuda().long()
        loss = self.criterion(out, label)
        return loss

    def train(self):
        for epoch in range(self.num_epochs):
            time_st = time.time()
            loss_item = 0.0
            idx_list = np.arange(self.n_items)
            for batch_idx in utils.minibatch(idx_list, batch_size=self.batch_size):
                loss_item += self.get_loss(self.item_embed.to(torch.device(self.device)), self.target_item, batch_idx)
            loss_item = loss_item / ((self.n_items - 1) / self.batch_size + 1)

            users_list = np.where(self.data_array_T[self.target_item] > 0)[0]
            loss_user = 0.0
            for user in users_list:
                idx_list = np.arange(self.n_users)
                for batch_idx in utils.minibatch(idx_list, batch_size=self.batch_size):
                    loss_user += self.get_loss(self.user_embed.to(torch.device(self.device)), user, batch_idx, False)
            loss_user = loss_user / len(users_list) if len(users_list) != 0.0 else 0.0
            loss_user = loss_user / ((self.n_users - 1) / self.batch_size + 1)
            loss = self.lambda_item * loss_item + self.lambda_user * loss_user

            print("[TRIAN attacker] [{:.1f} s], epoch: {}, loss: {:.2f}".format(time.time() - time_st, epoch + 1, loss))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def get_fake_users(self):
        fake_users = []
        num_interaction_users = self.get_num_interaction_users()
        p_num = num_interaction_users / np.sum(num_interaction_users)

        t_embed = self.item_embed[self.target_item].unsqueeze(0).repeat(self.n_users, 1)
        sim = torch.bmm(self.user_embed.view(self.n_users, 1, -1), t_embed.view(self.n_users, -1, 1))
        sim = sim.view(self.n_users).detach().cpu().numpy()
        sim = self.normalization(sim)
        p = (sim / sim.sum())
        len_filler = 0
        for user in range(self.num_fake_user):
            fake_user = np.random.choice(a=range(self.n_items), size=self.num_filler_pop, p=p_num, replace=False).tolist()
            fake_user += [self.target_item]

            indexs = np.random.choice(range(len(sim)), size=1, p=p, replace=False)
            for idx in indexs:
                items_list = np.where(self.data_array[idx] > 0)[0].tolist()
                fake_user += items_list
            fake_users.append(fake_user)
            len_filler += len(fake_user)
        print('mean length of fake user:', len_filler/self.num_fake_user)
        return fake_users

    def attack(self, path):
        args = importlib.import_module('args.args_rec')
        args_rec = Bunch(eval('args.{}_{}_args'.format(self.vic_rec, self.dataset)))
        print('RS_args:', eval('args.{}_{}_args'.format(self.vic_rec, self.dataset)))
        trainer = args_rec.pop('trainer')
        rec = trainer(dataset=self.dataset, target_item=self.target_item,
                      path_fake_matrix=path, device=self.device, **args_rec)

        results_rs, results_atk = rec.fit()
        return results_rs, results_atk

    def save(self, results_dir, path_save_fake, fake_users):
        print(f"[SAVE FakeData in attacker] Writing fake users to: {results_dir}")
        if not os.path.exists(results_dir):
            print(f"[SAVE FakeData in attacker] Creating output root directory {results_dir}")
            os.makedirs(results_dir)

        row, col, rating = [], [], []
        for idx, items in enumerate(fake_users):
            for item in items:
                row.append(idx)
                col.append(item)
                rating.append(1.0)
        matrix = sparse.csr_matrix((rating, (row, col)), shape=(self.num_fake_user, self.n_items))
        path = os.path.join(results_dir, path_save_fake + '.npz')
        sparse.save_npz(path, matrix)
        return path

    def generate_injectedFile(self, injected_dir, path_fake_array=None, path_fake_matrix=None):
        injected_path = injected_dir + self.dataset + '/%s_attacker_%d.data' % (self.dataset, self.target_item)
        if not os.path.exists(os.path.join(injected_dir, self.dataset)):
            os.makedirs(os.path.join(injected_dir, self.dataset))
        if os.path.exists(injected_path):
            os.remove(injected_path)
        shutil.copyfile(self.path_train, injected_path)

        if path_fake_matrix:
            fake_matrix = sparse.load_npz(path_fake_matrix)
            fake_array = fake_matrix.toarray()
        if path_fake_array:
            fake_array = np.load(path_fake_array)
        uids = np.where(fake_array > 0)[0] + self.n_users
        iids = np.where(fake_array > 0)[1]
        values = fake_array[fake_array > 0]

        data_to_write = np.concatenate([np.expand_dims(x, 1) for x in [uids, iids, values]], 1)
        F_tuple_encode = lambda x: '\t'.join(map(str, [int(x[0]), int(x[1]), x[2]]))
        data_to_write = '\n'.join([F_tuple_encode(tuple_i) for tuple_i in data_to_write])
        with open(injected_path, 'a+')as fout:
            fout.write(data_to_write)

    def run(self, item_embed, user_embed):
        loss_item = 0.0
        idx_list = np.arange(len(item_embed))
        np.random.shuffle(idx_list)
        for batch_idx in utils.minibatch(idx_list, batch_size=self.batch_size):
            loss_item += self.get_loss(item_embed.to(torch.device(self.device)), self.target_item, batch_idx)
        loss_item = loss_item / ((len(item_embed) - 1) / self.batch_size + 1)
        users_list = np.where(self.data_array_T[self.target_item] > 0)[0]
        loss_user = 0.0
        for user in users_list:
            idx_list = np.arange(len(user_embed))
            np.random.shuffle(idx_list)
            for batch_idx in utils.minibatch(idx_list, batch_size=self.batch_size):
                loss_user += self.get_loss(user_embed.to(torch.device(self.device)), user, batch_idx, False)
        loss_user = loss_user / len(users_list) if len(users_list) != 0.0 else 0.0
        loss_user = loss_user / ((len(user_embed) - 1) / self.batch_size + 1)
        loss = self.lambda_item * loss_item + self.lambda_user * loss_user
        return loss

    def fit(self):
        self.build_network()
        if self.path_atk_emb:
            data = np.load(self.path_atk_emb)
            self.item_embed.data = torch.tensor(data[self.n_users:]).to(torch.device(self.device))
            self.user_embed.data = torch.tensor(data[:self.n_users]).to(torch.device(self.device))
        self.train()
        fake_users = self.get_fake_users()
        results_dir = './results/fake_matrix/%s/' % self.dataset
        path_save_fake = 'fake_matrix_%s_%d' % (self.dataset, self.target_item)
        path = self.save(results_dir, path_save_fake, fake_users)
        results_rs, results_atk = self.attack(path)
        self.generate_injectedFile(injected_dir=self.injected_dir,
                                   path_fake_array=None,
                                   path_fake_matrix=results_dir + path_save_fake + '.npz')
        return results_rs, results_atk
