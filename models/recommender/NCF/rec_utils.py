import pandas as pd
import scipy.sparse as sp
import numpy as np
import torch.utils.data


def load_data(dataset, path_fake_data=None):
    path_train = './data/' + dataset + '/preprocess/train.data'
    if path_fake_data:
        path_train = path_fake_data
    path_test = './data/' + dataset + '/preprocess/test.data'
    sep = '\t'
    header = ['user_id', 'item_id', 'rating', 'timestamp']

    train_df = pd.read_csv(path_train, sep=sep, names=header)
    test_df = pd.read_csv(path_test, sep=sep, names=header)
    train_list = train_df.values.tolist()
    test_list = test_df.values.tolist()

    n_users = max(max(test_df.user_id.unique()), max(train_df.user_id.unique())) + 1
    n_items = max(max(test_df.item_id.unique()), max(train_df.item_id.unique())) + 1

    # load ratings as a dok matrix.
    train_matrix = sp.dok_matrix((n_users, n_items), dtype=np.float32)
    for x in train_list:
        train_matrix[x[0], x[1]] = 1.0

    test_matrix = sp.dok_matrix((n_users, n_items), dtype=np.float32)
    for x in test_list:
        test_matrix[x[0], x[1]] = 1.0

    return n_users, n_items, train_matrix, test_matrix, train_list, test_list


class NCFData(torch.utils.data.Dataset):
    def __init__(self, data, train_matrix, num_ng=0, is_training=False):
        super(NCFData, self).__init__()
        self.data_pos = data
        self.n_items = train_matrix.shape[1]
        self.train_matrix = train_matrix
        self.num_ng = num_ng
        self.is_training = is_training
        self.labels_pos = [0 for _ in range(len(data))]

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing.'

        self.data_neg = []
        for x in self.data_pos:
            u = x[0]
            for _ in range(self.num_ng):
                j = np.random.randint(self.n_items)
                while (u, j) in self.train_matrix:
                    j = np.random.randint(self.n_items)
                self.data_neg.append([u, j])

        labels_pos = [1 for _ in range(len(self.data_pos))]
        labels_neg = [0 for _ in range(len(self.data_neg))]
        self.data = self.data_pos + self.data_neg
        self.labels = labels_pos + labels_neg

    def __len__(self):
        return (self.num_ng + 1) * len(self.labels_pos)

    def __getitem__(self, idx):
        data = self.data if self.is_training else self.data_pos
        labels = self.labels if self.is_training else self.labels_pos

        user = data[idx][0]
        item = data[idx][1]
        label = labels[idx]
        return user, item, label