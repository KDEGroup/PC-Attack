import torch
from torch.utils import data
import pandas as pd
from scipy.sparse import csr_matrix


class Data(object):
    def __init__(self, path_train, path_test=None, test_bool=False, header=None, sep='\t', threshold=0, verbose=True,
                 type='', n_users=None, n_items=None):

        self.test_bool = test_bool
        self.path_train = path_train
        self.path_test = path_test
        self.header = header if header is not None else ['user_id', 'item_id', 'rating', 'timestamp']

        self.sep = sep
        self.threshold = threshold
        self.verbose = verbose
        self.type = type
        self.ori_n_users, self.ori_n_items = n_users, n_items

    def load_file_as_dataFrame(self):
        # load data to pandas dataframe

        if not self.test_bool:
            if self.verbose:
                print("\n[LOAD DATA in train %s] load data from %s ..." % (self.type, self.path_train), flush=True)
            data_df = pd.read_csv(self.path_train, sep=self.sep, names=self.header, engine='python')
            data_df = data_df.loc[:, ['user_id', 'item_id', 'rating']]

            # data statics]
            n_users = max(data_df.user_id.unique()) + 1
            n_items = max(data_df.item_id.unique()) + 1

            if self.verbose:
                print("[DATA INFO in train %s] Number of users : %d , Number of items : %d. " % (self.type, n_users, n_items), flush=True)
                print("[DATA INFO in train %s] Train data size : %d  " % (self.type, data_df.shape[0]), flush=True)

            return data_df, n_users, n_items
        else:
            if self.verbose:
                print("\n[LOAD DATA in train %s] load data from %s ..." % (self.type, self.path_train), flush=True)
            train_df = pd.read_csv(self.path_train, sep=self.sep, names=self.header, engine='python')
            if len(self.header) == 2:
                train_df.insert(loc=2, column='rating', value=[1] * len(train_df))

            train_df = train_df.loc[:, ['user_id', 'item_id', 'rating']]

            if self.verbose:
                print("[LOAD DATA in train %s] load data from %s ..." % (self.type, self.path_test), flush=True)
            test_df = pd.read_csv(self.path_test, sep=self.sep, names=self.header, engine='python')
            if len(self.header) == 2:
                test_df.insert(loc=2, column='rating', value=[1] * len(test_df))
            test_df = test_df.loc[:, ['user_id', 'item_id', 'rating']]

            # data statics
            n_users = max(max(test_df.user_id.unique()), max(train_df.user_id.unique())) + 1
            n_items = max(max(test_df.item_id.unique()), max(train_df.item_id.unique())) + 1

            if self.ori_n_users:
                n_users = self.ori_n_users
            if self.ori_n_items:
                n_items = self.ori_n_items

            if self.verbose:
                print("[DATA INFO in train %s] Number of users : %d , Number of items : %d. " % (self.type, n_users, n_items), flush=True)
                print("[DATA INFO in train %s] Train size : %d , Test size : %d. " % (self.type, train_df.shape[0], test_df.shape[0]), flush=True)

            return train_df, test_df, n_users, n_items

    def dataFrame_to_matrix(self, data_frame, n_users, n_items):
        row, col, rating, implicit_rating = [], [], [], []
        for line in data_frame.itertuples():
            uid, iid, r = list(line)[1:]
            r = int(r)
            implicit_r = 1.0 if r > self.threshold else 0.0

            row.append(uid)
            col.append(iid)
            rating.append(r)
            implicit_rating.append(implicit_r)

        matrix = csr_matrix((rating, (row, col)), shape=(n_users, n_items))
        matrix_implicit = csr_matrix((implicit_rating, (row, col)), shape=(n_users, n_items))
        return matrix, matrix_implicit


class Dataset(data.Dataset):
    def __init__(self, data_matrix):
        self.data_tensor = torch.tensor(data_matrix, dtype=torch.float32)

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.shape[0]

