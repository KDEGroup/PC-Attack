import numpy as np
import torch


class BPRData(torch.utils.data.Dataset):
    def __init__(self, train_data, train_matrix=None, is_training=None):
        super(BPRData, self).__init__()

        self.train_data = np.array(train_data)
        self.train_matrix = train_matrix
        self.is_training = is_training
        if self.is_training:
            self.ng_sample()

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'
        tmp_train_matrix = self.train_matrix.todok()
        n_items = self.train_matrix.shape[1]

        length = self.train_data.shape[0]
        self.neg_data = np.random.randint(low=0, high=n_items, size=length)

        for i in range(length):
            uid = self.train_data[i][0]
            iid = self.neg_data[i]
            if (uid, iid) in tmp_train_matrix:
                while (uid, iid) in tmp_train_matrix:
                    iid = np.random.randint(low=0, high=n_items)
                self.neg_data[i] = iid

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        user = self.train_data[idx][0]
        item_i = self.train_data[idx][1]
        if self.is_training:
            item_j = self.neg_data[idx]
            return user, item_i, item_j
        else:
            return user, item_i

