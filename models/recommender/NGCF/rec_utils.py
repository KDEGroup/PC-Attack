import numpy as np
import random as rd
import dgl
from sklearn.metrics import roc_auc_score
import os
import re
import heapq
from scipy import sparse


# ------------------------------Dataset class-------------------------------------
class Data(object):
    def __init__(self, path=None, dataset='filmtrust', batch_size=256):
        self.batch_size = batch_size
        train_file = './data/' + dataset + '/preprocess' + '/train.data'
        test_file = './data/' + dataset + '/preprocess' + '/test.data'
        if path:
            train_file = path

        # get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.dataset = dataset
        self.exist_users = [0]

        user_item_src, user_item_dst, ratings = [], [], []

        with open(train_file, 'r') as f:
            for line in f.readlines():
                arr = line.strip().split("\t")
                u, i, r = int(arr[0]), int(arr[1]), float(arr[2])
                user_item_src.append(int(u))
                user_item_dst.append(int(i))
                # ratings.append(int(float(r)))
                ratings.append(1)
                self.n_users, self.n_items = max(self.n_users, int(u)), max(self.n_items, int(i))
                self.n_train += 1

        user_item_src1, user_item_dst1, ratings1 = [], [], []
        with open(test_file, 'r') as f:
            for line in f.readlines():
                arr = line.strip().split("\t")
                u, i, r = int(arr[0]), int(arr[1]), float(arr[2])
                user_item_src1.append(int(u))
                user_item_dst1.append(int(i))
                # ratings.append(int(float(r)))
                ratings1.append(1)
                self.n_users, self.n_items = max(self.n_users, int(u)), max(self.n_items, int(i))
                self.n_test += 1

        self.n_users, self.n_items = self.n_users + 1, self.n_items + 1

        self.train_matrix = sparse.csr_matrix((ratings, (user_item_src, user_item_dst)), shape=(self.n_users, self.n_items))
        self.test_matrix = sparse.csr_matrix((ratings1, (user_item_src1, user_item_dst1)), shape=(self.n_users, self.n_items))

        self.print_statistics()

        u_ = 0
        self.train_items, items = {}, []
        with open(train_file, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                if u_ < u:
                    if items:
                        self.train_items[u_] = items
                        items = []
                        u_ += 1
                        self.exist_users.append(u_)
                items.append(i)
                line = f.readline()
        if items:
            self.train_items[u_] = items

        # get test_set(dict{user:item_list})
        u_ = 0
        self.test_set, items = {}, []
        with open(test_file, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                if u_ < u:
                    if items:
                        self.test_set[u_] = items
                        items = []
                        u_ += 1
                items.append(i)
                line = f.readline()
        if items:
            self.test_set[u_] = items

        # construct graph from the train data and add self-loops
        user_selfs = [i for i in range(self.n_users)]
        item_selfs = [i for i in range(self.n_items)]

        data_dict = {
            ('user', 'user_self', 'user'): (user_selfs, user_selfs),
            ('item', 'item_self', 'item'): (item_selfs, item_selfs),
            ('user', 'ui', 'item'): (user_item_src, user_item_dst),
            ('item', 'iu', 'user'): (user_item_dst, user_item_src)
        }
        num_dict = {
            'user': self.n_users, 'item': self.n_items
        }

        self.g = dgl.heterograph(data_dict, num_nodes_dict=num_dict)

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            # sample num pos items for u-th user
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d, n_interactions=%d, n_train=%d, n_test=%d, sparsity=%.5f' %
              (self.n_users, self.n_items, self.n_train + self.n_test,
               self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))

# ----------------------------------Metrics------------------------------------------
def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r,cut):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.
    return np.sum(out)/float(min(cut, np.sum(r)))


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


# def hit_at_k(r, k):
#     r = np.array(r)[:k]
#     if np.sum(r) > 0:
#         return 1.
#     else:
#         return 0.


def hit_at_k(target_item, test_items, rating, K):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max_item_score = heapq.nlargest(K, item_score, key=item_score.get)

    if target_item in K_max_item_score:
        return 1
    else:
        return 0


def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.


def auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res


# -------------------------------other function--------------------------------
def txt2list(file_src):
    orig_file = open(file_src, "r")
    lines = orig_file.readlines()
    return lines


def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


def uni2str(unicode_str):
    return str(unicode_str.encode('ascii', 'ignore')).replace('\n', '').strip()


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))


def delMultiChar(inputString, chars):
    for ch in chars:
        inputString = inputString.replace(ch, '')
    return inputString


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop
