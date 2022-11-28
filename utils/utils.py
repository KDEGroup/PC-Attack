# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
import random
from scipy import sparse
from collections import OrderedDict
from math import sqrt
import time

EPSILON = 1e-12


class DefaultListOrderedDict(OrderedDict):
    def __missing__(self, k):
        self[k] = []
        return self[k]


def set_seed(seed, cuda=False):
    """Set seed globally."""
    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    else:
        torch.manual_seed(seed)


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=False):
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard


def get_top_k(arr, k):
    top_k_arr = []
    for line in arr:
        idx = line.argsort()[::-1]
        top_k_idx = idx[:k]
        top_k_arr.append(np.expand_dims(top_k_idx, 0))
    top_k_arr = np.concatenate(top_k_arr, 0)
    return top_k_arr


def get_k(arr, k, top=True):
    top_k_arr = []
    for line in arr:
        tmp = list(map(list, zip(range(len(line)), line)))
        if top:
            large = sorted(tmp, key=lambda x: x[1], reverse=True)
            top_k_idx = np.array(large[:k])[:, 0]
            top_k_arr.append(np.expand_dims(top_k_idx, 0))
        else:
            small = sorted(tmp, key=lambda x: x[1], reverse=False)
            top_k_idx = np.array(small[:k])[:, 0]
            top_k_arr.append(np.expand_dims(top_k_idx, 0))
    top_k_arr = np.concatenate(top_k_arr, 0)
    return top_k_arr


def gumbel_softmax_k(logits, temperature=1, hard=False, k=1):

    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    ind = get_top_k(y.detach().numpy(), k)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, torch.tensor(ind), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard


def cos_sim(vector_a, vector_b):
    num = float(np.dot(vector_a, vector_b.T))
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    return cos


def cal_cosine_faiss(emb, d, n):
    time0 = time.time()
    xb = emb.cpu().detach().numpy()
    xq = emb.cpu().detach().numpy()
    faiss.normalize_L2(xb)
    faiss.normalize_L2(xq)
    nlist = 1
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    index.train(xb)
    index.add(xb)
    k = n
    D, I = index.search(xq, k)
    item_sim_array = np.zeros((n, n))
    for idx1 in range(n):
        idx = []
        for idx2 in range(n):
            idx.append(np.where(I[idx1] == idx2)[0][0])
        item_sim_array[idx1] = D[idx1][idx]
    print('CalsimTime=%.2fs' % (time.time() - time0))
    return item_sim_array


def matrix_cos_similar(v1):
    v2 = v1.t()
    dot_matrix = torch.matmul(v1, v2)
    v1_row_norm = torch.norm(v1, dim=1).reshape(-1, 1)
    v2_col_norm = torch.norm(v2, dim=0).reshape(1, -1)
    norm_matrix = torch.matmul(v1_row_norm, v2_col_norm)
    res = dot_matrix / norm_matrix
    return res


def _pairwise_jaccard(X):
    np.seterr(divide='ignore', invalid='ignore')
    X = X.astype(bool).astype(np.uint16)

    intrsct = X.dot(X.T)
    row_sums = intrsct.diagonal()
    unions = (row_sums[:, None] + row_sums - intrsct).A
    dist = np.asarray(intrsct / unions)
    np.fill_diagonal(dist, 0.0)
    return dist


def get_top_k_from_array(arr, k):
    idx = arr.argsort()[::-1]
    top_k_idx = idx[:k]
    return top_k_idx


def get_bottom_k_from_array(arr, k):
    if k == 0:
        return np.array([])
    idx = arr.argsort()
    bottom_k_idx = idx[:k]
    return bottom_k_idx


def counter(data):
    data_uniq = np.unique(data)
    data_num = []
    for i in data_uniq:
        data_num.append(np.sum(data == i))
    return data_uniq, data_num


def sparse2tensor(sparse_data):
    """Convert sparse csr matrix to pytorch tensor."""
    return torch.FloatTensor(sparse_data.toarray())


def tensor2sparse(tensor):
    """Convert pytorch tensor to sparse csr matrix."""
    return sparse.csr_matrix(tensor.detach().cpu().numpy())


def stack_csrdata(data1, data2):
    """Stack two sparse csr matrix."""
    return sparse.vstack((data1, data2), format="csr")


def save_fake_data(fake_data, path):
    """Save fake data to file."""
    file_path = "%s.npz" % path
    print("Saving fake data to {}".format(file_path))
    sparse.save_npz(file_path, fake_data)
    return file_path


def load_fake_data(file_path):
    """Load fake data from .npz file"""
    fake_data = sparse.load_npz(file_path)
    print("Loaded fake data from {}".format(file_path))
    return fake_data


def load_fake_array1(file_path, n_items):
    """Load fake array from .npy file"""
    fake_array = np.load(file_path, allow_pickle=True)
    print("Loaded fake data from {}".format(file_path))

    n_fake, n_attack = fake_array.shape
    row = list(np.array([[idx] * n_attack for idx in range(n_fake)]).reshape(-1))
    col = list(fake_array.reshape(-1))
    implicit_rating = [1.0] * (n_fake * n_attack)
    matrix_implicit = sparse.csr_matrix((implicit_rating, (row, col)), shape=(n_fake, n_items))
    matrix_implicit[matrix_implicit > 1] = 1
    fake_data = sparse.csr_matrix(matrix_implicit)

    return fake_data


def load_fake_array(file_path):
    """Load fake array from .npy file"""
    fake_array = np.load(file_path, allow_pickle=True)
    fake_array[fake_array > 0] = 1
    print("Loaded fake data from {}".format(file_path))
    return sparse.csr_matrix(fake_array)


def minibatch(*tensors, **kwargs):
    """Mini-batch generator for pytorch tensor."""
    batch_size = kwargs.get('batch_size', 128)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def save_checkpoint(path, model, optimizer=None, epoch=-1):
    """Save model checkpoint and optimizer state to file"""
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None
    }
    file_path = "%s.pt" % path
    print("Saving checkpoint to {}".format(file_path))
    torch.save(state, file_path)


def load_checkpoint(path):
    """Load model checkpoint and optimizer state from file"""
    file_path = "%s.pt" % path
    state = torch.load(file_path, map_location=torch.device('cpu'))
    print("Loaded checkpoint from {}(epoch {})".format(file_path, state["epoch"]))
    return state["epoch"], state["state_dict"], state["optimizer"]


def _array2sparsediag(x):
    values = x
    indices = np.vstack([np.arange(x.size), np.arange(x.size)])

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = [x.size, x.size]

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def extra_same_elem(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    iset = set1.intersection(set2)
    return list(iset)


def multiply(a, b):
    sum_ab = 0.0
    for i in range(len(a)):
        temp = a[i] * b[i]
        sum_ab += temp
    return sum_ab


def cal_pearson1(x, y):
    x = x.tolist()
    y = y.tolist()
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = multiply(x, y)
    sum_x2 = sum([pow(i, 2) for i in x])
    sum_y2 = sum([pow(j, 2) for j in y])
    molecular = sum_xy - (float(sum_x) * float(sum_y) / n)
    denominator = sqrt((sum_x2 - float(sum_x ** 2) / n) * (sum_y2 - float(sum_y ** 2) / n))
    if denominator != 0:
        return molecular / denominator
    else:
        return 0


def cal_pearson(x, y):
    n = len(x)
    sum_x = torch.sum(x)
    sum_y = torch.sum(y)
    sum_xy = torch.mul(x, y).sum()
    sum_x2 = torch.sum(torch.cat([pow(i, 2).unsqueeze(0) for i in x]))
    sum_y2 = torch.sum(torch.cat([pow(i, 2).unsqueeze(0) for i in y]))
    molecular = sum_xy - (sum_x * sum_y / n)
    denominator = sqrt((sum_x2 - sum_x ** 2 / n) * (sum_y2 - sum_y ** 2 / n))
    if denominator != 0:
        return molecular / denominator
    else:
        return torch.tensor(0.0)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return max((x - 1.0) / (warmup - 1.0), 0)


def adjust_learning_rate(epoch, opt, optimizer):
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr


def clip_grad_norm(params, max_norm):
    if max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    else:
        return torch.sqrt(
            sum(p.grad.train_data.norm() ** 2 for p in params if p.grad is not None)
        )


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

