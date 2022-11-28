import dgl
import torch
import numpy as np
import itertools
import os
import time
import warnings
import argparse
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.criterions import NCESoftmaxLoss, NCESoftmaxLossNS
from models.pretrain.memory_moco import MemoryMoCo
from utils.graph_data_load import (RecDataset, LoadBalanceGraphDataset, worker_init_fn)
from utils.graph_data_util import batcher
from models.pretrain.graph_encoder import GraphEncoder
from models.pretrain.seq_encoder import SeqEncoder
from utils.utils import adjust_learning_rate
from models.attacker.attacker import Attacker
from utils.data_load import Data


def parse_args():
    parser = argparse.ArgumentParser("argument for training")

    # dataset definition
    parser.add_argument("--dataset", type=str, default="filmtrust")
    parser.add_argument("--target-dataset", type=str, default="filmtrust")
    parser.add_argument("--target-item", type=int, default=5)
    parser.add_argument("--target-restart-prob", type=float, default=0.8)
    parser.add_argument("--target-rw-hops", type=int, default=64)
    #
    parser.add_argument("--is-load", type=int, default=0)
    parser.add_argument("--path_load_model", type=str, help="path to load model")

    # specify folder
    parser.add_argument("--model-path", type=str, default='saved', help="path to save model")
    parser.add_argument("--result-dir", type=str, default='results', help="path to save result")
    parser.add_argument("--model-type", type=str, default='attacker')
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    parser.add_argument("--seed", type=int, default=1234, help="random seed.")
    parser.add_argument("--save-freq", type=int, default=1, help="save frequency")

    # optimization
    parser.add_argument("--batch-size", type=int, default=32, help="batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--optimizer", type=str, default='adam', choices=['sgd', 'adam', 'adagrad'], help="optimizer")
    parser.add_argument("--learning-rate", type=float, default=0.005, help="learning rate")
    parser.add_argument("--lr_decay_epochs", type=str, default="120,160,200", help="where to decay lr, can be a list")

    # random walk, get 2 data_augmentation(sub_graph).
    parser.add_argument("--restart-prob", type=float, default=0.8)
    parser.add_argument("--rw-hops", type=int, default=64)
    parser.add_argument("--positional-embedding-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=12, help="num of workers to use")
    parser.add_argument("--num-copies", type=int, default=6, help="num of dataset copies that fit in memory")
    parser.add_argument("--num-samples", type=int, default=2000, help="num of samples per batch per worker")

    # graph encoder model
    parser.add_argument("--num-layer", type=int, default=5, help="gnn layers")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--max-degree", type=int, default=512)
    parser.add_argument("--degree-embedding-size", type=int, default=16)
    parser.add_argument("--norm", action="store_true", default=True, help="apply 2-norm on output feats")
    parser.add_argument("--moco", type=int, default=1, help="using MoCo (otherwise Instance Discrimination)")

    # loss function
    parser.add_argument("--nce-k", type=int, default=256)
    parser.add_argument("--nce-t", type=float, default=0.07)
    parser.add_argument("--alpha", type=float, default=0.999, help="exponential moving average weight")
    parser.add_argument("--lambda_g", type=float, default=0.5)
    parser.add_argument("--lambda_s", type=float, default=0.5)
    parser.add_argument("--lambda_1", type=float, default=0.5)
    parser.add_argument("--lambda_2", type=float, default=0.5)
    parser.add_argument("--lambda_user", type=float, default=0.5)
    parser.add_argument("--lambda_item", type=float, default=0.5)

    args = parser.parse_args()
    args.lr_decay_epochs = [int(x) for x in args.lr_decay_epochs.split(",")]
    return args


def build_result_graph(path_load, path_save):
    data = np.loadtxt(path_load, delimiter='\t')

    fig = plt.figure()
    plt.subplot(1, 1, 1)
    plt.plot(data[:, 0], data[:, 1], label='loss1')
    plt.plot(data[:, 0], data[:, 2], label='loss2')

    # plt.title('Training Loss')
    plt.legend()
    plt.xlabel(u'epoch')
    plt.ylabel(u'loss')
    plt.savefig(os.path.join(path_save, 'loss'))


def args_update(args):
    args.sub_target_dataset = args.target_dataset
    args.target_dataset = args.target_dataset.split('_')[0]

    path_train = './data/' + args.target_dataset + '/preprocess/train.data'
    path_test = './data/' + args.target_dataset + '/preprocess/test.data'
    sep = '\t'
    header = ['user_id', 'item_id', 'rating', 'timestamp']

    dataset_class = Data(path_train, path_test, test_bool=True, header=header, sep=sep, type='pretrain')
    _, _, args.ori_n_users, args.ori_n_items = dataset_class.load_file_as_dataFrame()

    args.model_name = "{}_to_{}_{}".format(args.dataset, args.sub_target_dataset, args.target_item)

    args.model_folder = os.path.join(args.model_path, args.model_type)

    args.model_save_dir = os.path.join(args.model_folder, args.model_name)
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    result_save_folder = os.path.join(args.result_dir, args.model_type, args.model_name)

    args.result_graph_save_path = os.path.join(result_save_folder, 'diagram')
    if not os.path.exists(args.result_graph_save_path):
        os.makedirs(args.result_graph_save_path)

    return args


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


def train_moco(epoch, train_loader, target_train_loader, model_graph, model_graph_ema, model_seq, model_seq_ema, contrast, atk, criterion, optimizer, args):
    """one epoch training for moco"""
    model_graph.train()
    model_graph_ema.eval()
    model_seq.train()
    model_seq_ema.eval()

    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.train()

    model_graph_ema.apply(set_bn_train)
    model_seq_ema.apply(set_bn_train)

    loss1_total = 0.0
    for idx, batch in enumerate(train_loader):
        graph_q, graph_k = batch

        graph_q.to(torch.device(args.gpu))
        graph_k.to(torch.device(args.gpu))

        # ===================Moco forward=====================
        graph_view_q = model_graph(graph_q)
        seq_view_q = model_seq(graph_q)
        with torch.no_grad():
            graph_view_k = model_graph_ema(graph_k)
            seq_view_k = model_seq_ema(graph_k)
        out_g, out_s = contrast(graph_view_q, graph_view_k, seq_view_q, seq_view_k)
        loss1 = args.lambda_g * criterion(out_g) + args.lambda_s * criterion(out_s)
        loss1_total += loss1

        # ===================backward=====================
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        if args.moco:
            moment_update(model_graph, model_graph_ema, args.alpha)
            moment_update(model_seq, model_seq_ema, args.alpha)

        print("[train-in-A] [{}]\t loss1: {:.2f}".format(epoch, loss1))
    loss1_total /= len(train_loader)

    emb_list = []
    for idx, batch in enumerate(target_train_loader):
        subgraph, _ = batch
        subgraph.to(torch.device(args.gpu))
        feat = args.lambda_1 * model_graph(subgraph) + args.lambda_2 * model_seq(subgraph)
        emb_list.append(feat)
    emb = torch.cat(emb_list)
    user_emb, item_emb = emb[0:args.ori_n_users], emb[args.ori_n_users:]
    loss2 = atk.run(item_emb, user_emb)

    optimizer.zero_grad()
    loss2.backward()
    optimizer.step()

    if args.moco:
        moment_update(model_graph, model_graph_ema, args.alpha)
        moment_update(model_seq, model_seq_ema, args.alpha)
    print("[train-in-B] [{}]\t loss2: {:.2f}".format(epoch, loss2))

    print("[pre-train][{}]\t loss1: {:.2f}\t loss2: {:.2f}".format(epoch, loss1_total, loss2))
    return loss1_total, loss2


def main(args):
    random.seed(args.seed)
    dgl.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.is_load:
        if os.path.isfile(args.path_load_model):
            print("[pre-train]=> loading checkpoint '{}'".format(args.path_load_model))
            checkpoint = torch.load(args.path_load_model)
            pretrain_args = checkpoint["opt"]
            pretrain_args.gpu = args.gpu
            pretrain_args.dataset = args.dataset
            pretrain_args.target_dataset = args.target_dataset
            pretrain_args.target_item = args.target_item
            pretrain_args.model_type = args.model_type
            pretrain_args.epochs = args.epochs
            pretrain_args.batch_size = args.batch_size
            pretrain_args.rw_hops = args.rw_hops
            pretrain_args.restart_prob = args.restart_prob
            pretrain_args.target_rw_hops = args.target_rw_hops
            pretrain_args.target_restart_prob = args.target_restart_prob
            pretrain_args.nce_k = args.nce_k
            pretrain_args.is_load = args.is_load
            pretrain_args.path_load_model = args.path_load_model
            args = pretrain_args
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    args = args_update(args)
    print('[PRETRIAN ARGS] ', args)

    if args.dataset == "dgl":
        train_dataset = LoadBalanceGraphDataset(
            rw_hops=256,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
            num_workers=args.num_workers,
            num_copies=args.num_copies,
            num_samples=args.num_samples,
            dgl_graphs_file="data/dgl_graph.bin",
        )
    else:
        train_dataset = RecDataset(
            rw_hops=args.rw_hops,
            dataset=args.dataset,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
        )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=batcher(),
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=None if args.dataset != "dgl" else worker_init_fn,
    )

    target_train_dataset = RecDataset(
        rw_hops=args.target_rw_hops,
        dataset=args.target_dataset,
        restart_prob=args.target_restart_prob,
        positional_embedding_size=args.positional_embedding_size,
    )
    target_train_loader = torch.utils.data.DataLoader(
        dataset=target_train_dataset,
        batch_size=len(target_train_dataset),
        collate_fn=batcher(),
        shuffle=False,
        num_workers=args.num_workers,
    )

    # create model and optimizer
    model_graph, model_graph_ema = [
        GraphEncoder(
            positional_embedding_size=args.positional_embedding_size,
            max_degree=args.max_degree,
            degree_embedding_size=args.degree_embedding_size,
            output_dim=args.hidden_size,
            node_hidden_dim=args.hidden_size,
            num_layers=args.num_layer,
            norm=args.norm,
            gnn_model='gin',
            degree_input=True,
        ).cuda(args.gpu)
        for _ in range(2)
    ]
    model_seq, model_seq_ema = [
        SeqEncoder(
            positional_embedding_size=args.positional_embedding_size,
            max_degree=args.max_degree,
            degree_embedding_size=args.degree_embedding_size,
            hidden_size=args.hidden_size,
            num_layers=2,
            degree_input=True,
        ).cuda(args.gpu)
        for _ in range(2)
    ]

    if args.moco:
        moment_update(model_graph, model_graph_ema, 0)
        moment_update(model_seq, model_seq_ema, 0)

    contrast = MemoryMoCo(args.hidden_size, args.nce_k, args.nce_t, use_softmax=True).cuda(args.gpu)
    atk = Attacker(args.sub_target_dataset, args.target_item, args.gpu, args.hidden_size,
                   lambda_item=args.lambda_item, lambda_user=args.lambda_user)

    criterion = NCESoftmaxLoss() if args.moco else NCESoftmaxLossNS()
    criterion = criterion.cuda(args.gpu)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            itertools.chain(model_graph.parameters(), model_seq.parameters()),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=1e-5,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            itertools.chain(model_graph.parameters(), model_seq.parameters()),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-5,
        )
    elif args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(
            itertools.chain(model_graph.parameters(), model_seq.parameters()),
            lr=args.learning_rate,
            lr_decay=0.0,
            weight_decay=1e-5,
        )
    else:
        raise NotImplementedError

    if args.is_load:
        model_graph.load_state_dict(checkpoint["model_graph"])
        model_seq.load_state_dict(checkpoint["model_seq"])
        contrast.load_state_dict(checkpoint["contrast"])
        if args.moco:
            model_graph_ema.load_state_dict(checkpoint["model_graph_ema"])
            model_seq_ema.load_state_dict(checkpoint["model_seq_ema"])
        del checkpoint
        torch.cuda.empty_cache()

    loss_save_path = os.path.join(args.result_graph_save_path, 'model_loss.txt')
    f = open(loss_save_path, mode='w')
    print("==> pre-training...")
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(epoch, args, optimizer)

        time1 = time.time()
        loss_list = train_moco(
            epoch,
            train_loader,
            target_train_loader,
            model_graph,
            model_graph_ema,
            model_seq,
            model_seq_ema,
            contrast,
            atk,
            criterion,
            optimizer,
            args,
        )

        line = '\t'.join(
            [str(epoch), str(loss_list[0].item()), str(loss_list[1].item())]) + '\n'
        f.write(line)

        time2 = time.time()
        print("pre-train epoch {}, total time {:.2f}".format(epoch, time2 - time1))

        # save model
        if epoch % args.save_freq == 0:
            print("==> model Saving(epoch=%d)..." % epoch)
            state = {
                "opt": args,
                "model_graph": model_graph.state_dict(),
                "model_seq": model_seq.state_dict(),
                "contrast": contrast.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            if args.moco:
                state["model_graph_ema"] = model_graph_ema.state_dict()
                state["model_seq_ema"] = model_seq_ema.state_dict()

            save_file = os.path.join(args.model_save_dir, "ckpt_epoch_{epoch}.pth".format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state
        torch.cuda.empty_cache()

    # saving the final model
    print("==> model saving(final model)...")
    state = {
        "opt": args,
        "model_graph": model_graph.state_dict(),
        "model_seq": model_seq.state_dict(),
        "contrast": contrast.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": args.epochs+1,
    }
    if args.moco:
        state["model_graph_ema"] = model_graph_ema.state_dict()
        state["model_seq_ema"] = model_seq_ema.state_dict()
    save_file = os.path.join(args.model_save_dir, "model.pth")
    torch.save(state, save_file)

    f.close()
    build_result_graph(loss_save_path, args.result_graph_save_path)


if __name__ == "__main__":
    warnings.simplefilter("once", UserWarning)
    args = parse_args()
    main(args)

