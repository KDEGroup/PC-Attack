import os
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dgl
import random
import shutil


from utils.graph_data_load import RecDataset
from utils.graph_data_util import batcher
from models.pretrain.graph_encoder import GraphEncoder
from utils.data_load import Data
from models.attacker.attacker import Attacker
from models.pretrain.seq_encoder import SeqEncoder


def test_moco(train_loader, model_graph, model_seq, opt):
    model_graph.eval()
    model_seq.eval()

    emb_list = []
    for idx, batch in enumerate(train_loader):
        graph_q, graph_k = batch
        graph_q.to(opt.device)
        graph_k.to(opt.device)

        with torch.no_grad():
            feat_q = (model_graph(graph_q) + model_seq(graph_q)) / 2
            feat_k = (model_graph(graph_k) + model_seq(graph_k)) / 2

        emb_list.append(((feat_q + feat_k) / 2).detach().cpu())
    return torch.cat(emb_list)


def build_result_graph(path_load, path_save, scope=5, num=2, name='atk'):
    data = np.loadtxt(path_load, delimiter='\t')
    labels = ['Precision', 'Recall', 'NDCG']

    fig = plt.figure()
    plt.subplot(1, 1, 1)
    if name == 'rec':
        for i in range(1, num):
            plt.plot(data[:, 0], data[:, i], label=labels[i-1]+'@'+str(scope))
    elif name == 'atk':
        plt.plot(data[:, 0], data[:, 1], label='HR@50')
    else:
        plt.plot(data[:, 0], data[:, 1], label='HR@50')
        plt.plot(data[:, 0], data[:, 2], label='Recall')

    plt.legend()
    plt.xlabel(u'epoch')
    plt.ylabel(u'%s_indicator' % name)
    plt.savefig(os.path.join(path_save, '{}_indicator_@{}').format(name, scope))


def generate_item_embedding(args_test):
    result_save_folder = os.path.join(args_test.result_dir, 'attacker')
    if not os.path.exists(result_save_folder):
        os.makedirs(result_save_folder)

    if args_test.result_path:
        result_save_folder = args_test.result_path

    result_save_path = os.path.join(result_save_folder, 'item_embedding')
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    print('[Generate ARGS]:', args_test)
    if os.path.isfile(args_test.load_path):
        checkpoint = torch.load(args_test.load_path, map_location="cpu")
        print("[pre-train] => loaded successfully '{}' (epoch {})".format(args_test.load_path, checkpoint["epoch"]))
    else:
        print("=> no checkpoint found at '{}'".format(args_test.load_path))
    args = checkpoint["opt"]

    assert args_test.gpu is None or torch.cuda.is_available()
    args.gpu = args_test.gpu
    args.device = torch.device("cpu") if args.gpu is None else torch.device(args.gpu)

    train_dataset = RecDataset(
        dataset=args_test.dataset,
        restart_prob=args.restart_prob,
        positional_embedding_size=args.positional_embedding_size,
    )

    args.batch_size = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=batcher(),
        shuffle=False,
        num_workers=args.num_workers,
    )

    model_graph = GraphEncoder(
        positional_embedding_size=args.positional_embedding_size,
        max_degree=args.max_degree,
        degree_embedding_size=args.degree_embedding_size,
        output_dim=args.hidden_size,
        node_hidden_dim=args.hidden_size,
        num_layers=args.num_layer,
        gnn_model='gin',
        norm=args.norm,
        degree_input=True,
    ).to(args.device)

    model_seq = SeqEncoder(
        positional_embedding_size=args.positional_embedding_size,
        max_degree=args.max_degree,
        degree_embedding_size=args.degree_embedding_size,
        hidden_size=args.hidden_size,
        num_layers=2,
        degree_input=True,
    ).to(args.device)

    model_graph.load_state_dict(checkpoint["model_graph"])
    model_seq.load_state_dict(checkpoint["model_seq"])
    del checkpoint

    emb = test_moco(train_loader, model_graph, model_seq, args)
    file_save_path = os.path.join(result_save_path, '{}_{}.npy'.format(args_test.dataset, args_test.target_item))
    np.save(file_save_path, emb.numpy())

    c = Attacker(args_test.sub_dataset, args_test.target_item, args_test.gpu, path_atk_emb=file_save_path)
    rec_atk_results = c.fit()
    matrix_file_path = './results/fake_matrix/%s/fake_matrix_%s_%d.npz' \
                       % (args_test.dataset, args_test.dataset, args_test.target_item)
    data_file_path = './results/fake_data/%s/%s_attacker_%d.data' \
                     % (args_test.dataset, args_test.dataset, args_test.target_item)

    injected_matrix_path = './results/fake_matrix/%s/fake_matrix_%s_to_%s_%d.npz' \
                           % (args_test.dataset, args_test.src_dataset, args_test.sub_dataset, args_test.target_item)
    injected_data_path = './results/fake_data/%s/fake_data_%s_to_%s_%d.data' \
                         % (args_test.dataset, args_test.src_dataset, args_test.sub_dataset, args_test.target_item)
    if os.path.exists(injected_matrix_path):
        os.remove(injected_matrix_path)
    if os.path.exists(injected_data_path):
        os.remove(injected_data_path)
    shutil.copyfile(matrix_file_path, injected_matrix_path)
    shutil.copyfile(data_file_path, injected_data_path)


def generate_and_attack(args_test):
    result_save_folder = os.path.join(args_test.result_dir, 'attacker')
    if not os.path.exists(result_save_folder):
        os.makedirs(result_save_folder)

    if args_test.result_path:
        result_save_folder = args_test.result_path

    result_save_path = os.path.join(result_save_folder, 'item_embedding')
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    result_graph_save_path = os.path.join(result_save_folder, 'diagram')
    if not os.path.exists(result_graph_save_path):
        os.makedirs(result_graph_save_path)

    rec_result_save_path_5 = os.path.join(result_graph_save_path, 'rec_result_5.txt')
    rec_result_save_path_10 = os.path.join(result_graph_save_path, 'rec_result_10.txt')
    rec_result_save_path_20 = os.path.join(result_graph_save_path, 'rec_result_20.txt')
    rec_result_save_path_50 = os.path.join(result_graph_save_path, 'rec_result_50.txt')

    atk_result_save_path = os.path.join(result_graph_save_path, 'atk_result.txt')
    rec_atk_result_save_path = os.path.join(result_graph_save_path, 'rec_atk_result.txt')

    f_5 = open(rec_result_save_path_5, mode='w')
    f_10 = open(rec_result_save_path_10, mode='w')
    f_20 = open(rec_result_save_path_20, mode='w')
    f_50 = open(rec_result_save_path_50, mode='w')
    f_atk = open(atk_result_save_path, mode='w')
    f = open(rec_atk_result_save_path, mode='w')

    best_rst = 0.0
    for idx in range(1, args_test.epochs + 1):
        if idx % args_test.gen_freq == 0:
            args_test.load_path = os.path.join(args_test.load_dir, 'ckpt_epoch_{}.pth').format(idx)

            if os.path.isfile(args_test.load_path):
                checkpoint = torch.load(args_test.load_path, map_location="cpu")
                print("[pre-train] => loaded successfully '{}' (epoch {})".format(args_test.load_path, checkpoint["epoch"]))
            else:
                print("=> no checkpoint found at '{}'".format(args_test.load_path))
            args = checkpoint["opt"]

            assert args_test.gpu is None or torch.cuda.is_available()
            args.gpu = args_test.gpu
            args.device = torch.device("cpu") if args.gpu is None else torch.device(args.gpu)

            train_dataset = RecDataset(
                rw_hops=args_test.rw_hops,
                dataset=args_test.dataset,
                restart_prob=args_test.restart_prob,
                positional_embedding_size=args.positional_embedding_size,
            )
            args.batch_size = len(train_dataset)
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=args.batch_size,
                collate_fn=batcher(),
                shuffle=False,
                num_workers=args.num_workers,
            )

            model_graph = GraphEncoder(
                positional_embedding_size=args.positional_embedding_size,
                max_degree=args.max_degree,
                degree_embedding_size=args.degree_embedding_size,
                output_dim=args.hidden_size,
                node_hidden_dim=args.hidden_size,
                num_layers=args.num_layer,
                gnn_model='gin',
                norm=args.norm,
                degree_input=True,
            ).to(args.device)

            model_seq = SeqEncoder(
                    positional_embedding_size=args.positional_embedding_size,
                    max_degree=args.max_degree,
                    degree_embedding_size=args.degree_embedding_size,
                    hidden_size=args.hidden_size,
                    num_layers=2,
                    degree_input=True,
                ).to(args.device)

            model_graph.load_state_dict(checkpoint["model_graph"])
            model_seq.load_state_dict(checkpoint["model_seq"])
            del checkpoint
            print('[Generate ARGS]:', args_test, args)

            emb = test_moco(train_loader, model_graph, model_seq, args)
            file_save_path = os.path.join(result_save_path, '{}_epoch_{}.npy').format(args_test.dataset, idx)
            np.save(file_save_path, emb.numpy())

            c = Attacker(args_test.sub_dataset, args_test.target_item, args_test.gpu, path_atk_emb=file_save_path)
            rec_atk_results = c.fit()

            rec_result_5, rec_result_10, rec_result_20, rec_result_50 = [], [], [], []
            for k, v in rec_atk_results[0].items():
                if '5' in k:
                    rec_result_5.append(v)
                if '10' in k:
                    rec_result_10.append(v)
                if '20' in k:
                    rec_result_20.append(v)
                if '50' in k:
                    rec_result_50.append(v)
            for k, v in rec_atk_results[1].items():
                if 'TargetHR@50_' in k:
                    atk_result = v
            if atk_result > best_rst:
                best_rst = atk_result
                matrix_file_path = './results/fake_matrix/%s/fake_matrix_%s_%d.npz' \
                                   % (args_test.dataset, args_test.dataset, args_test.target_item)
                data_file_path = './results/fake_data/%s/%s_attacker_%d.data' \
                                 % (args_test.dataset, args_test.dataset, args_test.target_item)

                injected_matrix_path = './results/fake_matrix/%s/best_fake_matrix_%s_to_%s_%d.npz' \
                                       % (args_test.dataset, args_test.src_dataset, args_test.sub_dataset, args_test.target_item)
                injected_data_path = './results/fake_data/%s/best_fake_data_%s_to_%s_%d.data' \
                                     % (args_test.dataset, args_test.src_dataset, args_test.sub_dataset, args_test.target_item)
                if os.path.exists(injected_matrix_path):
                    os.remove(injected_matrix_path)
                if os.path.exists(injected_data_path):
                    os.remove(injected_data_path)
                shutil.copyfile(matrix_file_path, injected_matrix_path)
                shutil.copyfile(data_file_path, injected_data_path)

            line1 = '\t'.join([str(idx), str(rec_result_5[0]), str(rec_result_5[1]),
                               str(rec_result_5[2])]) + '\n'
            line2 = '\t'.join([str(idx), str(rec_result_10[0]), str(rec_result_10[1]),
                               str(rec_result_10[2])]) + '\n'
            line3 = '\t'.join([str(idx), str(rec_result_20[0]), str(rec_result_20[1]),
                               str(rec_result_20[2])]) + '\n'
            line4 = '\t'.join([str(idx), str(rec_result_50[0]), str(rec_result_50[1]),
                               str(rec_result_50[2])]) + '\n'
            line5 = '\t'.join([str(idx), str(atk_result)]) + '\n'

            line6 = '\t'.join([str(idx), str(atk_result), str(rec_result_50[1])]) + '\n'

            f_5.write(line1)
            f_10.write(line2)
            f_20.write(line3)
            f_50.write(line4)
            f_atk.write(line5)
            f.write(line6)

    f_5.close()
    f_10.close()
    f_20.close()
    f_50.close()
    f_atk.close()
    f.close()

    build_result_graph(rec_result_save_path_5, result_graph_save_path, 5, 4, name='rec')
    build_result_graph(rec_result_save_path_10, result_graph_save_path, 10, 4, name='rec')
    build_result_graph(rec_result_save_path_20, result_graph_save_path, 20, 4, name='rec')
    build_result_graph(rec_result_save_path_50, result_graph_save_path, 50, 4, name='rec')
    build_result_graph(atk_result_save_path, result_graph_save_path, 50, 2, name='atk')
    build_result_graph(rec_atk_result_save_path, result_graph_save_path, 50, 3, name='rec_atk')

    print('HR@50: ', best_rst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--load-dir", default='./saved/pretrain/filmtrust/t[5]_w1[250]_w2[1_lr[0.005]]', type=str, help="path to load model")
    parser.add_argument("--load-path", type=str, help="path to load model")
    parser.add_argument("--result-dir", type=str, default='results/', help="path to save result")
    parser.add_argument("--result-path", type=str, default='', help="path to save result")

    parser.add_argument("--dataset", type=str, default="dgl")
    parser.add_argument("--target-item", type=int, default=5)
    parser.add_argument("--rw-hops", type=int, default=256)
    parser.add_argument("--restart-prob", type=float, default=0.8)

    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    parser.add_argument("--seed", type=int, default=1234, help="random seed.")

    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--gen-freq", type=int, default=1, help="print frequency")
    parser.add_argument("--gen-only", type=int, default=0)
    parser.add_argument("--src-dataset", type=str, default="filmtrust")
    args = parser.parse_args()

    args.sub_dataset = args.dataset
    args.dataset = args.dataset.split('_')[0]
    path_train = './data/' + args.dataset + '/preprocess/train.data'
    path_test = './data/' + args.dataset + '/preprocess/test.data'

    dataset_class = Data(path_train, path_test, test_bool=True, header=['user_id', 'item_id', 'rating', 'timestamp'],
                         sep='\t', type='generate')
    _, _, args.ori_n_users, args.ori_n_items = dataset_class.load_file_as_dataFrame()

    random.seed(args.seed)
    dgl.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.gen_only:
        generate_item_embedding(args)
    else:
        generate_and_attack(args)
