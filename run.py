import argparse
import os
PythonCommand = 'python'


class Run:
    def __init__(self):
        self.args = self.parse_args()

    def parse_args(self):
        parser = argparse.ArgumentParser("argument for training")

        parser.add_argument("--dataset", type=str, default="filmtrust")
        parser.add_argument("--epochs", type=int, default=64, help="number of training epochs")
        parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")

        parser.add_argument("--target-item", type=int, default=5)
        parser.add_argument("--target-dataset", type=str, default="filmtrust")
        parser.add_argument("--target-restart-prob", type=float, default=0.8)
        parser.add_argument("--target-rw-hops", type=int, default=64)
        # pre-train
        parser.add_argument("--learning-rate", type=float, default=0.01, help="learning rate")
        parser.add_argument("--model-path", type=str, default='saved', help="path to save model")
        parser.add_argument("--model-type", type=str, default='attacker')
        parser.add_argument("--nce-k", type=int, default=256)
        parser.add_argument("--batch-size", type=int, default=32, help="batch_size")
        parser.add_argument("--rw-hops", type=int, default=64)
        parser.add_argument("--restart-prob", type=float, default=0.8)
        # generate
        parser.add_argument("--load-path", default='', type=str, help="path to load model")
        parser.add_argument("--gen-freq", type=int, default=1, help="print frequency")
        parser.add_argument("--gen-only", type=int, default=0)
        parser.add_argument("--result-path", type=str, default='', help="path to save result")

        parser.add_argument("--lambda_g", type=float, default=0.5)
        parser.add_argument("--lambda_s", type=float, default=0.5)
        parser.add_argument("--lambda_1", type=float, default=0.5)
        parser.add_argument("--lambda_2", type=float, default=0.5)
        parser.add_argument("--lambda_item", type=float, default=0.5)
        parser.add_argument("--lambda_user", type=float, default=0.5)

        args = parser.parse_args()
        return args

    def execute(self, args):
        # pre-train.
        pretrain_args_dict = {'dataset': args.dataset,
                              'rw-hops': args.rw_hops,
                              'restart-prob': args.restart_prob,
                              'gpu': args.gpu,
                              'epochs': args.epochs,
                              'target-item': args.target_item,
                              'target-dataset': args.target_dataset,
                              'target-rw-hops': args.target_rw_hops,
                              'target-restart-prob': args.target_restart_prob,
                              'learning-rate': args.learning_rate,
                              'nce-k': args.nce_k,
                              'batch-size': args.batch_size,
                              "lambda_g": args.lambda_g,
                              "lambda_s": args.lambda_s,
                              "lambda_1": args.lambda_1,
                              "lambda_2": args.lambda_2,
                              "lambda_user": args.lambda_user,
                              "lambda_item": args.lambda_item,
                              }

        pretrain_args_str = ' '.join(["--%s %s" % (k, v) for (k, v) in pretrain_args_dict.items()])
        print(os.system('%s ./train.py %s' % (PythonCommand, pretrain_args_str)))

        # generate
        generate_args_dict = {'dataset': args.target_dataset,
                              'target-item': args.target_item,
                              'epochs': args.epochs,
                              'gpu': args.gpu,
                              'gen-freq': args.gen_freq,
                              'gen-only': args.gen_only,
                              'rw-hops': args.target_rw_hops,
                              'restart-prob': args.target_restart_prob,
                              "src-dataset": args.dataset
                              }

        model_name = "{}_to_{}_{}".format(args.dataset, args.target_dataset, args.target_item)
        model_dir = os.path.join(args.model_path, args.model_type, model_name)
        model_path = os.path.join(model_dir, 'model.pth')
        if args.gen_only:
            generate_args_dict['load-path'] = model_path
        else:
            generate_args_dict['load-dir'] = model_dir
            generate_args_dict['result-path'] = 'results/attacker/{}'.format(model_name)
        generate_args_str = ' '.join(["--%s %s" % (k, v) for (k, v) in generate_args_dict.items()])
        print(os.system('%s ./generate.py %s' % (PythonCommand, generate_args_str)))


if __name__ == "__main__":
    model = Run()
    model.execute(model.args)



