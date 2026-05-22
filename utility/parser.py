import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--data_path', nargs='?', default='', help='Input data path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset from {ml-1m, netflix,movielens_new}')
    parser.add_argument('--verbose', type=int, default=5, help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=1000, help='Number of epoch.')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]', help='Regularizations.')
    parser.add_argument('--embed_size', type=int, default=64, help='Embedding size.')
    parser.add_argument('--weight_size', nargs='?', default='[64, 64]', help='Output sizes of every layer')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='Early Stop Patience')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--Ks', nargs='?', default='[10, 20, 50]', help='K value of ndcg/recall @ k')
    parser.add_argument('--cf_model', nargs='?', default='lightgcn',
                        help='Downstream Collaborative Filtering model {mf, ngcf, lightgcn, mmgcn}')

    # train
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')

    # model
    parser.add_argument('--layers', type=int, default=1, help='Number of graph conv layers')
    parser.add_argument('--user_cat_rate', type=float, default=2.8, help='User cat rate')
    parser.add_argument('--item_cat_rate', type=float, default=0.005, help='Item cat rate')
    parser.add_argument('--model_cat_rate', type=float, default=0.02, help='Model cat rate')

    parser.add_argument('--tau', type=float, default=0.1, help='Temperature for InfoNCE loss')

    return parser.parse_args()
