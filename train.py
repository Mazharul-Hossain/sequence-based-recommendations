from __future__ import print_function

import sys

import numpy as np

import helpers.command_parser as parse
from helpers import early_stopping
from helpers.data_handling import DataHandler


def training_command_parser(parser):
    parser.add_argument('--tshuffle', help='Shuffle sequences during training.', action='store_true')

    parser.add_argument('--extended_set',
                        help='Use extended training set (contains first half of validation and test set).',
                        action='store_true')

    parser.add_argument('-d', dest='dataset', help='Directory name of the dataset.', default='', type=str)
    parser.add_argument('--dir', help='Directory name to save model.', default='', type=str)
    parser.add_argument('--save', choices=['All', 'Best', 'None'], help='Policy for saving models.', default='Best')
    parser.add_argument('--metrics', help='Metrics for validation, comma separated', default='sps', type=str)
    parser.add_argument('--time_based_progress', help='Follow progress based on time rather than iterations.',
                        action='store_true')
    parser.add_argument('--load_last_model', help='Load Last model before starting training.', action='store_true')
    parser.add_argument('--progress', help='Progress intervals', default='2.', type=str)
    parser.add_argument('--mpi', help='Max progress intervals', default=np.inf, type=float)
    parser.add_argument('--max_iter', help='Max number of iterations', default=np.inf, type=float)
    parser.add_argument('--max_time', help='Max training time in seconds', default=np.inf, type=float)
    parser.add_argument('--min_iter', help='Min number of iterations before showing progress', default=0., type=float)


def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def main():
    sys.argv.extend(['--tshuffle', '--load_last_model',  # '--extended_set',
                     '-d', 'datasets/',
                     '--save', 'Best',
                     '--progress', '200', '--mpi', '1000.0',
                     '--max_iter', '6000.0', '--max_time', '28800.0', '--min_iter', '100.0',
                     '--es_m', 'StopAfterN', '--es_n', '3',
                     '-m', 'RNN', '--r_t', 'GRU', '--r_l', '100-50',
                     '--u_m', 'rmsprop',
                     '--rf'])
    # ####################################################
    # # for RNNCluster
    # sys.argv.extend(['--dir', 'RNNCluster_',
    #                  '--metrics', 'recall,cluster_recall,sps,cluster_sps,ignored_items,assr',
    #                  '--loss', 'BPR', '--clusters', '10'])
    # ####################################################
    # for RNNOneHot
    sys.argv.extend(['--dir', 'RNNOneHot_',
                     '--metrics', 'recall,sps',  # ,ndcg,item_coverage,user_coverage,blockbuster_share
                     '--loss', 'CCE'])
    # ####################################################
    # # for RNNMargin
    # sys.argv.extend(['--dir', 'RNNMargin_',
    #                  '--metrics', 'recall,sps',
    #                  '--loss', 'logit'])
    # ####################################################
    # # for RNNSampling
    # sys.argv.extend(['--dir', 'RNNSampling_',
    #                  '--metrics', 'recall,sps',
    #                  '--loss', 'BPR'])
    # ####################################################
    # # without MOVIES_FEATURES
    # sys.argv.extend(['--r_emb', '100'])
    # # with MOVIES_FEATURES
    sys.argv.extend(['--mf'])
    # ####################################################

    args = parse.command_parser(parse.predictor_command_parser, training_command_parser,
                                early_stopping.early_stopping_command_parser)

    predictor = parse.get_predictor(args)

    dataset = DataHandler(dirname=args.dataset, extended_training_set=args.extended_set, shuffle_training=args.tshuffle)

    if args.mf:
        predictor.load_movies_features(dirname=dataset.dirname)

    predictor.prepare_model(dataset)
    predictor.train(dataset,
                    save_dir=dataset.dirname + "models/" + args.dir,
                    time_based_progress=args.time_based_progress,
                    progress=num(args.progress),
                    autosave=args.save,
                    max_progress_interval=args.mpi,
                    max_iter=args.max_iter,
                    min_iterations=args.min_iter,
                    max_time=args.max_time,
                    early_stopping=early_stopping.get_early_stopper(args),
                    load_last_model=args.load_last_model,
                    validation_metrics=args.metrics.split(','))


if __name__ == '__main__':
    main()
