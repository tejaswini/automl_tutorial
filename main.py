import argparse

from automl_tutorial.hpo_search_algos.hyperband_tune_runner import HyperBandRayTuneRunner
from automl_tutorial.hpo_search_algos.hyperopt_tune_runner import HyperoptRayTuneRunner
from automl_tutorial.hpo_search_algos.random_tune_runner import RandomRayTuneRunner
from automl_tutorial.hpo_search_algos.bohb_tune_runner import BOHBRayTuneRunner

from automl_tutorial.utils.search_space_defn import cnn_search_space
from automl_tutorial.utils.train_cnn_model import train_cifar


if __name__ == "__main__":
    algo_name_cls_mapping = {'random': RandomRayTuneRunner, 'hyperopt': HyperoptRayTuneRunner,
                             'hyperband': HyperBandRayTuneRunner, 'bohb':BOHBRayTuneRunner}
    search_algos = list(algo_name_cls_mapping.keys())
    search_spaces = ['cnn_search_space']
    mode = ['min', 'max']
    parser = argparse.ArgumentParser(
                        description='Code accompanying the AutoML ODSC tutorial')
    parser.add_argument('--search-algo-name', choices=search_algos,
                    help='--select a search algorithm from the list', required=True)
    parser.add_argument('--search-space', choices=search_spaces,
                        help='select a search space from the list', required=True)
    parser.add_argument('--working-dir', help='working dir',
                         required=True)
    parser.add_argument('--results-file-name', help='path where the results of the search are written',
                        default='/foo/bar/results.csv', required=True)
    parser.add_argument('--num-samples', type=int, help='number of trials', default='50')
    parser.add_argument('--num-gpus', type=int, help='number of gpus per trial', default='1')
    parser.add_argument('--num-cpus', type=int, help='number of cpus per trial', default=2)
    parser.add_argument('--mode', help='minimize or maximize the objective function', default='max', choices=mode)
    parser.add_argument('--metric', help='the objective value name in the results', default='accuracy')
    args = parser.parse_args()
    if args.search_space == 'cnn_search_space':
        trainer_method = train_cifar
        search_space = cnn_search_space()
    tune_runner = algo_name_cls_mapping[args.search_algo_name](args.num_samples, args.num_gpus, args.num_cpus,
                                                 args.results_file_name, search_space,
                                                 trainer_method, args.mode, args.metric, args.working_dir)
    tune_runner.search()

    
    
                        
    