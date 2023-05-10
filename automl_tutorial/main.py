from automl_tutorial.hpo_search_algos.hyperband_tune_runner import HyperBandRayTuneRunner
from automl_tutorial.hpo_search_algos.hyperopt_tune_runner import HyperoptRayTuneRunner
from automl_tutorial.hpo_search_algos.random_tune_runner import RandomRayTuneRunner
from automl_tutorial.hpo_search_algos.bohb_tune_runner import BOHBRayTuneRunner

from automl_tutorial.utils.search_space_defn import cnn_search_space
from automl_tutorial.utils.train_cnn_model import train_cifar


#HyperBandRayTuneRunner(num_samples=4, search_space=cnn_search_space()).search()                                                                                                                                                         
#HyperoptRayTuneRunner(num_samples=4, search_space=cnn_search_space()).search()                                                                                                                                                          
#RandomRayTuneRunner(num_samples=3, search_space=cnn_search_space()).search()                                                                                                                                                            
BOHBRayTuneRunner(num_samples=4, search_space=cnn_search_space(), trainer_method=train_cifar).search()
