
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.concurrency_limiter import ConcurrencyLimiter

from automl_tutorial.utils.utils import set_seeds
from automl_tutorial.ray_tune_runner import RayTuneRunner

class HyperoptRayTuneRunner(RayTuneRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tune_instance(self):
        trainable_res = tune.with_resources(self.trainer_method,
                                            {"gpu": self.gpus_per_trail, "cpu": self.gpus_per_trail})
        hyperopt_algo = HyperOptSearch(random_state_seed=0)
        hyperopt_algo = ConcurrencyLimiter(hyperopt_algo, max_concurrent=4)

        set_seeds()
        tuner = tune.Tuner(
            trainable_res,
            tune_config = tune.TuneConfig(
                mode=self.mode,
                metric=self.metric,
                search_alg=hyperopt_algo,
                num_samples=self.num_samples),
            param_space=self.search_space)
        return tuner
    
        
