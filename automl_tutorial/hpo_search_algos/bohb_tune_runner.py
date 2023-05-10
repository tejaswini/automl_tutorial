
from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search.concurrency_limiter import ConcurrencyLimiter

from automl_tutorial.utils.utils import set_seeds
from automl_tutorial.ray_tune_runner import RayTuneRunner

class BOHBRayTuneRunner(RayTuneRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def tune_instance(self):
        trainable_res = tune.with_resources(self.trainer_method,
                                            {"gpu": self.gpus_per_trail, "cpu": self.gpus_per_trail})
        bohb_algo = TuneBOHB()
        bohb_algo = ConcurrencyLimiter(bohb_algo, max_concurrent=4)
        
        hyperband = HyperBandForBOHB(time_attr="training_iteration", max_t=10, reduction_factor=3)

        set_seeds()
        tuner = tune.Tuner(
            trainable_res,
            tune_config = tune.TuneConfig(
                metric=self.metric,
                mode=self.mode,
                search_alg=bohb_algo,
                scheduler=hyperband,
                num_samples=self.num_samples),
            param_space=self.search_space)
        return tuner
