
from ray import tune
from ray.tune.schedulers import HyperBandScheduler

from automl_tutorial.utils.utils import set_seeds
from automl_tutorial.ray_tune_runner import RayTuneRunner

class HyperBandRayTuneRunner(RayTuneRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def tune_instance(self):
        trainable_res = tune.with_resources(self.trainer_method,
                                            {"gpu": self.gpus_per_trail, "cpu": self.gpus_per_trail})
        hyperband = HyperBandScheduler(time_attr="training_iteration", max_t=10,
                                       reduction_factor=3,
                                       metric=self.metric, mode=self.mode)

        set_seeds()
        tuner = tune.Tuner(
            trainable_res,
            tune_config = tune.TuneConfig(
                scheduler=hyperband,
                num_samples=self.num_samples),
            param_space=self.search_space)
        return tuner

    
        
