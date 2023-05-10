
from ray import tune

from automl_tutorial.ray_tune_runner import RayTuneRunner
from automl_tutorial.utils.utils import set_seeds

class RandomRayTuneRunner(RayTuneRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tune_instance(self):
        trainable_res = tune.with_resources(self.trainer_method, 
                                            {"gpu": self.gpus_per_trail, "cpu": self.gpus_per_trail})
        set_seeds()
        tuner = tune.Tuner(
            trainable_res,
            tune_config = tune.TuneConfig(
                mode=self.mode,
                metric=self.metric,
                num_samples=self.num_samples),
            param_space=self.search_space)
        return tuner
    
        
