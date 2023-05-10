import os

from ray import tune
import ray

from abc import ABC, abstractmethod

class RayTuneRunner(ABC):
    def __init__(self, num_samples=10, gpus_per_trial=2,
                 cpus_per_trail=1, result_file_name="results.csv",
                 search_space=None, trainer_method=None,
                 mode="max", metric="accuracy",
                 working_dir=""):
        self.num_samples = num_samples
        self.gpus_per_trial = gpus_per_trial
        self.cpus_per_trail = cpus_per_trail
        self.results_file_name = result_file_name
        self.search_space = search_space
        self.trainer_method = trainer_method
        self.mode = mode
        self.metric = metric
        self.working_dir = working_dir

    @abstractmethod    
    def tune_instance():
        raise NotImplementedError

    def search(self):
        ray.init(ignore_reinit_error=True, runtime_env={"working_dir": self.working_dir})
        tuner = self.tune_instance()
        print(tuner)

        result = tuner.fit()
        df = result.get_dataframe()
        df.to_csv(self.results_file_name, index=False)
