from abc import ABCMeta, abstractmethod


class BaselineBase(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def gen_cfgs(self, baseline_yaml_path: str, hpo_path: str, rerun_best_trial=False):
        '''
        Args:
            baseline_yaml_path: path of baseline.yaml
            hpo_path: path of hpo experiments, include hpo.yaml, trials folder result.csv and so on.
            rerun_best_trial: After hpo, re-run the best trial using all cards
        '''
        raise NotImplementedError('Baseline not implemented')
