from .exp import exp_yaml as cfg_template
from .scheduler import scheduler_json as sched_template
from .scheduler import hpo_scheduler_template as hpo_sched_template
from .hpo import hpo_base_yaml, hpo_cluster_trial_yaml, hpo_local_trial_yaml

local_hpo_template = hpo_base_yaml + '\n\n' + hpo_local_trial_yaml
cluster_hpo_template = hpo_base_yaml + '\n\n' + hpo_cluster_trial_yaml
