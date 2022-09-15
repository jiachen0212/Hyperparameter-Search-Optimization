from .exp_base import exp_yaml as cfg_template
from .hpo import hpo_base_yaml, hpo_cluster_trial_yaml, hpo_local_trial_yaml

from .exp_fast import exp_fast_yaml as cfg_template_fast
from .hpo import hpo_fast_yaml, hpo_cost_down_yaml

local_hpo_template = hpo_base_yaml + '\n\n' + hpo_local_trial_yaml
cluster_hpo_template = hpo_base_yaml + '\n\n' + hpo_cluster_trial_yaml

local_hpo_template_fast = hpo_fast_yaml + '\n\n' + hpo_local_trial_yaml
cluster_hpo_template_fast = hpo_fast_yaml + '\n\n' + hpo_cluster_trial_yaml

local_hpo_template_cost_down = hpo_cost_down_yaml + '\n\n' + hpo_local_trial_yaml
cluster_hpo_template_cost_down = hpo_cost_down_yaml + '\n\n' + hpo_cluster_trial_yaml
