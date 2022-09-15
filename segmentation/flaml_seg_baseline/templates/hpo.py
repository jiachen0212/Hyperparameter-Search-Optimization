hpo_base_yaml = \
    '''
common:
  log_level: INFO
  seed: 0 # change to time
  deterministic: False
  implement_layer: []
  plugin_layer: [SMore_hpo]


manager:
  type: SimpleManager
  max_trials: 20
  max_concurrency_trials: 4
  update_trail_clock: 10
  base_exp_path: ./exp_config/exp.yaml # dynamic
  rerun_best_trial: False

assessor:
  type: DummyAssessor

tuner:
  type: FLAMLTuner
  max_concurrency_trials: 4
  prune_attr: n_resources
  min_resource: 1
  max_resource: 4
  reduction_factor: 2
  max_trials: 20
#  optimize_mode: maximize
#  optimize_batch: 4
  search_space:
    lr:
      type: loguniform
      value: [0.0001, 0.001] # dynamic
    ce_weight:
      type: uniform
      value: [0.0, 2.0] # fixed
    rotate_angle:
      type: choice
      value: [[0,0], [-30,30], [-180,180]] # fixed
'''

hpo_local_trial_yaml = \
    '''
trial:
  type: SMapLocalDLTrial
  prune_attr: n_resources
  iter_per_resource: 0
  metrics:
    - data_name: val
      metric_type:  "PixelBasedEvaluator/iou/mean"
      metric_weight: 1.0
  resource:
    total_gpus: 8
    gpus_per_trial: 2
'''

hpo_cluster_trial_yaml = \
    '''
trial:
  type: SMapClusterDLTrial
  schedctl: latest
  prune_attr: n_resources
  sleep_after_start: 30
  sleep_after_describe: 60
  metrics:
    - data_name: val
      metric_type:  "PixelBasedEvaluator/iou/mean"
      metric_weight: 1.0
'''
