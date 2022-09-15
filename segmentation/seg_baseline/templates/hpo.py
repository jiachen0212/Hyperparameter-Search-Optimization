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
  update_trail_clock: 60
  base_exp_path: ./exp_config/exp.yaml # dynamic
  rerun_best_trial: False

assessor:
  type: DummyAssessor

tuner:
  type: HEBOTuner
  optimize_mode: maximize
  optimize_batch: 4
  search_space:
    lr:
      type: loguniform
      value: [0.0001, 0.001] # dynamic
    ce_weight:
      type: uniform
      value: [0.1, 2.5] # fixed
    rotate_angle:
      type: choice
      value: [[0,0], [-180,180]] # fixed
    loss_th:
      type: choice
      value: [0.001, 0.05, 0.3] # fixed
'''

hpo_local_trial_yaml = \
    '''
trial:
  type: SMapLocalDLTrial
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
  sleep_after_start: 30
  sleep_after_describe: 60
  metrics:
    - data_name: val
      metric_type:  "PixelBasedEvaluator/iou/mean"
      metric_weight: 1.0
'''

hpo_fast_yaml = \
    '''
common:
  log_level: INFO
  seed: 0 # change to time
  deterministic: False
  implement_layer: []
  plugin_layer: [SMore_hpo]


manager:
  type: SimpleManager
  max_trials: 6
  max_concurrency_trials: 6
  update_trail_clock: 60
  base_exp_path: ./exp_config/exp.yaml # dynamic
  rerun_best_trial: False

assessor:
  type: DummyAssessor

tuner:
  type: GridSearchTuner
  optimize_mode: maximize
  search_space:
    lr:
      type: choice
      value: [0.0005, 0.0015, 0.0045] # dynamic
    ce_weight:
      type: choice
      value: [0.8, 4.0] # fixed
'''

hpo_cost_down_yaml = \
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
  update_trail_clock: 60
  base_exp_path: ./exp_config/exp.yaml # dynamic
  rerun_best_trial: False

assessor:
  type: MedianStopAssessor
  start_step: 2
  min_trial: 8
  optimize_mode: maximize

tuner:
  type: OpenBoxTuner2
  optimize_mode: maximize
  max_concurrency_trials: 4
  search_space:
    lr:
      type: loguniform
      value: [0.0001, 0.001] # dynamic
    ce_weight:
      type: uniform
      value: [0.1, 2.5] # fixed
    rotate_angle:
      type: ordinal
      value: [[0,0], [-10,10], [-180,180]] # fixed
    loss_th:
      type: ordinal
      value: [0.001, 0.05, 0.3] # fixed
'''
