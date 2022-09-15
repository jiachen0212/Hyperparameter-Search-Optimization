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
  type: OpenBoxTuner2
  optimize_mode: maximize
  max_concurrency_trials: 4
  search_space:
    lr:
      type: loguniform
      value: [0.00003, 0.0005] # fixed
    ce_weight:
      type: uniform
      value: [0.1, 2.5] # fixed
    weight_decay:
      type: ordinal
      value: [0.00005, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05] # fixed
    sr_pool:
      type: choice
      value: [True, False] # fixed
    loss_th:
      type: ordinal
      value: [0.0001, 0.05, 0.1, 0.35] # fixed

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
