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
  max_trials: 40
  max_concurrency_trials: 4
  update_trail_clock: 10
  base_exp_path: ./exp_config/exp.yaml # dynamic

assessor:
  type: DummyAssessor

tuner:
  type: FLAMLTuner
  max_concurrency_trials: 4
#  algorithm_name: tpe
  search_space:
    lr:
      type: loguniform
      value: [0.00001, 0.01] # fixed
    weight_decay:
      type: loguniform
      value: [0.000001, 0.0005] # fixed
    size_bs_iter:
      type: choice_group
      value: [4000] # dynamic
    regress_ranges:
      type: choice
      value: [[[-1, 16], [16, 32], [32, 64], [64, 128], [128, 9999]],
              [[-1, 32], [32, 64], [64, 128], [128, 256], [256, 9999]],
              [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 9999]],
              [[-1, 128], [128, 256], [256, 512], [512, 1024], [1024, 9999]],
              ]

    '''

hpo_local_trial_yaml = \
    '''
trial:
  type: SMapLocalDLTrial
  metrics:
    - data_name: eval
      metric_type:  "DRPEvaluator/all/Recall"
      metric_weight: 0.5
    - data_name: eval
      metric_type:  "DRPEvaluator/all/Precision"
      metric_weight: 0.5
  resource:
    total_gpus: 8
    gpus_per_trial: 2
    '''

hpo_cluster_trial_yaml = \
    '''
trial:
  type: SMapClusterDLTrial
  schedctl: latest
  sleep_after_start: 60
  sleep_after_describe: 60
  metrics:
    - data_name: eval
      metric_type:  "DRPEvaluator/all/Recall"
      metric_weight: 0.5
    - data_name: eval
      metric_type:  "DRPEvaluator/all/Precision"
      metric_weight: 0.5
    '''
