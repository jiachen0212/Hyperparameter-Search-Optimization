hpo_local_yaml = \
    '''
common:
  log_level: INFO
  seed: 42  # change to time
  deterministic: False
  implement_layer: []
  plugin_layer: [SMore_hpo]

manager:
  type: SimpleManager
  max_trials: 40
  max_concurrency_trials: 4
  update_trial_clock: 10.0
  base_exp_path: ./exp_config/exp.yaml # dynamic

trial:
  type: SMapLocalDLTrial
  metrics:
    - data_name: eval
      metric_type:  "ProductMultiLabelClassificationEvaluator/overall_index/recall"
      metric_weight: 1.0
  resource:
    total_gpus: 4
    gpus_per_trial: 1

assessor:
  type: DummyAssessor

tuner:
  type: DummyTuner
  search_space:
    lr:
      type: choice
      value: [0.002, 0.005, 0.01, 0.02, 0.05]  # fix
    target_lr:
      type: choice
      value: [0.0001, 5.0e-05, 1.0e-05]  # fix
    momentum:
      type: choice
      value: [0.8, 0.9, 0.95]  # fix
    weight_decay:
      type: choice
      value: [0.00001, 0.0001, 0.0002, 0.0005, 0.001]  # fix
    warmup_iter:
      type: choice
      value: [0, 25, 50]  # fix
    rotation_prob:
      type: choice
      value: [0.0, 0.5, 1.0]  # fix
    rotation_angle:
      type: choice
      value: [[-0.0, 0.0], [-5.0, 5.0], [-15.0, 15.0], [-45.0, 45.0]]  # fix
    horizontal_flip_prob:
      type: choice
      value: [0.0, 0.5]
    vertical_flip_prob:
      type: choice
      value: [0.0, 0.5]
    randomnoise_size:
      type: choice
      value: [0, 5, 10, 15, 20]  # fix
    randomcrop_pad:
      type: choice
      value: [0, 4, 8, 16]  # fix
    out_indices:
      type: choice
      value: [[2,], [3,]]
    neck_type:
      type: choice
      value: ['GlobalAveragePooling', 'GlobalMaxPooling']
'''
