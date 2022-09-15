exp_yaml = \
    '''
common:
  log_level: INFO
  seed: 0
  deterministic: True
  implement_layer: [SMore_cls]
  plugin_layer: []

other:
  mean: &mean ##place_holder_mean##
  std: &std ##place_holder_std##
  scale: &scale 1.0
  workers: &workers 1
  max_iter: &max_iter ##place_holder_max_iter##
  ckpt_freq: &ckpt_freq ##place_holder_ckpt_freq##
  batch_size: &batch_size 32
  input_size: &input_size ##place_holder_input_size##
  num_classes: &num_classes ##place_holder_num_classes##
  resume: &resume None
  # search_space
  search_space:
    lr: &lr 0.01
    target_lr: &target_lr 0.0001
    momentum: &momentum 0.9
    weight_decay: &weight_decay 0.0005
    warmup_iter: &warmup_iter 0
    rotation_prob: &rotation_prob 1.0
    rotation_angle: &rotation_angle [-45.0,45.0]
    horizontal_flip_prob: &horizontal_flip_prob 0.0
    vertical_flip_prob: &vertical_flip_prob 0.0
    randomnoise_size: &randomnoise_size 10
    randomcrop_pad: &randomcrop_pad 0
    out_indices: &out_indices [3,]
    neck_type: &neck_type GlobalAveragePooling
  ##place_holder_baseline##


data:
  #label_map: *label_map
  train_data:
    batch_size: *batch_size
    workers: *workers
    dataset: *train_set
    transform:
      - &RS
        type: Resize
        output_size: *input_size
      - type: RandomCrop
        padding: *randomcrop_pad
        output_size: *input_size
      - type: RandomRotation
        angle: *rotation_angle
        prob: 1.0
      - type: RandomHorizontalFlip
        prob: *horizontal_flip_prob
      - type: RandomVerticalFlip
        prob: *vertical_flip_prob
      - type: RandomNoise
        size: *randomnoise_size
      - type: SMore_cls::RandomMask
        hval: [2,10,10,10]
      - &normalize
        type: Normalize
        mean: *mean
        std: *std
        scale: *scale
      - &toTensor
        type: ToTensor

  eval_data:
    batch_size: *batch_size
    workers: *workers
    dataset: *val_set
    transform:
      - *RS
      - *normalize
      - *toTensor

  test_data:
    batch_size: *batch_size
    workers: *workers
    dataset: *val_set
    transform:
      - *RS
      - *normalize
      - *toTensor

model:
  type: EncoderDecoder
  backbone:
    ##place_holder_backbone##
    out_indices: *out_indices
  neck:
    type: *neck_type
  head:
    type: FCHead
    num_classes: *num_classes
    post_process_mode: ##place_holder_post_process_mode##
    losses:
      - type: ##place_holder_loss_type##
        num_classes: *num_classes

train:
  type: SimpleTrainer
  max_iter: *max_iter
  ckpt_freq: *ckpt_freq
  print_freq: 100
  resume: *resume
  optimizer:
    type: SGD
    lr: *lr
    weight_decay: *weight_decay
    momentum: *momentum
  lr_scheduler:
    type: PolyLR
    max_iter: *max_iter
    power: 0.9
    target_lr: *target_lr
    warmup_iter: *warmup_iter
    warmup_method: linear

eval:
  type: SimpleInference
  evaluators:
    ##place_holder_evaluator##
  visualizer:
    type: MultiClassVisualizer
    visual_size: [512, 512]
    label_map: *label_map

visualize:
  type: MultiClassVisualizer

deploy:
  type: OnnxDeploy
  shapes: ##deploy_shape##
'''
