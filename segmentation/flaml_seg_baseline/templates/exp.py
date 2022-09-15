exp_yaml = \
    '''
other:
  mean: &mean ##place_holder_mean##
  std: &std ##place_holder_std##
  scale: &scale 1.0
  workers: &workers 4
  ignore_label: &ignore_label 255
  num_classes: &num_classes ##place_holder_num_classes##
  batch_size: &batch_size ##place_holder_batch_size##
  input_size: &input_size ##place_holder_input_size##
  search_space:
    lr: &lr 0.0003
    ce_weight: &ce_weight 0.5
    rotate_angle: &rotate_angle [0,0]
    max_iter: &max_iter ##place_holder_num_iter##
  ##place_holder_baseline##

common:
  log_level: INFO
  seed: 42
  deterministic: False
  implement_layer: [SMore_seg]
  plugin_layer: []

data:
  train_data:
    batch_size: *batch_size
    workers: *workers
    dataset: *train_set
    transform:
      - &cvtColor
        type: CvtColor
        mode: BGR2RGB
      - &resize
        type: Resize
        output_size: *input_size
      - type: RandomHorizontalFlip
        prob: 0.5
      - type: RandomVerticalFlip
        prob: 0.5
      - type: RandomScale
        scale: [0.8, 1.25]
        aspect_ratio: [0.8, 1.25]
      - type: RandomRotation
        prob: 0.5
        angle: *rotate_angle
      - type: SMore_seg::RandomCrop
        output_size: *input_size
      - type: RandomApply
        p: 0.3
        transforms:
          - type: PhotoMetricDistortion
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
      - *cvtColor
      - *resize
      - *normalize
      - *toTensor

  test_data:
    batch_size: *batch_size
    workers: *workers
    dataset: *val_set
    transform:
      - *cvtColor
      - *resize
      - *normalize
      - *toTensor

model:
  type: EncoderDecoder
  backbone:
    type: ##place_holder_backbone##
    ##place_holder_pretrained_weights##
  head:
    type: FCNHead
    num_classes: *num_classes
    in_index: [0, 1, 2, 3]
    num_convs: 1
    kernel_size: 1
    input_transform: resize_concat
    se: False
    losses:
      - type: BootstrapCE
        weight: *ce_weight
        num_classes: *num_classes
        ignore_label: *ignore_label
      - type: Dice
        weight: 0.5
        num_classes: *num_classes
        ignore_label: *ignore_label

train:
  type: SimpleTrainer
  with_amp: True
  max_iter: *max_iter
  ckpt_freq: 1000
  print_freq: 100
  optimizer:
    type: Adam
    lr: *lr
    weight_decay: 0.0
  lr_scheduler:
    type: PolyLR
    max_iter: *max_iter
    power: 0.9
    target_lr: 0.00001
    warmup_iter: 100

eval:
  type: SimpleInference
  evaluators:
    - type: PixelBasedEvaluator
      num_classes: *num_classes
      label_map: *label_map
      ignore_label: *ignore_label
  visualizer:
    type: KillRateVisualizer
    num_classes: *num_classes
    label_map: *label_map

visualize:
  type: KillRateVisualizer
  num_classes: *num_classes
  label_map: *label_map

deploy:
  type: OnnxDeploy
  shapes: ##place_holder_deploy_shapes## # [1, 3, h, w]

analyze:
  type: TransformAnalyzer
  sample: 50
  num_classes: *num_classes
  label_map: *label_map
'''
