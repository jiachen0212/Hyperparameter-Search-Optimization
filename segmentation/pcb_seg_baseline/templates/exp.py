exp_yaml = \
    '''
other:
  mean: &mean {{ mean }}
  std: &std {{ std }}
  scale: &scale 1.0
  workers: &workers 4
  ignore_label: &ignore_label 255
  num_classes: &num_classes {{ num_classes }}
  iter: &iter {{ num_iter }}
  batch_size: &batch_size {{ batch_size }}
  input_size: &input_size {{ input_size }}
  search_space:
    lr: &lr 0.00006
    ce_weight: &ce_weight 1.0
    weight_decay: &weight_decay 0.01
    sr_pool: &sr_pool False
    loss_th: &loss_th 0.05
  {{ baseline }}

common:
  log_level: INFO
  seed: 42
  deterministic: False
  implement_layer: [SMore_seg]
  plugin_layer: []

data:
  train_data:
    pin_memory: False
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
        scale: [0.95, 1.05]
        aspect_ratio: [0.95, 1.05]
      - type: RandomRotation
        prob: 0.3
        angle: [-5, 5]
      - type: RandomShift
        prob: 0.5
        max_shift_px: 16
      - type: SMore_seg::RandomCrop
        output_size: *input_size
      - type: RandomApply
        p: 0.3
        transforms:
          - type: PhotoMetricDistortion
            brightness_delta: 32
            contrast_range: [0.9, 1.11]
            saturation_range: [0.95, 1.05]
            hue_delta: 2
      - &normalize
        type: Normalize
        mean: *mean
        std: *std
        scale: *scale
      - &toTensor
        type: ToTensor

  eval_data:
    pin_memory: False
    batch_size: *batch_size
    workers: 2
    dataset: *val_set
    transform:
      - *cvtColor
      - *resize
      - *normalize
      - *toTensor

  test_data:
    pin_memory: False
    batch_size: *batch_size
    workers: 2
    dataset: *val_set
    transform:
      - *cvtColor
      - *resize
      - *normalize
      - *toTensor

model:
  type: EncoderDecoder
  backbone:
    type: SSA_T
    patch_size: 4
    sr_pool: *sr_pool
  head:
    type: SegFormerHead
    num_classes: *num_classes
    in_index: [0, 1, 2, 3]
    channels: 256
    input_transform: multiple_select
    losses:
      - type: BootstrapCE
        min_K: 4096
        loss_th: *loss_th
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
  max_iter: *iter
  ckpt_freq: {{ ckpt_freq }}
  print_freq: 1000
  optimizer:
    type: AdamW
    lr: *lr
    weight_decay: *weight_decay
    eps: 0.00001
    gradient_clip_cfg:
      type: norm
      max_norm: 10.0
    paramwise_cfg:
      bias_lr_mult: 2.0
      bias_decay_mult: 0.0
      custom_keys:
        backbone:
          lr_mult: 0.5
  lr_scheduler:
    type: PolyLR
    max_iter: *iter
    power: 0.9
    target_lr: 0.00001
    warmup_iter: {{ warmup_iter }}

eval:
  type: SimpleInference
  evaluators:
    - type: PixelBasedEvaluator
      num_classes: *num_classes
      label_map: *label_map
      ignore_label: *ignore_label
    - type: KillRateRelatedEvaluator
      label_map: *label_map
      num_classes: *num_classes
      OK_label: 0
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
  shapes: {{ deploy_shapes }} # [1, 3, h, w]

analyze:
  type: TransformAnalyzer
  sample: 50
  num_classes: *num_classes
  label_map: *label_map
'''
