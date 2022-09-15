exp_fast_yaml = \
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
    lr: &lr {{ lr }}
    ce_weight: &ce_weight 1.0
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
        scale: [0.90, 1.11]
        aspect_ratio: [0.8, 1.25]
      - type: SMore_seg::RandomCrop
        output_size: *input_size
      - type: RandomApply
        p: 0.5
        transforms:
          - type: ColorJitter
            brightness: 0.2
            contrast: 0.2
            saturation: 0.04
            hue: 0.02
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
    workers: *workers
    dataset: *val_set
    transform:
      - *cvtColor
      - *resize
      - *normalize
      - *toTensor

  test_data:
    pin_memory: False
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
    type: {{ backbone }}
    {{ pretrained_weights }}
  head:
    type: FCNHead
    num_classes: *num_classes
    in_index: [0, 1, 2, 3]
    num_convs: 1
    kernel_size: 1
    input_transform: resize_concat
    losses:
      - type: BootstrapCE
        min_K: 4096
        loss_th: 0.001
        weight: *ce_weight
        num_classes: *num_classes
        ignore_label: *ignore_label
      - type: Dice
        weight: 1.0
        num_classes: *num_classes
        ignore_label: *ignore_label

train:
  type: SimpleTrainer
  with_amp: True
  max_iter: *iter
  ckpt_freq: *iter
  print_freq: 100
  optimizer:
    type: Adam
    lr: *lr
    weight_decay: 0.0
    eps: 0.00001
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
