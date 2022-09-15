exp_yaml = \
    '''
common:
  log_level: INFO
  seed: 0
  deterministic: True
  implement_layer: [SMore_det]
  plugin_layer: []

other:
  subdivisions: &subdivisions 1
  mean: &mean [123.675, 116.280, 103.530]
  std: &std [58.395, 57.120, 57.375]
  scale: &scale 255.0
  workers: &workers 4
  anchor:
    num_classes: &num_classes {{ num_classes }}
    label_map: &label_map {{ label_map }}
  focal_gamma: &focal_gamma 2.0
  focal_alpha: &focal_alpha 0.25
  focal_weight: &focal_weight 1.0
  iou_weight: &iou_weight 1.0
  center_weight: &center_weight 1.0
  brightness_delta: &brightness_delta 50
  contrast_range: &contrast_range [0.3, 1.6]
  hue_delta: &hue_delta 9
  search_space:
    lr: &lr 0.001
    weight_decay: &weight_decay 0.0001
    regress_ranges: &regress_ranges [[-1, 32], [32, 64], [64, 128], [128, 256], [256, 9999]]
    size_bs_iter: &size_bs_iter {{ size_bs_iter }}
    eval_size: &eval_size {{ eval_size }} ##place_holder_eval_size##
    max-iter: &max-iter {{ max_iter }} ##place_holder_max-iter##
    input_size: &input_size {{ input_size }} ##place_holder_input_size##
    batch_size: &batch_size {{ batch_size }} ##place_holder_batch_size##
  ##place_holder_baseline##

data:
  train_data:
    batch_size: *batch_size
    workers: *workers
    group_dataset:
      type: AspectRatioGroupedDataset
      drop_last: False
    dataset: *train_set
    transform:
      - type: MultiRescale
        img_scale: *input_size
        prob: [0.3, 0.4, 0.3]
      - type: RandomHorizontalFlip
        prob: 0.5
#      - type: PhotoMetricDistortion
#        brightness_delta: *brightness_delta
#        contrast_range: *contrast_range
#        hue_delta: *hue_delta
      - type: Normalize
        mean: *mean
        std: *std
        scale: *scale
      - &toTensor
        type: ToTensor

  eval_data:
    batch_size: 1
    workers: *workers
    group_dataset:
      type: AspectRatioGroupedDataset
      drop_last: False
    dataset: *val_set
    transform:
      - type: Rescale
        img_scale: *eval_size
      - type: Normalize
        mean: *mean
        std: *std
        scale: *scale
      - *toTensor

  test_data:
    batch_size: 1
    workers: *workers
    group_dataset:
      type: AspectRatioGroupedDataset
      drop_last: False
    dataset: *val_set
    transform:
      - type: Rescale
        img_scale: *eval_size
      - type: Normalize
        mean: *mean
        std: *std
        scale: *scale
      - *toTensor

model:
  type: SingleStageDetector
  backbone:
    type: {{ backbone }}
    out_indices: [0,1,2,3]
    pretrained_weights: /mnt/yfs/sharedir/industrial/PUBLIC/classification/imagenet_pretrained/resnet18-5c106cde.pth
    frozen_stages: 1
    norm_cfg:
      type: FrozenBN
      requires_grad: True
    style: pytorch
  neck:
    type: FPN
    out_channels: 256
    start_level: 1
    add_extra_convs: on_output
    num_outs: 5
    relu_before_extra_convs: True
  head:
    type: FCOSHead
    num_classes: *num_classes
    in_channels: 256
    strides: [8, 16, 32, 64, 128]
    regress_ranges: *regress_ranges
    stacked_convs: 4
    conv_bias: True
    loss_cls:
      - type: FocalLoss
        gamma: *focal_gamma
        alpha: *focal_alpha
        reduction: sum
        loss_weight: *focal_weight
    loss_bbox:
      - type: IoULoss
        loss_weight: *iou_weight
    loss_centerness:
      - type: CrossEntropyLoss
        use_sigmoid: True
    norm_cfg:
      type: 'GN'
      num_groups: 32
      requires_grad: True
    post_process:
      type: MultiClassNMS
      max_num: {{ num_bbox }}
      score_threshold: 0.05
      iou_threshold: 0.5

train:
  type: SimpleTrainer
  with_amp: False
  max_iter: *max-iter
  ckpt_freq: *max-iter # the same as eval freq.
  print_freq: 10
  optimizer:
    type: SGD
    lr: *lr
    momentum: 0.9
    weight_decay: *weight_decay
    nesterov: False
    paramwise_cfg:
      bias_lr_mult: 1.0
      bias_decay_mult: 0.0
      norm_decay_mult: 0.0

  lr_scheduler:
    type: WarmupCosineLR
    max_iter: *max-iter
    warmup_factor: 0.001
    warmup_iter: 300
    warmup_method: "linear"

eval:
  type: SimpleInference
  evaluators:
    - type: mAPEvaluator
      label_map: *label_map
    - type: DRPEvaluator
      label_map: *label_map
      scores_threshold: 0.5
      iou_threshold: 0.85
      distance_range: {"all":  [0, 100086],}
  visualizer:
    type: ConcatVisualizer
    painters:
      - type: BoxPainter
        num_classes: *num_classes
        label_map: *label_map
        vis_threshold: False
        compare: True
        score_threshold: 0.5

analyze:
  type: TransformAnalyzer
  sample: 100
  label_map: *label_map
  num_classes: *num_classes

deploy:
  type: OnnxDeploy
  shapes: [1, 3, 768, 768]
    '''
