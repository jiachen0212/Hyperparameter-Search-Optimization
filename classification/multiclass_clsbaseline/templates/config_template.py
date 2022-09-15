# flake8: noqa
BackBone_SEResNet18 = "type: ResNet18\n    block_cfg:\n      type: SEBasicBlock\n    pretrained_weights: /mnt/yfs/industrial/PUBLIC/classification/imagenet_pretrained/seresnet18.pth"
BackBone_ResNet18 = "type: ResNet18\n    pretrained_weights: /mnt/yfs/industrial/PUBLIC/classification/imagenet_pretrained/resnet18.pth"
BackBone_ResNet50 = "type: ResNet50\n    pretrained_weights: /mnt/yfs/industrial/PUBLIC/classification/imagenet_pretrained/resnet50.pth"
BackBone_RepVGG = "type: RepVGG_A0\n    pretrained_weights: /mnt/yfs/industrial/PUBLIC/classification/imagenet_pretrained/RepVGG_A0.pth"

Evaluator_MultiClassClassificationEvaluator = "- type: MultiClassClassificationEvaluator\n"
Evaluator_ProductMultiLabelClassificationEvaluator = "- type: ProductMultiLabelClassificationEvaluator\n      num_classes: *num_classes\n      default_thresholds: 0.5\n      label_map: *label_map"
Evaluator_MultiLabelWithAllNegLabelsClassificationEvaluator = "- type: MultiLabelWithAllNegLabelsClassificationEvaluator\n      num_classes: *num_classes\n      thresholds: 0.5\n      label_map: *label_map"
