import os
import sys
import time
import json
import math
from ruamel import yaml
import numpy as np
from .templates import cfg_template, hpo_template, sched_template, hpo_local_template
from .templates import config_template

from SMore_core.utils.config import merge_dict
from SMore_core.utils.registry import Registry, build_from_cfg  # noqa
from SMore_core.dataset.dataset_builder import DATASET
from SMore_core.utils.sm_yaml import replace_scinot_file

from SMore_hpo.baseline.baseline_base import BaselineBase
from SMore_hpo.baseline.baseline_build import BASELINE
from SMore_hpo.default_config.baseline_defaults import BaselineDefaults


def process_data_path(data_set, data_prefix):
    # Process the data_path of different dataloader form
    # if data_prefix == True: add "/data" to data_path
    # if data_prefix == False: remove "/data" from data_path

    assert data_set['type'] in ["ListDataset", "ProductMultiLabelDataset"]

    if data_set['type'] == "ListDataset":
        for data_path in data_set['data_path']:
            if data_prefix and (not str.startswith(data_path['path'], "/data")):
                data_path['path'] = "/data" + data_path['path']
            elif (not data_prefix) and str.startswith(data_path['path'], "/data"):
                data_path['path'] = data_path[5:]

    elif data_set['type'] == "ProductMultiLabelDataset":
        for path_name in ["image_path", "tag_path", "label_path"]:
            if data_prefix and (not str.startswith(data_set[path_name], "/data")):
                data_set[path_name] = "/data" + data_set[path_name]
            elif (not data_prefix) and str.startswith(data_set[path_name], "/data"):
                data_set[path_name] = data_set[path_name][5:]

    return data_set


def calculate_dataset_statistics(dataset):
    from SMore_cls.common.constants import ClassificationInputsConstants

    h_counter = 0
    w_counter = 0
    mean_counter = 0
    std_counter = 0

    for data in dataset:
        img = data[ClassificationInputsConstants.IMG]
        h_counter += img.shape[0]
        w_counter += img.shape[1]
        mean_counter += (np.mean(img / 255., axis=(0, 1))[::-1])
        std_counter += (np.std(img / 255., axis=(0, 1))[::-1])

    h_avg = h_counter / len(dataset)
    w_avg = w_counter / len(dataset)
    mean_avg = mean_counter / len(dataset)
    std_avg = std_counter / len(dataset)
    return h_avg, w_avg, mean_avg, std_avg


def calculate_recommanded_params(baseline_yaml_path, baseline_cfg, base_path='/data'):
    from SMore_cls import cifar_dataset, image_dataset, list_dataset, product_multi_label_dataset  # noqa

    recommanded_params = {}

    base_dir = os.path.dirname(os.path.abspath(baseline_yaml_path))
    customized_dir = os.path.join(base_dir, 'customized')
    recommanded_params['base_dir'] = base_dir

    if os.path.exists(customized_dir):
        sys.path.append(base_dir)
        import customized  # noqa

    recommanded_params['arch'] = baseline_cfg.get('arch')
    recommanded_params['backbone'] = baseline_cfg.get('backbone')
    recommanded_params['post_process_mode'] = baseline_cfg.get('post_process_mode')
    recommanded_params['evaluator'] = baseline_cfg.get('evaluator')
    recommanded_params['input_size'] = baseline_cfg.get('input_size')
    recommanded_params['tuner'] = baseline_cfg.get('tuner')
    recommanded_params['max_trials'] = baseline_cfg.get('max_trials')
    recommanded_params['max_iter'] = baseline_cfg.get('max_iter')
    recommanded_params['early_stop_ratio'] = baseline_cfg.get('early_stop_ratio')

    if recommanded_params['post_process_mode'] == 'sigmoid':
        recommanded_params['loss_type'] = 'BinaryCrossEntropyLoss'
    else:
        recommanded_params['loss_type'] = 'CrossEntropyLoss'

    valset_cfg = baseline_cfg.get('val_set')
    val_names = []
    for val_set in valset_cfg:
        val_names.append(val_set['data_name'])
    recommanded_params['val_names'] = val_names

    label_map = baseline_cfg.get('label_map')
    num_classes = len(label_map)
    recommanded_params['num_classes'] = num_classes

    trainset_cfg = baseline_cfg.get('train_set')

    # get dataset statistics
    trainset_cfg['transform_cfg'] = []
    dataset = build_from_cfg(module_cfg=trainset_cfg, registry=DATASET, extra_args=None)

    if recommanded_params['max_iter'] is None:
        recommanded_params['max_iter'] = \
            int(100 + 20 * math.sqrt(len(dataset)) * math.sqrt(num_classes))

    recommanded_params['ckpt_freq'] = int(recommanded_params['max_iter'] * recommanded_params['early_stop_ratio'])
    h_avg, w_avg, mean_avg, std_avg = calculate_dataset_statistics(dataset)

    recommanded_params['mean'] = [0, 0, 0]
    recommanded_params['std'] = [1, 1, 1]

    if baseline_cfg.get('calc_mean_std'):
        recommanded_params['mean'] = mean_avg.tolist()
        recommanded_params['std'] = std_avg.tolist()

    return recommanded_params


def gen_cfg_file(cfg_output_path, recommanded_params, baseline_yaml_path, cfg_template):

    cfg_file = cfg_template

    # fill place holder in exp.yaml
    cfg_file = cfg_file.replace('##place_holder_num_classes##', str(recommanded_params['num_classes']))
    cfg_file = cfg_file.replace('##place_holder_max_iter##', str(recommanded_params['max_iter']))
    cfg_file = cfg_file.replace('##place_holder_ckpt_freq##', str(recommanded_params['ckpt_freq']))
    cfg_file = cfg_file.replace('##place_holder_input_size##', str(recommanded_params['input_size']))
    cfg_file = cfg_file.replace('##place_holder_mean##', str(recommanded_params['mean']))
    cfg_file = cfg_file.replace('##place_holder_std##', str(recommanded_params['std']))
    cfg_file = cfg_file.replace('##place_holder_post_process_mode##', str(recommanded_params['post_process_mode']))
    cfg_file = cfg_file.replace('##place_holder_loss_type##', str(recommanded_params['loss_type']))

    # BACKBONE
    if recommanded_params['backbone'] == 'SEResNet18':
        backbone_info = config_template.BackBone_SEResNet18
    elif recommanded_params['backbone'] == 'ResNet18':
        backbone_info = config_template.BackBone_ResNet18
    elif recommanded_params['backbone'] == 'ResNet50':
        backbone_info = config_template.BackBone_ResNet50
    elif recommanded_params['backbone'] == 'RepVGG':
        backbone_info = config_template.BackBone_RepVGG
    cfg_file = cfg_file.replace('##place_holder_backbone##', backbone_info)

    # EVALUATOR
    if recommanded_params['evaluator'] == 'ProductMultiLabelClassificationEvaluator':
        evaluator_info = config_template.Evaluator_ProductMultiLabelClassificationEvaluator
    elif recommanded_params['evaluator'] == 'MultiLabelWithAllNegLabelsClassificationEvaluator':
        evaluator_info = config_template.Evaluator_MultiLabelWithAllNegLabelsClassificationEvaluator
    elif recommanded_params['evaluator'] == 'MultiClassClassificationEvaluator':
        evaluator_info = config_template.Evaluator_MultiClassClassificationEvaluator

    cfg_file = cfg_file.replace('##place_holder_evaluator##', evaluator_info)
    cfg_file = cfg_file.replace('##deploy_shape##', f'[1, 3, '
                                                    f'{recommanded_params["input_size"][1]}, '
                                                    f'{recommanded_params["input_size"][0]}]')

    with open(baseline_yaml_path, 'r') as baseline_file:
        baseline_file = baseline_file.read()

    # add YAML anchor
    baseline_file = baseline_file.replace('label_map:', 'label_map: &label_map')
    baseline_file = baseline_file.replace('train_set:', 'train_set: &train_set')
    baseline_file = baseline_file.replace('val_set:', 'val_set: &val_set')
    baseline_file = baseline_file.replace('baseline:', '# baseline')  # to adapt core2.0

    cfg_file = cfg_file.replace('##place_holder_baseline##', baseline_file)

    with open(cfg_output_path, 'w') as f:
        f.write(cfg_file)


def gen_hpo_file(hpo_output_path, base_exp_path, recommanded_params, rerun_best_trial, hpo_template):

    hpo = yaml.load(hpo_template, Loader=yaml.Loader)

    hpo['common']['seed'] = int(time.time()) % 1000000
    hpo['manager']['rerun_best_trial'] = rerun_best_trial
    hpo['manager']['base_exp_path'] = base_exp_path
    hpo['manager']['max_trials'] = recommanded_params['max_trials']

    # evaluation metric
    metrics = []
    for val_name in recommanded_params['val_names']:
        if recommanded_params['evaluator'] == 'ProductMultiLabelClassificationEvaluator':
            metrics.append(
                {'data_name': val_name,
                 'metric_type': "ProductMultiLabelClassificationEvaluator/overall_index/global_recall",
                 'metric_weight': 0.5})
            metrics.append(
                {'data_name': val_name,
                 'metric_type': "ProductMultiLabelClassificationEvaluator/overall_index/global_precision",
                 'metric_weight': 0.5})
        elif recommanded_params['evaluator'] == 'MultiLabelWithAllNegLabelsClassificationEvaluator':
            metrics.append(
                {'data_name': val_name,
                 'metric_type': "MultiLabelWithAllNegLabelsClassificationEvaluator/F1/mean",
                 'metric_weight': 1.0})
        elif recommanded_params['evaluator'] == 'MultiClassClassificationEvaluator':
            metrics.append(
                {'data_name': val_name,
                 'metric_type': "MultiClassClassificationEvaluator/Top1 Acc",
                 'metric_weight': 1.0})

    hpo['trial']['metrics'] = metrics

    # tuner
    if recommanded_params['tuner'] == 'tpe':
        hpo['tuner']['type'] = 'HyperoptTuner'
        hpo['tuner']['algorithm_name'] = 'tpe'
    elif recommanded_params['tuner'] == 'hebo':
        hpo['tuner']['type'] = 'HEBOTuner'
        hpo['tuner']['optimize_batch'] = 4
    elif recommanded_params['tuner'] == 'openbox':
        hpo['tuner']['type'] = 'OpenBoxTuner'
        hpo['tuner']['max_concurrency_trials'] = 4

    # assessor
    if recommanded_params['early_stop_ratio'] < 1.0:
        hpo['assessor']['type'] = 'MedianStopAssessor'
    else:
        hpo['assessor']['type'] = 'DummyAssessor'

    with open(hpo_output_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(hpo, yaml_file, Dumper=yaml.RoundTripDumper)
    replace_scinot_file(hpo_output_path)


def gen_scheduler(sched_output_path, sched_template, recommanded_params):

    load_scheduler = json.loads(sched_template)
    arch = recommanded_params['arch']

    if arch:
        if arch == 'ampere':
            load_scheduler['image'] = load_scheduler['image'].replace('.cu10', '.cu11')
        load_scheduler['arch'] = arch
    with open(sched_output_path, 'w') as scheduler_f:
        json.dump(load_scheduler, scheduler_f, indent=4)


@BASELINE.register_module()
class ClsBaseline(BaselineBase):
    DEFAULT_CONFIG = BaselineDefaults.ClsBaseline_cfg

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = merge_dict(BaselineDefaults.ClsBaseline_cfg, self.kwargs)

    def gen_cfgs(self, baseline_yaml_path, hpo_path, rerun_best_trial=False):

        recommanded_params = calculate_recommanded_params(baseline_yaml_path, self.kwargs)

        base_dir = recommanded_params['base_dir']
        cfg_output_path = os.path.join(base_dir, 'exp.yaml')
        hpo_output_path = os.path.join(hpo_path, 'hpo.yaml')
        sched_output_path = os.path.join(base_dir, 'scheduler.json')

        gen_cfg_file(cfg_output_path=cfg_output_path,
                     recommanded_params=recommanded_params,
                     baseline_yaml_path=baseline_yaml_path,
                     cfg_template=cfg_template)

        gen_hpo_file(hpo_output_path=hpo_output_path,
                     base_exp_path=cfg_output_path,
                     recommanded_params=recommanded_params,
                     rerun_best_trial=False,
                     hpo_template=hpo_template)

        gen_scheduler(sched_output_path, sched_template,
                      recommanded_params=recommanded_params)


@BASELINE.register_module()
class LocalClsBaseline(BaselineBase):
    DEFAULT_CONFIG = BaselineDefaults.ClsBaseline_cfg

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = merge_dict(BaselineDefaults.ClsBaseline_cfg, self.kwargs)

    def gen_cfgs(self, baseline_yaml_path, hpo_path, rerun_best_trial=False):

        recommanded_params = calculate_recommanded_params(baseline_yaml_path, self.kwargs, base_path='')

        base_dir = recommanded_params['base_dir']
        cfg_output_path = os.path.join(base_dir, 'exp.yaml')
        hpo_output_path = os.path.join(hpo_path, 'hpo.yaml')

        gen_cfg_file(cfg_output_path=cfg_output_path,
                     recommanded_params=recommanded_params,
                     baseline_yaml_path=baseline_yaml_path,
                     cfg_template=cfg_template)

        gen_hpo_file(hpo_output_path=hpo_output_path,
                     base_exp_path=cfg_output_path,
                     recommanded_params=recommanded_params,
                     rerun_best_trial=False,
                     hpo_template=hpo_local_template)
