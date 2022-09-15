import os
import sys
import time
import math
import numpy as np
import json
import random
from tqdm import tqdm
from ruamel import yaml

from .templates import cfg_template, sched_template
from .templates import cluster_hpo_template

from SMore_core.utils.config import merge_dict
from SMore_core.utils.registry import build_from_cfg
from SMore_core.dataset.dataset_builder import DATASET

from SMore_hpo.baseline.baseline_base import BaselineBase
from SMore_hpo.baseline.baseline_build import BASELINE
from SMore_core.utils.sm_yaml import replace_scinot_file

from SMore_hpo.default_config.baseline_defaults import BaselineDefaults

RESIZE = [768, 1024, 2048]
BATCH_SIZE = [32, 24, 8]


def get_dataset(base_dir, trainset_cfg):
    from SMore_seg.dataset import LabelMeDataset, MaskDataset, ZipMaskDataset  # noqa

    customized_dir = os.path.join(base_dir, 'customized')

    if os.path.exists(customized_dir):
        sys.path.append(base_dir)
        import customized  # noqa

    trainset_cfg['transform_cfg'] = []

    dataset = build_from_cfg(module_cfg=trainset_cfg,
                             registry=DATASET,
                             extra_args=None)

    return dataset


def calculate_dataset_statistics(dataset, num_classes):
    from SMore_seg.common.constants import SegmentationInputsConstants

    pixel_counter = []
    shape_counter = []
    mean_list = []
    std_list = []

    if len(dataset) > 300:
        samples = random.sample(list(range(len(dataset))), 300)
    else:
        samples = list(range(len(dataset)))

    for idx in tqdm(samples):
        data = dataset[idx]
        img = data[SegmentationInputsConstants.IMG]
        target = data[SegmentationInputsConstants.TARGET]
        shape_counter.append(target.shape)  # h, w
        bg = np.sum(target == 0)
        ng = np.sum(target != 0)
        ratio = ng / bg
        pixel_counter.append(ratio)
        mean_list.append(np.mean(img / 255., axis=(0, 1))[::-1])
        std_list.append(np.std(img / 255., axis=(0, 1))[::-1])

    mean = sum(mean_list) / len(mean_list)
    std = sum(std_list) / len(std_list)
    return pixel_counter, shape_counter, mean, std


def calculate_recommanded_params(dataset, num_classes):
    """
    Return:
        {'input_size':[], 'batch_size':int, 'num_iter':int, 'lr':[], 'mean':float, 'std':float}
    """
    recommanded_params = {}

    pixel_counter, shape_counter, mean, std = calculate_dataset_statistics(
        dataset, num_classes)
    num_data = len(dataset)

    # input size & batch size
    input_size_list = []
    avg_w = 0
    avg_h = 0
    for shape in shape_counter:
        avg_h += shape[0]
        avg_w += shape[1]
    avg_h /= len(shape_counter)
    avg_w /= len(shape_counter)

    if max(avg_h, avg_w) / min(avg_h, avg_w) > 3:
        scale = 2
    elif max(avg_h, avg_w) / min(avg_h, avg_w) > 2:
        scale = 1.5
    else:
        scale = 1
    for i in range(len(RESIZE)):
        RESIZE[i] = int(RESIZE[i] * scale)

    for max_lenth in RESIZE:
        ratio = max_lenth / max(avg_h, avg_w)
        if avg_h > avg_w:
            h = max_lenth
            w = round(ratio * avg_w / 32) * 32
        else:
            w = max_lenth
            h = round(ratio * avg_h / 32) * 32
        input_size_list.append([w, h])

    pixel_counter = np.array(pixel_counter)
    pixel_counter = np.sort(pixel_counter)
    max_20p = min(int(len(pixel_counter) * 0.8), len(pixel_counter) - 1)
    max_ratio = np.mean(pixel_counter[max_20p:])
    min_5p = max(int(len(pixel_counter[pixel_counter > 0]) * 0.05), 1)
    min_ratio = np.mean(pixel_counter[pixel_counter > 0][:min_5p])

    if max_ratio > 0.1 and min_ratio > 0.02 and max(avg_h, avg_w) < 1600:
        input_size = input_size_list[0]
        bs = BATCH_SIZE[0]
    elif max_ratio < 0.05 and min_ratio < 0.0001 and max(avg_h, avg_w) > 2000:
        input_size = input_size_list[2]
        bs = BATCH_SIZE[2]
    else:
        input_size = input_size_list[1]
        bs = BATCH_SIZE[1]

    recommanded_params['input_size'] = input_size
    recommanded_params['batch_size'] = bs

    # num iter
    num_iter = math.ceil(num_data / 50) * 1000
    if input_size[0] * input_size[1] > 1000000:
        num_iter *= 2
    num_iter = min(num_iter, 8000)
    num_iter += 1000 * (num_data // 1000)

    recommanded_params['num_iter'] = num_iter

    # lr
    lr_small = input_size[0] * input_size[1] / 800000 * 0.000125
    lr_max = 16 * lr_small
    lr = [lr_small, lr_max]
    recommanded_params['lr'] = lr

    # mean std
    recommanded_params['mean'] = mean
    recommanded_params['std'] = std

    return recommanded_params


def gen_cfg_file(cfg_output_path, mean, std, num_classes, num_iter, batch_size,
                 input_size, backbone, pretrained_weight, deploy_shapes,
                 baseline_yaml_path, cfg_template):

    cfg_file = cfg_template

    str_mean = '[%.4f, %.4f, %.4f]' % tuple(mean)
    str_std = '[%.4f, %.4f, %.4f]' % tuple(std)

    # fill place holder in exp.yaml
    cfg_file = cfg_file.replace('##place_holder_mean##', str_mean)
    cfg_file = cfg_file.replace('##place_holder_std##', str_std)
    cfg_file = cfg_file.replace('##place_holder_num_classes##',
                                str(num_classes))
    cfg_file = cfg_file.replace('##place_holder_num_iter##', str(num_iter))
    cfg_file = cfg_file.replace('##place_holder_batch_size##', str(batch_size))
    cfg_file = cfg_file.replace('##place_holder_input_size##', str(input_size))
    cfg_file = cfg_file.replace('##place_holder_backbone##', backbone)
    if pretrained_weight:
        cfg_file = cfg_file.replace('##place_holder_pretrained_weights##',
                                    'pretrained_weights: ' + pretrained_weight)
    cfg_file = cfg_file.replace('##place_holder_deploy_shapes##',
                                str(deploy_shapes))

    with open(baseline_yaml_path, 'r') as baseline_file:
        baseline_file = baseline_file.read()

    # add YAML anchor
    baseline_file = baseline_file.replace('label_map:',
                                          'label_map: &label_map')
    baseline_file = baseline_file.replace('train_set:',
                                          'train_set: &train_set')
    baseline_file = baseline_file.replace('val_set:', 'val_set: &val_set')

    # concat exp.yaml
    baseline_cfg = baseline_file.replace('baseline:', '# baseline')
    cfg_file = cfg_file.replace('##place_holder_baseline##', baseline_cfg)

    with open(cfg_output_path, 'w') as f:
        f.write(cfg_file)


def gen_hpo_file(
    hpo_output_path,
    hpo_template,
    base_exp_path,
    val_names,
    lr=[0.0002, 0.004],
    max_trials=20,
    num_gpus=2,
    total_gpus=8,
    metric_type='iou',
    rerun_best_trial=False,
    max_iter=0,
):

    hpo = yaml.load(hpo_template, Loader=yaml.Loader)

    hpo['common']['seed'] = int(time.time()) % 10000
    hpo['tuner']['search_space']['lr']['value'] = lr
    if 'optimize_batch' in hpo['tuner'].keys():
        hpo['tuner']['optimize_batch'] = total_gpus // num_gpus
    hpo['manager']['rerun_best_trial'] = rerun_best_trial
    hpo['manager']['max_trials'] = max_trials
    hpo['manager']['max_concurrency_trials'] = total_gpus // num_gpus
    hpo['manager']['base_exp_path'] = base_exp_path

    hpo['trial']['iter_per_resource'] = max_iter // 4

    if 'resource' in hpo['trial'].keys():
        hpo['trial']['resource']['total_gpus'] = total_gpus
        hpo['trial']['resource']['gpus_per_trial'] = num_gpus

    metrics = []
    if metric_type == 'product':
        for val_name in val_names:
            metrics.append({
                'data_name': val_name,
                'metric_type': 'ProductEvaluator/overall_index/precision',
                'metric_weight': 0.5 / len(val_names)
            })
            metrics.append({
                'data_name': val_name,
                'metric_type': 'ProductEvaluator/overall_index/recall',
                'metric_weight': 0.5 / len(val_names)
            })
    elif metric_type == 'iou':
        for val_name in val_names:
            metrics.append({
                'data_name': val_name,
                'metric_type': 'PixelBasedEvaluator/iou/mean',
                'metric_weight': 1.0 / len(val_names)
            })
    else:
        for val_name in val_names:
            metrics.append({
                'data_name': val_name,
                'metric_type': metric_type,
                'metric_weight': 1.0 / len(val_names)
            })

    hpo['trial']['metrics'] = metrics

    with open(hpo_output_path, 'w', encoding='utf-8') as yaml_file:
        yaml.dump(hpo, yaml_file, Dumper=yaml.RoundTripDumper)

    replace_scinot_file(hpo_output_path)


def gen_scheduler(sched_output_path,
                  sched_template,
                  num_gpus=1,
                  arch='ampere'):

    load_scheduler = json.loads(sched_template)

    if arch:
        if arch != 'ampere':
            load_scheduler['image'] = load_scheduler['image'].replace(
                '.cu11', '.cu10')
        load_scheduler['arch'] = arch

    load_scheduler['train']['GPU'] = num_gpus
    load_scheduler['train']['num_gpus_per_machine'] = num_gpus

    with open(sched_output_path, 'w') as scheduler_f:
        json.dump(load_scheduler, scheduler_f, indent=4)


@BASELINE.register_module()
class FLAMLSegBaseline(BaselineBase):
    DEFAULT_CONFIG = BaselineDefaults.FLAMLSegBaseline_cfg

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = merge_dict(BaselineDefaults.FLAMLSegBaseline_cfg,
                                 self.kwargs)

    def gen_cfgs(self, baseline_yaml_path, hpo_path, rerun_best_trial=False):

        label_map = self.kwargs.get('label_map')
        data_cfg = self.kwargs.get('train_set')
        valset_cfg = self.kwargs.get('val_set')
        val_names = []
        for valset in valset_cfg:
            val_names.append(valset['data_name'])

        base_dir = os.path.dirname(os.path.abspath(baseline_yaml_path))
        dataset = get_dataset(base_dir, data_cfg)
        recommanded_params = calculate_recommanded_params(
            dataset, len(label_map))

        cfg_output_path = os.path.join(base_dir, 'exp.yaml')
        hpo_output_path = os.path.join(hpo_path, 'hpo.yaml')
        sched_output_path = os.path.join(base_dir, 'scheduler.json')

        batch_size = self.kwargs.get('batch_size')
        if not batch_size:
            batch_size = recommanded_params['batch_size']

        num_iter = self.kwargs.get('max_iter')
        if not num_iter:
            num_iter = recommanded_params['num_iter']

        input_size = self.kwargs.get('input_size')
        if not input_size:
            input_size = recommanded_params['input_size']

        total_gpus = self.kwargs.get('gpu_per_trial') * \
            self.kwargs.get('max_concurrency_trials')

        # scale lr
        lr_small = recommanded_params['lr'][0]
        lr_small *= (input_size[0] * input_size[1])
        lr_small /= (recommanded_params['input_size'][0] *
                     recommanded_params['input_size'][1])
        lr_small *= batch_size
        lr_small /= 8.0
        lr_small *= self.kwargs.get('gpu_per_trial')
        lr_small /= 2.0
        recommanded_params['lr'][0] = lr_small
        recommanded_params['lr'][1] = lr_small * 16

        gen_cfg_file(cfg_output_path=cfg_output_path,
                     mean=recommanded_params['mean'],
                     std=recommanded_params['std'],
                     num_classes=len(label_map),
                     num_iter=num_iter,
                     batch_size=batch_size,
                     input_size=input_size,
                     backbone=self.kwargs.get('backbone'),
                     pretrained_weight=self.kwargs.get('pretrained_weight'),
                     deploy_shapes=[1, 3, input_size[1], input_size[0]],
                     baseline_yaml_path=baseline_yaml_path,
                     cfg_template=cfg_template)

        gen_hpo_file(hpo_output_path=hpo_output_path,
                     hpo_template=cluster_hpo_template,
                     base_exp_path=cfg_output_path,
                     val_names=val_names,
                     lr=recommanded_params['lr'],
                     max_trials=self.kwargs.get('max_trials'),
                     num_gpus=self.kwargs.get('gpu_per_trial'),
                     total_gpus=total_gpus,
                     metric_type=self.kwargs.get('metric'),
                     rerun_best_trial=rerun_best_trial,
                     max_iter=recommanded_params['num_iter'])

        gen_scheduler(sched_output_path,
                      sched_template,
                      num_gpus=self.kwargs.get('gpu_per_trial'),
                      arch=self.kwargs.get('arch'))
