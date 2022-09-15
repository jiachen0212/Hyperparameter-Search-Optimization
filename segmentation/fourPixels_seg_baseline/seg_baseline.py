import os
import sys
import time
import math
import jinja2
import numpy as np
import json
import random
from tqdm import tqdm
from ruamel import yaml

from .templates import cfg_template
from .templates import cluster_hpo_template

from SMore_core.utils.config import merge_dict
from SMore_core.utils.registry import build_from_cfg
from SMore_core.dataset.dataset_builder import DATASET
from SMore_core.utils.sm_yaml import replace_scinot_file

from SMore_hpo.baseline.cfg_generator import HPOSchedulerGenerator, APPSchedulerGenerator

from SMore_hpo.baseline.baseline_base import BaselineBase
from SMore_hpo.baseline.baseline_build import BASELINE
from SMore_hpo.default_config.baseline_defaults import BaselineDefaults

NUM_SAMPLES = 100


class PresetLevel_HRNetW18SmallV1:
    BACKBONE = 'HRNetW18SmallV1'
    RESIZE = [768, 1024, 2048]
    BATCH_SIZE = [32, 18, 6]
    BASE_ITER = [1000, 2000, 4000]


class PresetLevel_HRNetW18SmallV2:
    BACKBONE = 'HRNetW18SmallV2'
    RESIZE = [768, 1024, 2048]
    BATCH_SIZE = [16, 9, 3]
    BASE_ITER = [1000, 2000, 4000]


class PresetLevel_HRNetW18:
    BACKBONE = 'HRNetW18'
    RESIZE = [768, 1024, 1280]
    BATCH_SIZE = [6, 4, 2]
    BASE_ITER = [1000, 2000, 3000]


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

    if len(dataset) > NUM_SAMPLES:
        samples = random.sample(list(range(len(dataset))), NUM_SAMPLES)
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


def calculate_recommanded_params(dataset, num_classes, preset):
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
    resized = []
    for i in range(len(preset.RESIZE)):
        resized.append(int(preset.RESIZE[i] * scale))

    for max_lenth in resized:
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
        level = 0
    elif max_ratio < 0.05 and min_ratio < 0.0001 and max(avg_h, avg_w) > 2000:
        level = 2
    else:
        level = 1
    input_size = input_size_list[level]
    bs = preset.BATCH_SIZE[level]
    bs = max(round(bs * 0.9 ** ((num_classes - 2) // 10)), 2)

    recommanded_params['input_size'] = input_size
    recommanded_params['batch_size'] = bs

    # num iter
    num_iter = preset.BASE_ITER[level]
    num_iter += math.ceil(num_data / 100) * 1000
    num_iter += (num_classes - 2) * 1000

    recommanded_params['num_iter'] = num_iter
    recommanded_params['warmup_iter'] = int(num_iter * 0.05 // 50 * 50)

    # lr
    lr_small = input_size[0] * input_size[1] / 800000 * 0.000125
    lr_max = 16 * lr_small
    lr = [lr_small, lr_max]
    recommanded_params['lr'] = lr

    # mean std
    recommanded_params['mean'] = mean
    recommanded_params['std'] = std

    # backbone
    recommanded_params['backbone'] = preset.BACKBONE

    return recommanded_params


class EXPGenerator():
    "generate exp.yaml"

    REQUIRED = ['mean', 'std', 'num_classes', 'num_iter', 'batch_size', 'input_size', 'lr',
                'backbone', 'ckpt_freq', 'warmup_iter', 'deploy_shapes']

    def __init__(self, template, baseline_yaml_path):
        self.template = template
        self.baseline = self.preprocess_baseline(baseline_yaml_path)

    def preprocess_baseline(self, baseline_yaml_path):
        with open(baseline_yaml_path, 'r') as baseline_file:
            baseline_file = baseline_file.read()

        # add YAML anchor
        baseline_file = baseline_file.replace('label_map:', 'label_map: &label_map')
        baseline_file = baseline_file.replace('train_set:', 'train_set: &train_set')
        baseline_file = baseline_file.replace('val_set:', 'val_set: &val_set')
        # prepare to concat exp.yaml
        baseline_cfg = baseline_file.replace('baseline:', '# baseline')
        return baseline_cfg

    def get_params(self, param_dict):
        self.param_dict = param_dict
        for key in self.REQUIRED:
            assert key in self.param_dict.keys()

    def dump(self, dump_path):
        mean, std = self.param_dict['mean'], self.param_dict['std']
        self.param_dict['mean'] = '[%.4f, %.4f, %.4f]' % tuple(mean)
        self.param_dict['std'] = '[%.4f, %.4f, %.4f]' % tuple(std)
        for key in self.param_dict.keys():
            self.param_dict[key] = str(self.param_dict[key])
        if 'pretrained_weights' not in self.param_dict.keys():
            self.param_dict['pretrained_weights'] = ''
        else:
            self.param_dict['pretrained_weights'] = \
                'pretrained_weights: ' + self.param_dict['pretrained_weights']
        self.param_dict['baseline'] = self.baseline
        tm = jinja2.Template(self.template)
        cfg = tm.render(**self.param_dict)
        with open(dump_path, 'w') as f:
            f.write(cfg)


def gen_hpo_file(hpo_output_path,
                 hpo_template,
                 base_exp_path,
                 val_names,
                 lr=[0.0002, 0.004],
                 max_trials=20,
                 num_gpus=2,
                 total_gpus=8,
                 metric_type='iou',
                 rerun_best_trial=False):

    hpo = yaml.load(hpo_template, Loader=yaml.Loader)

    hpo['common']['seed'] = int(time.time()) % 10000
    hpo['tuner']['search_space']['lr']['value'] = lr
    if 'optimize_batch' in hpo['tuner'].keys():
        hpo['tuner']['optimize_batch'] = total_gpus // num_gpus
    hpo['manager']['rerun_best_trial'] = rerun_best_trial
    hpo['manager']['max_trials'] = max_trials
    hpo['manager']['max_concurrency_trials'] = total_gpus // num_gpus
    hpo['manager']['base_exp_path'] = base_exp_path

    if 'resource' in hpo['trial'].keys():
        hpo['trial']['resource']['total_gpus'] = total_gpus
        hpo['trial']['resource']['gpus_per_trial'] = num_gpus

    metrics = []
    if metric_type == 'product':
        for val_name in val_names:
            metrics.append({'data_name': val_name,
                            'metric_type': "ProductEvaluator/overall_index/precision",
                            'metric_weight': 0.5 / len(val_names)})
            metrics.append({'data_name': val_name,
                            'metric_type': "ProductEvaluator/overall_index/recall",
                            'metric_weight': 0.5 / len(val_names)})
    elif metric_type == 'iou':
        for val_name in val_names:
            metrics.append({'data_name': val_name,
                            'metric_type': "PixelBasedEvaluator/iou/mean",
                            'metric_weight': 1.0 / len(val_names)})
    else:
        for val_name in val_names:
            metrics.append({'data_name': val_name,
                            'metric_type': metric_type,
                            'metric_weight': 1.0 / len(val_names)})

    hpo['trial']['metrics'] = metrics

    with open(hpo_output_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(hpo, yaml_file, Dumper=yaml.RoundTripDumper)

    replace_scinot_file(hpo_output_path)


def gen_scheduler(sched_output_path,
                  sched_template,
                  num_gpus=1,
                  arch="ampere"):

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
class FourPixelsSegBaseline(BaselineBase):
    DEFAULT_CONFIG = BaselineDefaults.FourPixelsSegBaseline_cfg

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = merge_dict(BaselineDefaults.FourPixelsSegBaseline_cfg, self.kwargs)
        self.cfg_template = cfg_template
        self.cluster_hpo_template = cluster_hpo_template

    def gen_cfgs(self, baseline_yaml_path, hpo_path, rerun_best_trial=False):
        label_map = self.kwargs.get('label_map')
        data_cfg = self.kwargs.get('train_set')
        valset_cfg = self.kwargs.get('val_set')
        val_names = []
        for valset in valset_cfg:
            val_names.append(valset['data_name'])

        base_dir = os.path.dirname(os.path.abspath(baseline_yaml_path))
        dataset = get_dataset(base_dir, data_cfg)
        os.makedirs(hpo_path, exist_ok=True)

        cfg_output_path = os.path.join(base_dir, 'exp.yaml')
        hpo_output_path = os.path.join(hpo_path, 'hpo.yaml')
        hpo_scheduler_path = os.path.join(hpo_path, 'scheduler.json')
        sched_output_path = os.path.join(base_dir, 'scheduler.json')

        preset = self.kwargs.get('preset')
        preset = [PresetLevel_HRNetW18SmallV1(), PresetLevel_HRNetW18SmallV2(), PresetLevel_HRNetW18()][preset - 1]

        recommended_params = calculate_recommanded_params(dataset, len(label_map), preset)

        # scale bs
        arch = self.kwargs.get('arch')
        if arch != 'ampere':
            recommended_params['batch_size'] = recommended_params['batch_size'] // 2

        # user setting
        batch_size = self.kwargs.get('batch_size')
        if batch_size:
            recommended_params['batch_size'] = batch_size

        num_iter = self.kwargs.get('max_iter')
        if num_iter:
            recommended_params['num_iter'] = num_iter
        recommended_params['ckpt_freq'] = int(recommended_params['num_iter']) // 4

        input_size = self.kwargs.get('input_size')
        if input_size:
            raw_recommended_size = recommended_params['input_size']
            recommended_params['input_size'] = input_size

        total_gpus = self.kwargs.get('gpu_per_trial') * \
            self.kwargs.get('max_concurrency_trials')

        # scale lr
        lr_small = recommended_params['lr'][0]
        if input_size:
            lr_small *= (recommended_params['input_size'][0] * recommended_params['input_size'][1])
            lr_small /= (raw_recommended_size[0] * raw_recommended_size[1])
        lr_small *= recommended_params['batch_size']
        lr_small /= 8.0
        lr_small *= self.kwargs.get('gpu_per_trial')
        lr_small /= 2.0
        lr = [lr_small, lr_small * 16]

        # other params
        recommended_params['lr'] = lr_small * 4
        recommended_params['num_classes'] = len(label_map)
        recommended_params['label_map'] = label_map
        pretrained_weight = self.kwargs.get('pretrained_weight')
        if pretrained_weight:
            recommended_params['pretrained_weight'] = pretrained_weight
        recommended_params['deploy_shapes'] = \
            [1, 3, recommended_params['input_size'][1], recommended_params['input_size'][0]]

        # gen exp.yaml
        exp_generator = EXPGenerator(self.cfg_template, baseline_yaml_path)
        exp_generator.get_params(recommended_params)
        exp_generator.dump(cfg_output_path)

        # gen scheduler.json for hpo
        core_module = self.kwargs.get('core_module')
        seg_module = self.kwargs.get('seg_module')
        hpo_module = self.kwargs.get('hpo_module')
        image = self.kwargs.get('image')

        hpo_generator = HPOSchedulerGenerator()
        hpo_generator.get_params(image=image,
                                 arch=arch,
                                 core_module=core_module,
                                 seg_module=seg_module,
                                 hpo_module=hpo_module)
        hpo_generator.dump(hpo_scheduler_path)

        # gen hpo.yaml
        gen_hpo_file(hpo_output_path=hpo_output_path,
                     hpo_template=self.cluster_hpo_template,
                     base_exp_path=cfg_output_path,
                     val_names=val_names,
                     lr=lr,
                     max_trials=self.kwargs.get('max_trials'),
                     num_gpus=self.kwargs.get('gpu_per_trial'),
                     total_gpus=total_gpus,
                     metric_type=self.kwargs.get('metric'),
                     rerun_best_trial=rerun_best_trial)

        # gen scheduler.json
        seg_scheduler_generator = APPSchedulerGenerator()
        seg_scheduler_generator.get_params(
            image=image,
            arch=arch,
            core_module=core_module,
            app_module=seg_module,
            num_gpus=self.kwargs.get('gpu_per_trial')
        )
        seg_scheduler_generator.dump(sched_output_path)
