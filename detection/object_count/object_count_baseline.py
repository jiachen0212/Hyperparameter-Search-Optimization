import os
import sys
import time
import json
import jinja2
import random
import numpy as np
from tqdm import tqdm
from ruamel import yaml

from .templates import cfg_template, sched_template, hpo_sched_template
from .templates import local_hpo_template, cluster_hpo_template

from SMore_core.utils.config import merge_dict
from SMore_core.utils.registry import build_from_cfg
from SMore_core.dataset.dataset_builder import DATASET
from SMore_core.utils.sm_yaml import replace_scinot_file

from SMore_hpo.baseline.baseline_base import BaselineBase
from SMore_hpo.baseline.baseline_build import BASELINE

from SMore_hpo.default_config.baseline_defaults import BaselineDefaults

EPOCHES = [200]
RESIZE = [1024]


def get_bs(memory, size):
    a = 1278.94 - 1.5e-5 * size
    b = 2.06 + 0.000634 * size
    bs = (memory - a) / b
    return bs


def get_size(memory, bs=4):
    size = memory * 394.638 - 501980.811
    return size


def get_dataset(base_dir, trainset_cfg, base_path='/data'):
    from SMore_det.dataset import LabelmeZipDataset  # noqa

    customized_dir = os.path.join(base_dir, 'customized')

    if os.path.exists(customized_dir):
        sys.path.append(base_dir)
        import customized  # noqa

    if trainset_cfg['type'] == 'LabelmeZipDataset':
        for data in trainset_cfg['data_path']:
            data['path'] = base_path + data['path']
            data['zip_path'] = base_path + data['zip_path']
    elif trainset_cfg['type'] == 'LabelmeDataset':
        for data in trainset_cfg['data_path']:
            data['root'] = base_path + data['root']
            data['path'] = base_path + data['path']
    elif trainset_cfg['type'] == 'OldLabelmeDataset':
        trainset_cfg['data_root'] = base_path + trainset_cfg['data_root']
    else:
        raise NotImplementedError(f"Not support {trainset_cfg['type']}")

    trainset_cfg['transform_cfg'] = []

    dataset = build_from_cfg(module_cfg=trainset_cfg,
                             registry=DATASET,
                             extra_args=None)

    return dataset


def calculate_dataset_statistics(dataset, num_classes):
    from SMore_det.common.constants import DetectionInputsConstants
    pixel_counter = [0] * num_classes
    shape_counter = []
    bbox = []
    num_bbox = 0
    mean_list = []
    std_list = []

    if len(dataset) > 300:
        samples = random.sample(list(range(len(dataset))), 300)
    else:
        samples = list(range(len(dataset)))

    for idx in tqdm(samples):
        data = dataset[idx]
        img = data[DetectionInputsConstants.IMG]
        mean_list.append(np.mean(img / 255., axis=(0, 1)))
        std_list.append(np.std(img / 255., axis=(0, 1)))
        shape_counter.append(data[DetectionInputsConstants.IMG_SHAPE])  # h, w
        counts = np.unique(data[DetectionInputsConstants.LABELS], return_counts=True)
        bbox.extend(data[DetectionInputsConstants.BBOXES])
        num_bbox = max(num_bbox, len(data[DetectionInputsConstants.BBOXES]))
        for idx in range(len(counts[0])):
            try:
                pixel_counter[int(counts[0][idx])] += counts[1][idx]
            except:
                pass

    mean = sum(mean_list) / len(mean_list)
    std = sum(std_list) / len(std_list)

    return pixel_counter, shape_counter, bbox, num_bbox, mean, std


def calculate_recommended_params(dataset, num_classes, arch, max_iter=None):
    """
    Return:
        {'input_size_list':[], 'iter_list':[], 'ckpt_freq':int}
    """
    recommended_params = {}

    pixel_counter, shape_counter, bbox, num_bbox, mean, std = calculate_dataset_statistics(dataset, num_classes)
    # num_data = len(dataset)

    # input size
    avg_w = 0
    avg_h = 0
    for shape in shape_counter:
        avg_h += shape[0]
        avg_w += shape[1]
    avg_h /= len(shape_counter)
    avg_w /= len(shape_counter)

    bbox = np.array(bbox)
    bbox_h = bbox[:, 2] - bbox[:, 0]
    bbox_w = bbox[:, 3] - bbox[:, 1]
    bbox_h = np.concatenate([bbox_h, bbox_w], axis=0)
    bbox_h.sort()
    # calculate minimize size according to box size
    min_h = bbox_h[int(len(bbox_h) * 0.1)]
    min_scale = min_h / 16
    min_h = avg_h / min_scale
    # min_w = avg_w / min_scale
    # min_size = avg_h * avg_w / min_scale / min_scale
    # calculate maximize size according to gpu memory
    if arch == 'turing':
        gpu_total_memory = 11 * 1024
    elif arch == 'ampere':
        gpu_total_memory = 24 * 1024
    elif arch == 'tesla':
        gpu_total_memory = 16 * 1024
    max_size = get_size(gpu_total_memory * 0.7)
    # the maximum size for experiments is set to 2 * origin image size
    max_size = min(max_size, avg_h * avg_w * 2)
    max_w = np.int(np.sqrt(max_size * avg_w / avg_h))
    max_h = np.int(max_w * avg_h / avg_w)
    # size used for hpo are chosen from min to max with a step size of 0.2*max
    img_h_list = [max_h]
    img_w_list = [max_w]
    delta_h = max_h * 0.2
    delta_w = max_w * 0.2
    while max_h - delta_h > min_h:
        max_h -= delta_h
        max_w -= delta_w
        img_h_list.append(max_h)
        img_w_list.append(max_w)
    # calculate batch size and iteration according to image size
    recommended_params['size_bs_iter'] = []
    recommended_params['mean'] = mean
    recommended_params['std'] = std
    for img_h, img_w in zip(img_h_list, img_w_list):
        small = min(img_w, img_h)
        big = max(img_w, img_h)
        # batch_size = int(get_bs(gpu_total_memory * 0.7, img_w * img_h))
        # batch_size = int(np.clip(batch_size, 4, 16))
        # max_iter = int(4000 * 4 / batch_size)
        recommended_params['size_bs_iter'].append({
            'input_size': [[int(small * 0.9),
                            int(small),
                            int(small * 1.1)], big * 1.3],
            'eval_size': [int(small), int(big)],
            'max-iter': 4000 if max_iter is None else max_iter,  # max_iter,
            'batch_size': 4,  # batch_size,
        })
        break
        # currently, use the largest size is good
    recommended_params['num_bbox'] = num_bbox

    return recommended_params


def gen_cfg_file(cfg_output_path, backbone, num_classes, label_map, metric_type,
                 pretrained_weight, baseline_yaml_path, cfg_template, recommended_params):
    with open(baseline_yaml_path, 'r') as baseline_file:
        baseline_file = baseline_file.read()

    # add YAML anchor
    baseline_file = baseline_file.replace('train_set:',
                                          'train_set: &train_set')
    baseline_file = baseline_file.replace('val_set:', 'val_set: &val_set')

    # concat exp.yaml
    baseline_cfg = baseline_file.replace('baseline:', '# baseline')

    batch_size = recommended_params['size_bs_iter'][0]['batch_size']
    eval_size = recommended_params['size_bs_iter'][0]['eval_size']
    input_size = recommended_params['size_bs_iter'][0]['input_size']
    max_iter = recommended_params['size_bs_iter'][0]['max-iter']
    mean = recommended_params['mean']
    std = recommended_params['std']
    # fill placeholder in exp.yaml
    cfg_template = jinja2.Template(cfg_template)
    cfg_file = cfg_template.render(num_classes=num_classes,
                                   backbone=backbone,
                                   size_bs_iter='\'$' + json.dumps(recommended_params['size_bs_iter'][0]) + '\'',
                                   num_bbox=str(recommended_params['num_bbox'] * 2),
                                   label_map=label_map,
                                   eval_size=eval_size,
                                   max_iter=max_iter,
                                   input_size=input_size,
                                   batch_size=batch_size,
                                   mean='[' + ','.join([str(v) for v in mean]) + ']',
                                   std='[' + ','.join([str(v) for v in std]) + ']')

    cfg_file = cfg_file.replace('##place_holder_baseline##', baseline_cfg)

    # change metric
    if metric_type == 'product':
        cfg_file = cfg_file.replace('SimpleInference',
                                    'ProductSimpleInference')
        cfg_file = cfg_file.replace('PixelBasedEvaluator', 'ProductEvaluator')

    if pretrained_weight:
        cfg_file = cfg_file.replace('##place_holder_pretrained_weights##',
                                    'pretrained_weights: ' + pretrained_weight)

    with open(cfg_output_path, 'w') as f:
        f.write(cfg_file)


def gen_hpo_file(
        hpo_output_path,
        hpo_scheduler_path,
        hpo_scheduler_json,
        hpo_template,
        base_exp_path,
        size_bs_iter,
        val_names,
        max_trials=40,
        num_gpus=2,
        max_concurrency_trials=4,
        metric_type='p/r',
        rerun_best_trial=False,
        optimize_mode='maximize',
):
    hpo = yaml.load(hpo_template, Loader=yaml.Loader)

    hpo['common']['seed'] = int(time.time()) % 1000000
    hpo['tuner']['search_space']['size_bs_iter']['value'] = size_bs_iter
    #    hpo['tuner']['search_space']['max-iter']['value'] = iter_list
    hpo['tuner']['optimize_mode'] = optimize_mode
    hpo['manager']['rerun_best_trial'] = rerun_best_trial
    hpo['manager']['max_trials'] = max_trials
    hpo['manager']['max_concurrency_trials'] = max_concurrency_trials
    hpo['manager']['base_exp_path'] = base_exp_path

    if 'resource' in hpo['trial'].keys():
        hpo['trial']['resource'][
            'total_gpus'] = max_concurrency_trials * num_gpus

    metrics = []
    for val_name in val_names:
        metrics.append({
            'data_name': val_name,
            'metric_type': 'DRPEvaluator/all/Recall',
            'metric_weight': 0.5 / len(val_names)
        })
        metrics.append({
            'data_name': val_name,
            'metric_type': 'DRPEvaluator/all/Precision',
            'metric_weight': 0.5 / len(val_names)
        })
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

    with open(hpo_scheduler_path, 'w', encoding='utf-8') as w:
        w.write(hpo_scheduler_json)


def gen_scheduler(sched_output_path, sched_template,
                  image, core_module, det_module, hpo_module, arch, num_gpus=2):
    tm = jinja2.Template(sched_template)
    template = tm.render(image=image,
                         core_module=json.dumps(core_module),
                         det_module=json.dumps(det_module),
                         hpo_module=json.dumps(hpo_module),
                         arch=arch)

    load_scheduler = json.loads(template)

    load_scheduler['train']['GPU'] = num_gpus
    load_scheduler['train']['num_gpus_per_machine'] = num_gpus

    with open(sched_output_path, 'w') as scheduler_f:
        json.dump(load_scheduler, scheduler_f, indent=4)


@BASELINE.register_module()
class ObjectCountBaseline(BaselineBase):
    DEFAULT_CONFIG = BaselineDefaults.DetBaseline_cfg

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def gen_cfgs(self, baseline_yaml_path, hpo_path, rerun_best_trial=False):
        label_map = self.kwargs.get('label_map')
        data_cfg = self.kwargs.get('train_set')
        val_set_cfg = self.kwargs.get('val_set')
        val_names = []
        for val_set in val_set_cfg:
            val_names.append(val_set['data_name'])

        base_dir = os.path.dirname(os.path.abspath(baseline_yaml_path))
        dataset = get_dataset(base_dir, data_cfg, base_path='')
        recommended_params = calculate_recommended_params(dataset, len(label_map), arch=self.kwargs.get('arch'),
                                                          max_iter=self.kwargs.get('max_iter'))

        cfg_output_path = os.path.join(base_dir, 'exp.yaml')
        hpo_output_path = os.path.join(hpo_path, 'hpo.yaml')
        hpo_scheduler_path = os.path.join(hpo_path, 'scheduler.json')
        sched_output_path = os.path.join(base_dir, 'scheduler.json')

        gen_cfg_file(
            cfg_output_path=cfg_output_path,
            backbone=self.kwargs.get('backbone'),
            num_classes=len(label_map),
            label_map=label_map,
            metric_type=self.kwargs.get('metric'),
            pretrained_weight=None,
            baseline_yaml_path=baseline_yaml_path,
            cfg_template=cfg_template,
            recommended_params=recommended_params,
        )

        tm = jinja2.Template(hpo_sched_template)
        hpo_scheduler_json = \
            tm.render(image=self.kwargs.get('image').replace('cu11', 'cu10'),
                      core_module=json.dumps(self.kwargs.get('core_module'), ensure_ascii=False, indent=4),
                      det_module=json.dumps(self.kwargs.get('det_module'), ensure_ascii=False, indent=4),
                      hpo_module=json.dumps(self.kwargs.get('hpo_module'), ensure_ascii=False, indent=4))

        gen_hpo_file(
            hpo_output_path=hpo_output_path,
            hpo_scheduler_path=hpo_scheduler_path,
            hpo_scheduler_json=hpo_scheduler_json,
            hpo_template=cluster_hpo_template,
            base_exp_path=cfg_output_path,
            size_bs_iter=recommended_params['size_bs_iter'],
            val_names=val_names,
            max_trials=self.kwargs.get('max_trials'),
            max_concurrency_trials=self.kwargs.get('max_concurrency_trials'),
            metric_type=self.kwargs.get('metric'),
            rerun_best_trial=rerun_best_trial,
            optimize_mode=self.kwargs.get('optimize_mode'),
        )

        gen_scheduler(sched_output_path,
                      sched_template,
                      image=self.kwargs.get('image'),
                      core_module=self.kwargs.get('core_module'),
                      det_module=self.kwargs.get('det_module'),
                      hpo_module=self.kwargs.get('hpo_module'),
                      arch=self.kwargs.get('arch'))


@BASELINE.register_module()
class LocalObjectCountBaseline(BaselineBase):
    DEFAULT_CONFIG = BaselineDefaults.DetBaseline_cfg

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = merge_dict(BaselineDefaults.DetBaseline_cfg,
                                 self.kwargs)

    def gen_cfgs(self, baseline_yaml_path, hpo_path, rerun_best_trial=False):
        label_map = self.kwargs.get('label_map')
        data_cfg = self.kwargs.get('train_set')
        valset_cfg = self.kwargs.get('val_set')
        val_names = []
        for valset in valset_cfg:
            val_names.append(valset['data_name'])

        base_dir = os.path.dirname(os.path.abspath(baseline_yaml_path))
        dataset = get_dataset(base_dir, data_cfg, base_path='')
        recommanded_params = calculate_recommended_params(
            dataset, len(label_map))

        cfg_output_path = os.path.join(base_dir, 'exp.yaml')
        hpo_output_path = os.path.join(hpo_path, 'hpo.yaml')

        gen_cfg_file(cfg_output_path=cfg_output_path,
                     backbone=self.kwargs.get('backbone'),
                     num_classes=len(label_map),
                     label_map=label_map,
                     metric_type=self.kwargs.get('metric'),
                     pretrained_weight=self.kwargs.get('pretrained_weight'),
                     baseline_yaml_path=baseline_yaml_path,
                     cfg_template=cfg_template,
                     recommanded_params=recommanded_params)

        gen_hpo_file(hpo_output_path=hpo_output_path,
                     hpo_template=local_hpo_template,
                     base_exp_path=cfg_output_path,
                     input_size_list=recommanded_params['input_size_list'],
                     iter_list=recommanded_params['iter_list'],
                     val_names=val_names,
                     max_trials=self.kwargs.get('max_trials'),
                     num_gpus=self.kwargs.get('num_gpus'),
                     total_gpus=self.kwargs.get('total_gpus'),
                     metric_type=self.kwargs.get('metric'),
                     rerun_best_trial=rerun_best_trial)
