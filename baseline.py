import os
import sys
import click
from ruamel import yaml

from .baseline_build import build_baseline
from .segmentation import * # noqa
from .classification import * # noqa
from .detection import * # noqa
from SMore_hpo.api.tune import _tune as tune


def gen_baseline(baseline_path, hpo_path, rerun_best_trial):
    ''' generate baseline exp.yaml, hpo.yaml and scheduler.json '''

    customized_dir = os.path.join(baseline_path, 'customized')
    if os.path.exists(customized_dir):
        sys.path.append(baseline_path)
        import customized # noqa

    baseline_yaml_path = os.path.join(baseline_path, 'baseline.yaml')
    with open(baseline_yaml_path, 'r') as yaml_file:
        baseline_cfg = yaml.load(yaml_file.read(), Loader=yaml.Loader)['baseline']
    os.makedirs(hpo_path, exist_ok=True)
    baseline = build_baseline(baseline_cfg)
    baseline.gen_cfgs(baseline_yaml_path, hpo_path, rerun_best_trial)


@click.command(help='run baseline')
@click.argument('baseline_path')  # folder for baseline.yaml
@click.argument('hpo_path', default='.')  # folder for hpo exp
@click.option('--extra_exp', '-e', is_flag=True, default=False, help='rerun best exp using 4 cards')
@click.option('--cfg_only', '-c', is_flag=True, default=False, help='only generate hpo and exp config files')
def baseline(baseline_path, hpo_path, extra_exp, cfg_only):
    gen_baseline(baseline_path, hpo_path, extra_exp)
    if not cfg_only:
        tune(hpo_path)
