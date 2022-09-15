from typing import Dict
from SMore_core.utils.registry import Registry, build_from_cfg

BASELINE = Registry('BASELINE')


def build_baseline(model_cfg: Dict, **kwargs):
    return build_from_cfg(model_cfg, BASELINE, extra_args=kwargs, ignore_extra_keys=True)
