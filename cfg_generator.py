import jinja2
import json

from SMore_hpo.baseline.cluster_template import app_scheduler_template, hpo_scheduler_template


class HPOSchedulerGenerator():
    "generate cluster scheduler.json for hpo"

    def __init__(self, template=hpo_scheduler_template):
        super().__init__()
        self.template = template

    def get_params(self, **kwargs):
        self.image = kwargs.get('image')
        self.arch = kwargs.get('arch')
        if self.arch != 'ampere':
            self.image = self.image.replace('cu11', 'cu10')
        self.core_module = kwargs.get('core_module')
        self.app_module = kwargs.get('app_module')
        self.hpo_module = kwargs.get('hpo_module')

    def dump(self, dump_path):
        tm = jinja2.Template(self.template)
        hpo_scheduler_json = tm.render(
            image=self.image,
            arch=self.arch,
            core_module=json.dumps(self.core_module, ensure_ascii=False, indent=4),
            app_module=json.dumps(self.app_module, ensure_ascii=False, indent=4),
            hpo_module=json.dumps(self.hpo_module, ensure_ascii=False, indent=4),
        )
        with open(dump_path, 'w', encoding='utf-8') as w:
            w.write(hpo_scheduler_json)


class APPSchedulerGenerator():
    "generate cluster scheduler.json for cls/seg/det/..."

    def __init__(self, template=app_scheduler_template):
        self.template = template

    def get_params(self, **kwargs):
        self.image = kwargs.get('image')
        self.arch = kwargs.get('arch')
        if self.arch != 'ampere':
            self.image = self.image.replace('cu11', 'cu10')
        self.core_module = kwargs.get('core_module')
        self.app_module = kwargs.get('app_module')
        self.num_gpus = str(kwargs.get('num_gpus'))

    def dump(self, dump_path):
        tm = jinja2.Template(self.template)
        app_scheduler_json = tm.render(
            image=self.image,
            arch=self.arch,
            core_module=json.dumps(self.core_module, ensure_ascii=False, indent=4),
            app_module=json.dumps(self.app_module, ensure_ascii=False, indent=4),
            num_gpus=self.num_gpus,
        )
        with open(dump_path, 'w', encoding='utf-8') as w:
            w.write(app_scheduler_json)


class BaseGenerator():
    "generate hpo.yaml or exp.yaml"

    def __init__(self, template):
        self.template = template

    def get_params(self, param_dict):
        self.param_dict = param_dict

    def dump(self, dump_path):
        tm = jinja2.Template(self.template)
        cfg = tm.render(**self.param_dict)
        with open(dump_path, 'w', encoding='utf-8') as w:
            w.write(cfg)
