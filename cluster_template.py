app_scheduler_template = \
    '''
{
  "image": "{{ image }}",
  "schedctl": "latest",
  "arch": "{{ arch }}",
  "core": {{ core_module }},
  "submodules": [
    {{ app_module }}
  ],
  "train": {
    "GPU": {{ num_gpus }},
    "num_gpus_per_machine": {{ num_gpus }},
    "num_machines": 1,
    "workers": 8,
    "no_notify": true
  },
  "evaluate": {
    "GPU": 1,
    "num_gpus_per_machine": 1,
    "iter": -1,
    "force": true,
    "workers": 8,
    "dump_vis": true,
    "no_notify": true
  },
  "visualize": {
    "iter": -1,
    "GPU": 1,
    "no_notify": true
  },
  "deploy": {
    "GPU": 1,
    "num_gpus_per_machine": 1,
    "iter": -1,
    "force": true,
    "no_notify": true
  },
  "analyze": {
    "GPU": 1,
    "no_notify": true
  }
}
'''


hpo_scheduler_template = \
    '''
{
  "image": "{{ image }}",
  "schedctl": "latest",
  "arch": "{{ arch }}",
  "core": {{ core_module }},
  "submodules": [
    {{ app_module }},
    {{ hpo_module }}
  ],
  "tune": {}
}
'''
