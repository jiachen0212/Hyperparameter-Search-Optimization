scheduler_json = \
    '''
{
  "image": "{{ image }}",
  "schedctl": "latest",
  "arch": "{{ arch }}",
  "core": {{ core_module }},
  "submodules": [
    {{ det_module }}
  ],
  "train": {
    "GPU": 2,
    "num_gpus_per_machine": 2,
    "num_machines": 1,
    "workers": 4
  },
  "evaluate": {
    "GPU": 1,
    "num_gpus_per_machine": 1,
    "iter": -1,
    "force": true
  },
  "visualize": {
    "iter": -1,
    "GPU": 1
  },
  "deploy": {
    "GPU": 1,
    "num_gpus_per_machine": 1,
    "iter": -1,
    "force": true
  }
}
    '''

hpo_scheduler_template = \
    '''
{
  "image": "{{ image }}",
  "schedctl": "latest",
  "arch": "turing",
  "core": {{ core_module }},
  "submodules": [
    {{ det_module }},
    {{ hpo_module }}
  ],
  "tune": {}
}
'''
