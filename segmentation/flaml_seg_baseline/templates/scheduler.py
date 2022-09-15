scheduler_json = \
    '''
{
  "image": "harbor.smoa.cc/public/smore_core:v2.2.1.cu11",
  "schedctl": "latest",
  "arch": "ampere",
  "core": {
    "type": "pip",
    "repo_name": "SMore_core",
    "version": "2.2.1"
  },
  "submodules": [
    {
      "type": "pip",
      "repo_name": "SMore_seg",
      "version": "2.1.0"
    }
  ],
  "train": {
    "GPU": 2,
    "num_gpus_per_machine": 2,
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
