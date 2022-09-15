scheduler_json = \
    '''
{
  "image": "10.81.138.201:5000/smore_core:v2.1.3.cu10",
  "schedctl": "/data/home/sharedir/industrial/xifang/schedctl",
  "core": {
    "type": "pip",
    "repo_name": "SMore_core",
    "version": "2.1.3"
  },
  "submodules": [
    {
        "type": "pip",
        "repo_name": "SMore_cls",
        "version": "2.1.0"
    }
  ],
  "train": {
    "GPU": 1,
    "num_gpus_per_machine": 1,
    "num_machines": 1,
    "workers": 2,
    "force": true
  },
  "evaluate": {
    "GPU": 1,
    "num_gpus_per_machine": 1,
    "iter": -1,
    "dump_vis": true,
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
