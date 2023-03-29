import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json


class MyConfigParser:
    def __init__(self, config, task_id=None, resume=None):
        self.log_dir = None
        self.resume = resume
        self._config = config
        save_dir = Path(self._config['trainer']['save_dir'])
        task_name = self._config['name']
        if task_id is None:
            task_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self._save_dir = save_dir / 'models' / task_name / task_id
        self._log_dir = save_dir / 'log' / task_name / task_id

        exist_ok = task_id == ''
        self.get_save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.get_log_dir.mkdir(parents=True, exist_ok=exist_ok)
        write_json(self._config, self.get_save_dir / 'config.json')

        setup_logging(self.get_log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    def init_ftn(self, name, module, *args, **kwargs):
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs])
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def init_obj(self, name, module, *args, **kwargs):
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs])
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        return self._config[name]

    def get_logger(self, name, verbosity=2):
        assert verbosity in self.log_levels
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    @classmethod
    def from_json(cls, config_file):
        cfg_fname = Path(config_file)
        config = read_json(cfg_fname)
        return cls(config)

    @classmethod
    def from_yaml(cls):
        # TODO 有空再写
        pass

    @property
    def get_config(self):
        return self._config

    @property
    def get_save_dir(self):
        return self._save_dir

    @property
    def get_log_dir(self):
        return self._log_dir

# def _update_config(config, modification):
#     if modification is None:
#         return config
#     for k, v in modification.items():
#         if v is not None:
#             _set_by_path(config, k, v)
#     return config
#
#
# def _get_opt_name(flags):
#     for flg in flags:
#         if flg.startswith('--'):
#             return flg.replace('--', '')
#     return flags[0].replace('--', '')
#
#
# def _set_by_path(tree, keys, value):
#     keys = keys.split(';')
#     _get_by_path(tree, keys[:-1])[keys[-1]] = value
#
#
# def _get_by_path(tree, keys):
#     return reduce(getitem, keys, tree)
