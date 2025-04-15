#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.parent.parent.resolve()
import sys
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "nanoconfig" / "src"))

import clearml
import importlib
import logging
import typing as ty

import nanoconfig.utils
from nanoconfig import Config
from nanoconfig.experiment.clearml import ClearMLExperiment

def run():
    task : clearml.Task = clearml.Task.init(
        auto_connect_arg_parser=False,
        auto_connect_streams={
            "stdout": False,
            "stderr": True,
            "logging": True
        }
    )
    params : dict = nanoconfig.utils.unflatten_dict(task.get_parameters()) # type: ignore
    task_params = params["Task"]
    config_params = params["General"] if "General" in params else {}

    main : str = task_params["main"]
    func_module, func_name = main.split(":")
    main_func : ty.Callable = getattr(importlib.import_module(func_module), func_name)

    logger = logging.getLogger(task_params["logger_name"]) if "logger_name" in task_params else None
    console_intervals = ({k: int(v) for k, v in task_params["console_intervals"].items()}
        if "console_intervals" in task_params else {})
    if "config_type" in task_params:
        config_type_name : str = task_params["config_type"]
        idx = config_type_name.rfind(".")
        config_module, config_class = config_type_name[:idx], config_type_name[idx+1:]
        config_type : ty.Type = getattr(importlib.import_module(config_module), config_class)
        assert issubclass(config_type, Config)
        config = config_type.from_dict(config_params)
    else:
        config = None

    experiment = ClearMLExperiment(
        logger=logger,
        console_intervals=console_intervals,
        remote=True,
        task=task,
        main=main_func,
        config=config
    )
    main_func(experiment) # type: ignore

if __name__=="__main__":
    run()
