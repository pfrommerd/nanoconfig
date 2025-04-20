import abc
import torch
import pandas as pd
import typing as ty
import numpy as np
import functools
import logging
import plotly.io as pio
import io

import PIL.Image as PILImage

from logging import Logger
from pathlib import Path

from .. import Config, config, field, utils

class Dummy: ...

class Experiment(abc.ABC):
    def __init__(self, main: ty.Callable | None, config: Config | None) -> None:
        self.main = main
        self.config = config

    def run(self):
        if not self.main:
            return
        return self.main(self)

    def log(self, result: "NestedResult",
                  path: str | None = None,
                  series: str | None = None,
                  step: int | None = None):
        for k, v in utils.flatten_items(result):
            if isinstance(v, Result):
                v.log(self, path=k, step=step, series=series)
            else:
                raise TypeError(f"Unsupported type {type(v)} for logging")

    @abc.abstractmethod
    def log_metric(self, path: str, value: float,
                   series: str |None = None, step: int | None = None): ...

    @abc.abstractmethod
    def log_image(self, path: str, image: PILImage.Image | np.ndarray | torch.Tensor,
                   series: str | None = None, step: int | None = None): ...

    @abc.abstractmethod
    def log_figure(self, path: str, figure: ty.Any | dict,
                   series: str | None = None, step: int | None = None, static: bool = False): ...

    @abc.abstractmethod
    def log_table(self, path: str, table: pd.DataFrame,
                  series: str | None = None, step: int | None = None): ...

class ConsoleMixin:
    def __init__(self, logger : Logger | None = None,
                console_intervals : dict[str, int] = {}):
        self.logger = logger
        self.console_intervals = console_intervals

    def log_metric(self, path: str, value: float,
                   series: str | None = None, step: int | None = None):
        if self.logger is None:
            return
        console_path = path.replace("/", ".")
        if step is not None:
            interval = self.console_intervals.get(series, 1) if series else 1
            if step % interval == 0:
                self.logger.info(f"{step} - {console_path}: {value} ({series})")

class LocalExperiment(ConsoleMixin, Experiment):
    def __init__(self, *, logger : Logger | None = None,
                    console_intervals : dict[str, int] ={},
                    main : ty.Callable | None = None,
                    config: Config | None = None):
        ConsoleMixin.__init__(self, logger, console_intervals)
        Experiment.__init__(self, main=main, config=config)

    def log_figure(self, path: str, figure: ty.Any | dict,
                   series: str | None = None, step: int | None = None,
                   static: bool = False):
        pass

    def log_image(self, path: str, image: PILImage.Image | np.ndarray | torch.Tensor,
                    series: str | None = None, step: int | None = None):
        pass

    def log_table(self, path: str, table: pd.DataFrame,
                  series: str | None = None, step: int | None = None):
        pass

class Result(abc.ABC):
    @abc.abstractmethod
    def log(self, experiment: Experiment, path: str,
            series: str | None = None, step: int | None = None): ...

NestedResult = dict[str, "NestedResult"] | Result

class Metric(Result):
    def __init__(self, value: float):
        self.value = value

    def log(self, experiment: Experiment, path: str,
            series: str | None = None, step: int | None = None):
        experiment.log_metric(path, self.value, series=series, step=step)

class Table(Result):
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def log(self, experiment: Experiment, path: str,
            series: str | None = None, step: int | None = None):
        experiment.log_table(path, self.dataframe,
                             series=series, step=step)

class Figure(Result):
    def __init__(self, figure, static = False):
        self.figure = figure
        self.static = static

    def log(self, experiment: Experiment, path: str,
            series: str | None = None, step: int | None = None):
        experiment.log_figure(path, self.figure, series=series, step=step, static=self.static)

    def _display_(self):
        return self.figure

@config
class ExperimentConfig:
    project: str | None = None

    console: bool = True
    remote: bool = False
    queue: str = "default"

    clearml: bool = False
    wandb: bool = False

    console_intervals: dict[str, int] = field(default_factory=lambda: {
        "train": 100,
        "test": 1
    })

    def create(self, logger : Logger | None = None,
                main: ty.Callable | None = None,
                config: Config | None = None) -> Experiment:
        if not self.console:
            logger = None
        elif logger is None:
            logger = logging.getLogger(__name__)
        experiments = []
        if self.clearml:
            from .clearml import ClearMLExperiment
            return ClearMLExperiment(
                project_name=self.project,
                logger=logger, remote=self.remote,
                console_intervals=self.console_intervals,
                main=main,
                config=config
            )
        elif self.wandb:
            from .wandb import WandbExperiment
            return WandbExperiment(
                project_name=self.project,
                logger=logger,
                console_intervals=self.console_intervals,
                main=main,
                config=config
            )
        else:
            return LocalExperiment(
                logger=logger,
                console_intervals=self.console_intervals,
                main=main,
                config=config
            )
