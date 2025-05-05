import abc
import torch
import pandas as pd
import typing as ty
import numpy as np
import functools
import logging
import plotly.io as pio
import io
import contextlib

import PIL.Image as PILImage

from logging import Logger
from pathlib import Path
from dataclasses import dataclass

from .. import Config, config, field, utils

class Dummy: ...

@dataclass(frozen=True)
class ArtifactInfo:
    name: str
    path: str
    type: str
    version: str
    digest: str

class Artifact(abc.ABC):
    def __init__(self, name: str, path: str, type: str, version: str,  digest: str):
        self.name = name
        self.path = path
        self.type = type
        self.version = version
        self.digest = digest

    @property
    def info(self) -> ArtifactInfo:
        return ArtifactInfo(self.name, self.path, self.type, self.version, self.digest)
#
    @abc.abstractmethod
    def open_file(self, path: str) -> ty.ContextManager[io.BufferedReader]:
        ...

class ArtifactBuilder(abc.ABC):
    def __init__(self, name: str, type: str):
        self.name = name
        self.type = type

    @abc.abstractmethod
    def create_file(self, name: str) -> ty.ContextManager[io.BufferedWriter]:
        ...

    @abc.abstractmethod
    def build(self) -> Artifact:
        ...

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
                continue
            else:
                try:
                    v = float(v)
                    self.log_metric(path=k, value=v, series=series, step=step)
                    continue
                except ValueError:
                    pass
            raise TypeError(f"Unsupported type {type(v)} for logging")

    @abc.abstractmethod
    def step_offset(self, offset: int = 0) -> int:
        ...

    @property
    @abc.abstractmethod
    def step(self) -> int:
        ...

    @abc.abstractmethod
    def find_artifact(self, name: str, version: str | None = None, type: str | None = None) -> ArtifactInfo | None:
        ...

    @abc.abstractmethod
    def use_artifact(self, artifact: ArtifactInfo) -> Artifact | None:
        ...

    @abc.abstractmethod
    def create_artifact(self, name: str, type: str) -> ty.ContextManager[ArtifactBuilder]:
        ...

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
        self._step_offset = 0
        self._step = 0

    @property
    def step(self) -> int:
        return self._step - self._step_offset

    def step_offset(self, offset: int = 0) -> int:
        self._step_offset += offset
        return self._step_offset

    @contextlib.contextmanager
    def create_artifact(self, name: str, type: str) -> ty.Iterator[ArtifactBuilder]:
        yield None # type: ignore

    def find_artifact(self, name: str, version: str | None = None,
                      type: str | None = None) -> ArtifactInfo | None: ...
    def use_artifact(self, artifact: ArtifactInfo) -> Artifact | None: ...

    def log_metric(self, path: str, value: float, series: str | None = None, step: int | None = None):
        self._step = max(self._step, step or self._step)
        super().log_metric(path, value, series, step)

    def log_figure(self, path: str, figure: ty.Any | dict,
                   series: str | None = None, step: int | None = None,
                   static: bool = False):
        self._step = max(self._step, step or self._step)

    def log_image(self, path: str, image: PILImage.Image | np.ndarray | torch.Tensor,
                    series: str | None = None, step: int | None = None):
        self._step = max(self._step, step or self._step)

    def log_table(self, path: str, table: pd.DataFrame,
                  series: str | None = None, step: int | None = None):
        self._step = max(self._step, step or self._step)

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

    def _display_(self):
        return self.dataframe

class Figure(Result):
    def __init__(self, figure, static = False):
        self.figure = figure
        self.static = static

    def log(self, experiment: Experiment, path: str,
            series: str | None = None, step: int | None = None):
        experiment.log_figure(path, self.figure, series=series, step=step, static=self.static)

    def _display_(self):
        return self.figure

class Image(Result):
    def __init__(self, image):
        self.image = image

    def log(self, experiment: Experiment, path: str,
            series: str | None = None, step: int | None = None):
        image = self.image
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        if image.dtype == np.float32:
            image = np.nan_to_num((image*255).clip(0, 255), nan=0., posinf=255., neginf=0.)
            image = image.astype(np.uint8).squeeze(-1)
        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image)
        experiment.log_image(path, image, series=series, step=step)

    def _display_(self):
        image = self.image
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image)
        return image

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
            ) # type: ignore
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
