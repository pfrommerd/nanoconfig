from pyparsing import C
import wandb
import typing as ty
import os
import io
import numpy as np
import torch
import pandas as pd

from wandb.sdk import wandb_run
from wandb.sdk.lib.printer import WARN
from wandb.sdk.wandb_run import Run as WandbRun

from logging import Logger
from pathlib import Path

from .. import Config
from . import ConsoleMixin, Experiment, Artifact, ArtifactBuilder, ArtifactInfo

import PIL.Image as PILImage
import plotly.graph_objects as go
import plotly.tools as tls
import contextlib
import tempfile

CACHE_DIR = Path.home() / ".cache" / "nanoconfig" / "wandb"

class WandbArtifact(Artifact):
    def __init__(self, wandb_artifact: wandb.Artifact):
        name = wandb_artifact.name
        if ":" in name:
            name, _ = name.split(":")
        path = f"{wandb_artifact.entity}/{wandb_artifact.project}/{name}:{wandb_artifact.version}"
        super().__init__(
            name=wandb_artifact.name,
            path=path,
            type=wandb_artifact.type,
            version=wandb_artifact.version,
            digest=wandb_artifact.digest
        )
        self.wandb_artifact = wandb_artifact

    @contextlib.contextmanager
    def open_file(self, path: str) -> ty.Iterator[io.BufferedReader]:
        entry = self.wandb_artifact.get_entry(path)
        local_path = entry.download()
        with open(local_path, "rb") as f:
            yield f

class WandbArtifactBuilder(ArtifactBuilder):
    def __init__(self, wandb_artifact: wandb.Artifact, run: WandbRun):
        super().__init__(
            name=wandb_artifact.name,
            type=wandb_artifact.type,
        )
        self.wandb_artifact = wandb_artifact
        self.wandb_run = run
        self._temporary_files = set()
        self.result = None

    def _cleanup_temp(self):
        for file in self._temporary_files:
            os.remove(file)
        self._temporary_files.clear()

    @contextlib.contextmanager
    def create_file(self, name: str):
        if self.wandb_artifact is None:
            raise ValueError("WandbArtifact has already been built")
        local_path = tempfile.mktemp()
        self._temporary_files.add(local_path)
        with open(local_path, "wb") as f:
            yield f
        self.wandb_artifact.add_file(local_path, name, overwrite=True)

    def build(self) -> WandbArtifact:
        if self.wandb_artifact is not None:
            self._cleanup_temp()
            artifact = self.wandb_run.log_artifact(self.wandb_artifact)
            artifact.wait()
            self.result = WandbArtifact(artifact)
            self.wandb_artifact = None
        assert self.result is not None
        return self.result

class WandbExperiment(Experiment, ConsoleMixin):
    def __init__(self, *,
                logger: Logger | None = None,
                console_intervals: dict[str,int] = {},
                project_name: str | None = None,
                run_name: str | None = None,
                entity: str | None = None,
                main: ty.Callable | None = None,
                config: Config | None = None,
                run: WandbRun | None = None
            ):
        ConsoleMixin.__init__(self, logger=logger, console_intervals=console_intervals)
        Experiment.__init__(self, main=main, config=config)

        self.wandb_run = run if run is not None else wandb.init(
            project=project_name,
            entity=entity,
            name=run_name,
            config=config.to_dict() if config is not None else None,
            reinit="finish_previous"
        )
        self._step_offset = 0

    @property
    def step(self) -> int:
        return self.wandb_run.step - self._step_offset

    def step_offset(self, offset: int = 0) -> int:
        self._step_offset += offset
        return self._step_offset

    def find_artifact(self, name: str, version: str | None = None,
                        type: str | None = None) -> ArtifactInfo | None:
        version = version or "latest"
        artifact = self.wandb_run._public_api().artifact(f"{name}:{version}", type)
        if artifact is None:
            return None
        name = artifact.name
        if ":" in name:
            name, _ = name.split(":")
        path = f"{artifact.entity}/{artifact.project}/{name}:{artifact.version}"
        return ArtifactInfo(name, path, artifact.type, artifact.version, artifact.digest) # type: ignore

    def use_artifact(self, artifact: ArtifactInfo) -> Artifact | None:
        wandb_artifact = self.wandb_run.use_artifact(artifact.path, artifact.type)
        name = wandb_artifact.name
        if ":" in name: name, _ = name.split(":")
        assert name == artifact.name
        assert wandb_artifact.type == artifact.type
        assert wandb_artifact.version == artifact.version
        assert wandb_artifact.digest == artifact.digest
        return WandbArtifact(wandb_artifact)

    @contextlib.contextmanager
    def create_artifact(self, name: str, type: str,
            aliases: list[str] | None = None) -> ty.Iterator[WandbArtifactBuilder]:
        wandb_artifact = wandb.Artifact(name, type)
        builder = WandbArtifactBuilder(wandb_artifact, self.wandb_run)
        try:
            yield builder
        finally:
            # Ensure the artifact is logged
            builder.build()

    def log_metric(self, path: str, value: float,
                   series: str | None = None, step: int | None = None):
        ConsoleMixin.log_metric(self, path, value, series=series, step=step)
        if series is not None:
            path = f"{path}/{series}"
        step = self.wandb_run.step if step is None else step
        assert step + self._step_offset >= self.wandb_run.step
        self.wandb_run.log({path: value}, step=step + self._step_offset)

    def log_figure(self, path: str, figure : ty.Any, series: str | None = None, step: int | None = None,
                            static: bool = False):
        if static:
            img_bytes = PILImage.open(io.BytesIO(figure.to_image(format="jpg")))
            return self.log_image(path, img_bytes, series=series, step=step)
        else:
            if not isinstance(figure, (go.Figure, dict)):
                figure = tls.mpl_to_plotly(figure)
            if series is not None:
                path = f"{path}/{series}"
            step = self.wandb_run.step if step is None else step
            assert step + self._step_offset >= self.wandb_run.step
            return self.wandb_run.log({path : wandb.Plotly(figure)}, step=step + self._step_offset)

    def log_image(self, path: str, image: PILImage.Image | np.ndarray | torch.Tensor,
                    series: str | None = None, step: int | None = None):
        if series is not None:
            path = f"{path}/{series}"
        step = self.wandb_run.step if step is None else step
        assert step + self._step_offset >= self.wandb_run.step
        self.wandb_run.log({
            path: wandb.Image(image)
        }, step=step + self._step_offset)

    def log_table(self, path: str, table: pd.DataFrame, series: str | None = None, step: int | None = None):
        if series is not None:
            path = f"{path}/{series}"
        step = self.wandb_run.step if step is None else step
        assert step + self._step_offset >= self.wandb_run.step
        # Convert all PIL images to wandb.Image
        for column, type in zip(table.columns, table.dtypes):
            if type == object:
                table[column] = table[column].apply(lambda x: wandb.Image(x) if isinstance(x, PILImage.Image) else x)
        self.wandb_run.log({
            path: wandb.Table(dataframe=table)
        }, step=step + self._step_offset)

    def finish(self):
        self.wandb_run.finish()
