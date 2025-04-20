import wandb
import typing as ty
import io
import numpy as np
import torch

from wandb.sdk.wandb_run import Run as WandbRun

from logging import Logger
from .. import Config
from . import ConsoleMixin, Experiment

import PIL.Image as PILImage
import plotly.graph_objects as go
import plotly.tools as tls

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
            config=config.to_dict() if config is not None else None
        )

    def log_metric(self, path: str, value: float,
                   series: str | None = None, step: int | None = None):
        ConsoleMixin.log_metric(self, path, value, series=series, step=step)
        if series is not None:
            path = f"{path}/{series}"
        step = self.wandb_run.step if step is None else step
        assert step >= self.wandb_run.step
        self.wandb_run.log({path: value}, step=step)

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
            assert step >= self.wandb_run.step
            return self.wandb_run.log({path : wandb.Plotly(figure)}, step=step)

    def log_image(self, path: str, image: PILImage.Image | np.ndarray | torch.Tensor,
                    series: str | None = None, step: int | None = None):
        if series is not None:
            path = f"{path}/{series}"
        step = self.wandb_run.step if step is None else step
        assert step >= self.wandb_run.step
        self.wandb_run.log({
            path: wandb.Image(image)
        }, step=step)

    def log_table(self, path: str, table: ty.Any, series: str | None = None, step: int | None = None):
        if series is not None:
            path = f"{path}/{series}"
        step = self.wandb_run.step if step is None else step
        assert step >= self.wandb_run.step
        self.wandb_run.log({
            path: wandb.Table(data=table)
        }, step=step)

    def finish(self):
        self.wandb_run.finish()
