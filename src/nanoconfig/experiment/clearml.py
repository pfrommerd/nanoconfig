from . import Experiment, ConsoleMixin
from .. import Config, utils

from pathlib import Path
from logging import Logger

import subprocess
import functools
import clearml
import typing as ty

####### ClearML Integration #######

class ClearMLExperiment(Experiment, ConsoleMixin):
    def __init__(self, *,
                logger: Logger | None = None,
                console_intervals: dict[str,int] = {},
                project_name : str | None = None,
                # For remote execution
                remote: bool = False,
                queue: str = "default",
                # For use with run()
                main: ty.Callable | None = None,
                config : Config | None = None,
                task: clearml.Task | None = None
            ):
        ConsoleMixin.__init__(self, logger=logger, console_intervals=console_intervals)
        Experiment.__init__(self, main=main, config=config)

        self.remote = remote
        self.queue = queue
        self.task : clearml.Task = task if task else clearml.Task.init(
            project_name=project_name,
            auto_connect_arg_parser=False,
            auto_connect_streams={
                "stdout": False,
                "stderr": True,
                "logging": True
            }
        )
        params = {}
        if self.config is not None:
            params.update({f"General/{k}": v for k, v in
                utils.flatten_dict(self.config.to_dict()).items()
            })
        if self.logger is not None:
            params["Task/logger_name"] = self.logger.name
        if self.console_intervals:
            for k, v in self.console_intervals.items():
                params[f"Task/console_intervals.{k}"] = v
        if self.main:
            params["Task/main"] = str(self.main.__module__) + ":" + self.main.__qualname__
        if self.config is not None:
            params["Task/config_type"] = str(type(self.config).__module__) + "." + type(self.config).__qualname__
        self.task.set_parameters(params)
        self.task_logger = self.task.get_logger()

    def run(self) -> ty.Any:
        if self.remote:
            launch_script = (Path(__file__) / ".." / "util" / "clearml_launcher.py").resolve()
            launch_relative = launch_script.relative_to(Path.cwd())
            # Get the requirements from uv directly
            requirements = subprocess.run(["uv", "pip", "freeze", "--exclude-editable"], check=True, capture_output=True)
            requirements = requirements.stdout.decode("utf-8").split("\n")
            requirements = [r for r in requirements if r.strip() != ""]
            self.task.set_packages(
                packages=requirements
            )
            self.task.set_script(
                entry_point=str(launch_relative)
            )
            self.task.execute_remotely(
                queue_name=self.queue,
            )
        else:
            super().run()

    def set_parameters(self, parameters: dict[str, ty.Any]):
        self.task.set_parameters(parameters)

    def log_metric(self, path: str, value: float,
                   series: str | None = None, step: int | None = None):
        ConsoleMixin.log_metric(self, path, value, series=series, step=step)
        self.task_logger.report_scalar(path,
            series if series is not None else "train", value,
            iteration=step if step is not None else 0
        )

    def log_figure(self, path: str, figure : ty.Any, series: str | None = None, step: int | None = None):
        self.task_logger.report_plotly(path,
            series if series is not None else "train",
            figure=figure, iteration=step if step is not None else 0
        )

    def log_table(self, path: str, table: ty.Any, series: str | None = None, step: int | None = None):
        self.task_logger.report_table(
            path, series if series is not None else "train",
            table_plot=table, iteration=step if step is not None else 0
        )
