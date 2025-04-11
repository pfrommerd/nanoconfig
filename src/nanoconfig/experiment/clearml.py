from . import Experiment, ConsoleMixin

from pathlib import Path
from logging import Logger

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
                root: Path | None = None,
                main: ty.Callable | None = None,
                args: dict[str, ty.Any] = {}
            ):
        ConsoleMixin.__init__(self, logger=logger, console_intervals=console_intervals)
        Experiment.__init__(self, main=main, args=args)

        self.remote = remote
        self.queue = queue
        self.root = root
        self.task = clearml.Task.init(
            project_name=project_name,
            auto_connect_streams={
                "stdout": False,
                "stderr": True,
                "logging": True
            }
        )
        self.task_logger = self.task.get_logger()

    def run(self) -> ty.Any:
        if self.remote:
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
