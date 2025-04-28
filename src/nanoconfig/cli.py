import argparse
import click
import logging
import rich
import tempfile

from rich.logging import RichHandler
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem

from .data.generator import GeneratorSource
from .data.source import DataRepository
from .data.fs import FsDataRepository
from .data.visualizer import DataVisualizer
from .data.source import DataSource

logger = logging.getLogger(__name__)

@click.argument("args", nargs=-1)
@click.argument("cmd", type=str)
@click.argument("name", type=str)
@click.command("generate")
def generate_data(name, cmd, args):
    source = GeneratorSource.from_command(cmd, *args)
    repo = DataRepository.default()
    if repo.lookup(source) is not None:
        repo.register(name, source.sha256)
        logger.info(f"Data already exists: {source.sha256}")
    else:
        data = repo.get(source)
        repo.register(name, data)
        logger.info(f"Generated data: {data.sha256}")

@click.argument("url", type=str)
@click.argument("name", type=str)
@click.command("pull")
def pull_data(name, url):
    source = DataSource.from_url(url)
    repo = DataRepository.default()
    if repo.lookup(source) is not None:
        repo.register(name, source.sha256)
        logger.info(f"Data already exists: {source.sha256}")
    else:
        data = repo.get(source)
        repo.register(name, data)
        logger.info(f"Pulled data: {data.sha256}")

@click.command("list")
def list_data():
    repo = DataRepository.default()
    keys = repo.keys()
    if isinstance(repo, FsDataRepository) and \
            isinstance(repo.fs, DirFileSystem) and \
            isinstance(repo.fs.fs, LocalFileSystem):
        rich.print(f"Repoistory: {repo.fs.path}") # type: ignore
    else:
        rich.print(f"Repoistory: {repo}")
    for key in keys:
        data = repo.lookup(key)
        if data is not None:
            rich.print(f"  [green]{key}[/green]: {data.sha256}")
            for split_info in data.split_infos().values():
                rich.print(f"    - [yellow]{split_info.name}[/yellow]: {split_info.size} ({split_info.mime_type})")

@click.argument("keys", nargs=-1)
@click.command("rm")
def remove_data(keys):
    repo = DataRepository.default()
    for key in keys:
        data = repo.lookup(key)
        if data is None:
            rich.print(f"Data not found: {key}")
            return
    for key in keys:
        repo.deregister(key)
    repo.gc()
    rich.print(f"{" ".join(keys)}")

@click.option("--port", default=8000)
@click.option("--host", default="127.0.0.1")
@click.option("--visualizer", default="nanoconfig.data.visualizer:DataVisualizer")
@click.argument("data")
@click.command("visualize")
def visualize_data(data, visualizer, host, port):
    repo = DataRepository.default()
    data = repo.lookup(data)
    if data is None:
        rich.print(f"Data not found: {data}")
        return
    DataVisualizer.host_marimo_notebook(host, port, visualizer, data)

@click.group()
def data():
    setup_logging()

data.add_command(list_data)
data.add_command(pull_data)
data.add_command(remove_data)
data.add_command(generate_data)
data.add_command(visualize_data)

class CustomLogRender(rich._log_render.LogRender): # type: ignore
    def __call__(self, *args, **kwargs):
        output = super().__call__(*args, **kwargs)
        if not self.show_path:
            output.expand = False
        return output

FORMAT = "%(name)s - %(message)s"

def setup_logging(show_path=False):
    # add_log_level("TRACE", logging.DEBUG - 5)
    logging.getLogger("nanoconfig").setLevel(logging.INFO)
    if rich.get_console().is_jupyter:
        return rich.reconfigure(
            force_jupyter=False,
        )
    console = rich.get_console()
    handler = RichHandler(
        markup=True,
        rich_tracebacks=True,
        show_path=show_path,
        console=console
    )
    renderer = CustomLogRender(
        show_time=handler._log_render.show_time,
        show_level=handler._log_render.show_level,
        show_path=handler._log_render.show_path,
        time_format=handler._log_render.time_format,
        omit_repeated_times=handler._log_render.omit_repeated_times,
    )
    handler._log_render = renderer
    logging.basicConfig(
        level=logging.WARNING,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[handler]
    )
