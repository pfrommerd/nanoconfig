import argparse
import click
import logging
import rich
import tempfile
import json
import importlib

from rich.logging import RichHandler
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem

from .data.generator import GeneratorSource
from .data.source import DataRepository
from .data.transform import DataPipeline, SetMimeType
from .data.fs import FsDataRepository
from .data.visualizer import DataVisualizer
from .data.source import DataSource

logger = logging.getLogger(__name__)


@click.group()
def data():
    setup_logging()

@click.option("--transform", "-t", type=str, multiple=True)
@click.option("--mime-type", "-m", type=str, default=None)
@click.argument("args", nargs=-1)
@click.argument("cmd", type=str)
@click.argument("name", type=str)
@data.command()
def generate(name, cmd, args, transform, mime_type):
    """Generate data using a command."""
    source = GeneratorSource.from_command(cmd, *args)
    transforms = _parse_constructors(transform)
    if mime_type:
        transforms.append(SetMimeType(mime_type))
    if transforms: source = DataPipeline(source, *transforms)
    repo = DataRepository.default()
    if repo.lookup(source) is not None:
        repo.register(name, source.sha256)
        rich.print("="*80)
        rich.print(f"Data already exists:")
    else:
        data = repo.get(source)
        repo.register(name, data)
        rich.print("="*80)
        rich.print(f"Generated data:")
    info.callback(name) # type: ignore

@click.option("--transform", "-t", type=str, multiple=True)
@click.option("--mime-type", "-m", type=str, default=None)
@click.argument("url", type=str)
@click.argument("name", type=str)
@data.command()
def pull(name, url, transform, mime_type):
    """Fetch data from a remote source."""
    source = DataSource.from_url(url)
    transforms = _parse_constructors(transform)
    if mime_type:
        transforms.append(SetMimeType(mime_type))
    if transforms: source = DataPipeline(source, *transforms)
    repo = DataRepository.default()
    if repo.lookup(source) is not None:
        repo.register(name, source.sha256)
        rich.print("="*80)
        rich.print(f"Using cached data:")
    else:
        data = repo.get(source)
        repo.register(name, data)
        rich.print("="*80)
        rich.print(f"Pulled data:")
    info.callback(name) # type: ignore

@data.command("list")
def list_():
    """List all registered data keys."""
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
            rich.print(f"  [green]{key}[/green]: [blue]{data.sha256}[/blue]")
            for split_info in data.split_infos().values():
                rich.print(f"    - [yellow]{split_info.name}[/yellow]: {split_info.size} ({split_info.mime_type})")

data.add_command(list_, name="ls")

@click.argument("name")
@data.command()
def info(name):
    """Get detailed schema information about a dataset."""
    repo = DataRepository.default()
    data = repo.lookup(name)
    if data is None:
        rich.print(f"Data not found: {name}")
        return
    rich.print(f"[green]{name}[/green]: [blue]{data.sha256}[/blue]")
    for split_info in data.split_infos().values():
        rich.print(f"  [yellow]{split_info.name}[/yellow]: {split_info.size} ({split_info.mime_type})")
        schema = split_info.schema
        for field_name in schema.names:
            field = schema.field(field_name)
            # field_type = field.type
            field_mime = field.metadata.get(b"mime_type", b"unknown").decode()
            metadata = {k.decode(): v.decode()[:64] + "..." if len(v) > 64 else v.decode() for k, v in field.metadata.items()} \
                if field.metadata else {}
            field_metadata = (
                ", ".join(f"{k}={v}" for k, v in metadata.items() if k != "mime_type")
            )
            rich.print(f"    - [blue]{field_name}[/blue]: {field.type} ({field_mime})")
            for k, v in metadata.items():
                if k != "mime_type":
                    rich.print(f"        {k}: {v}")

@click.argument("keys", nargs=-1)
@data.command()
def remove(keys):
    """
    Remove specified keys from the repository.
    Use the 'gc' command to then physically remove the data.
    """
    repo = DataRepository.default()
    for key in keys:
        data = repo.lookup(key)
        if data is None:
            rich.print(f"Data not found: {key}")
            return
    for key in keys:
        repo.deregister(key)
    rich.print(f"{" ".join(keys)}")
# Add under an alias
data.add_command(remove, name="rm")

@data.command("purge")
def purge_data():
    """Remove all data from the repository."""
    repo = DataRepository.default()
    for key in repo.keys():
        repo.deregister(key)
    repo.gc()
    rich.print("All data purged.")

@data.command("gc")
def garbage_collect():
    """Remove unused data from the repository."""
    repo = DataRepository.default()
    removed = repo.gc()
    rich.print("Garbage collection complete")
    for sha in removed:
        rich.print(f"  Removed: [blue]{sha}[/blue]")

@click.option("--port", default=8000)
@click.option("--host", default="127.0.0.1")
@click.option("--visualizer", default="nanoconfig.data.visualizer.DataVisualizer()")
@click.argument("data")
@data.command()
def visualize(data, visualizer, host, port):
    """Will start a marimo notebook to visualize the data."""
    repo = DataRepository.default()
    data = repo.lookup(data)
    if data is None:
        rich.print(f"Data not found: {data}")
        return
    DataVisualizer.host_marimo_notebook(host, port, visualizer, data)

class CustomLogRender(rich._log_render.LogRender): # type: ignore
    def __call__(self, *args, **kwargs):
        output = super().__call__(*args, **kwargs)
        if not self.show_path:
            output.expand = False
        return output

FORMAT = "%(name)s - %(message)s"

def setup_logging(show_path=False):
    # add_log_level("TRACE", logging.DEBUG - 5)
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
        level=logging.INFO,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[handler]
    )

def _parse_ctor(t: str):
    t = t.strip()
    idx = t.find("(")
    if idx >= 0:
        base, args = t[:idx], t[idx+1:-1]
        args = json.loads(f"[{args}]")
    else:
        base, args = t, None
    idx = base.rfind(".")
    if idx >= 0:
        module = base[:idx]
        name = base[idx+1:]
    else:
        module = "nanoconfig.data.transform"
        name = base
    transform = getattr(importlib.import_module(module), name)
    if args is None:
        return transform
    else:
        return transform(*args)

def _parse_constructors(ctors: list[str]):
    return [_parse_ctor(t) for t in ctors]
