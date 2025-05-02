import abc

from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.dirfs import DirFileSystem

import huggingface_hub as hf
import hashlib
import logging
import re
import fsspec
import requests
import contextlib
import typing as ty
import pyarrow as pa
import tempfile
import contextlib
import shutil

from dataclasses import dataclass
from pathlib import Path
from rich.progress import Progress

from . import Data

import urllib.parse

logger = logging.getLogger(__name__)

class DataSource(abc.ABC):
    @abc.abstractmethod
    def prepare(self, repo: "DataRepository | None" = None) -> Data:
        pass

    @property
    @abc.abstractmethod
    def sha256(self) -> str:
        pass

    @staticmethod
    def from_url(url: str) -> "DataSource":
        parsed = urllib.parse.urlparse(url)
        query_params = urllib.parse.parse_qs(parsed.query)
        if parsed.scheme == "file":
            from .fs import FsDataSource
            path = parsed.netloc + parsed.path
            sha = query_params.get("sha", [None])[0]
            if sha is None:
                raise ValueError("Missing sha parameter to file data url.")
            return FsDataSource.from_path(path, sha)
        elif parsed.scheme == "hf":
            from .huggingface import HfDataSource
            repo = parsed.netloc + parsed.path
            repo_parts = repo.split("/")
            if len(repo_parts) < 1:
                raise ValueError(f"Invalid repository format: {repo}")
            repo = repo_parts[0] + "/" + repo_parts[1]
            subset = "/".join(repo_parts[2:]) if len(repo_parts) > 2 else None
            rev = parsed.fragment if parsed.fragment else None
            return HfDataSource.from_repo(repo, subset, rev)
        elif parsed.scheme == "gen":
            from .generator import GeneratorSource
            exec = parsed.netloc + parsed.path
            args = [f"--{k}={v}" for k,v in query_params.items()]
            return GeneratorSource.from_command(exec, *args)
        else:
            raise ValueError(f"Unsupported scheme: {parsed.scheme}")

class SplitWriter(abc.ABC):
    @abc.abstractmethod
    def write_batch(self, batch: pa.RecordBatch) -> None: ...

class DataWriter(abc.ABC):
    @abc.abstractmethod
    @contextlib.contextmanager
    def split(self, name: str) -> ty.Iterator[SplitWriter]:
        pass

    @abc.abstractmethod
    @contextlib.contextmanager
    def aux(self) -> ty.Iterator[AbstractFileSystem]:
        pass

    # Will copy all data over. Can be overridden
    # by subclasses to implement more efficient copying.
    def write(self, data: Data):
        for name, split in data.splits():
            with self.split(name) as split_writer:
                for batch in split.to_batches():
                    split_writer.write_batch(batch)

    @abc.abstractmethod
    def close(self) -> Data:
        pass

class DataRepository(abc.ABC):
    @abc.abstractmethod
    def keys(self) -> ty.Iterable[str]:
        pass

    @abc.abstractmethod
    def register(self, alias: str, sha: str | Data | DataSource):
        pass

    @abc.abstractmethod
    def deregister(self, alias: str):
        pass

    @abc.abstractmethod
    def gc(self) -> set[str]:
        pass

    @abc.abstractmethod
    def lookup(self, alias_or_sha: str | Data | DataSource) -> Data | None:
        pass

    @abc.abstractmethod
    @contextlib.contextmanager
    def init(self, data_sha: str) -> ty.Iterator[DataWriter]:
        pass

    def get(self, source: str | Data | DataSource) -> Data:
        if isinstance(source, str):
            data = self.lookup(source)
        else:
            data = self.lookup(source.sha256)
        if data is not None:
            return data
        if isinstance(source, str):
            raise ValueError(f"Data not found for alias '{source}'")
        elif isinstance(source, DataSource):
            data = source.prepare(self)
            assert data.sha256 == source.sha256
        else:
            data = source
        if self.lookup(data.sha256) is None:
            with self.init(data.sha256) as writer:
                writer.write(data)
            assert data is not None
        data = self.lookup(data.sha256)
        assert data is not None
        return data

    @staticmethod
    def default() -> "DataRepository":
        from .fs import FsDataRepository
        return FsDataRepository()
