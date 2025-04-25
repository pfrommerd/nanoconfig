import abc
from fsspec.asyn import AbstractFileSystem
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

logger = logging.getLogger(__name__)

class DataSource(abc.ABC):
    @abc.abstractmethod
    def prepare(self, repo: "DataRepository | None" = None) -> Data:
        pass

    @property
    @abc.abstractmethod
    def sha256(self) -> str:
        pass

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
        for split in data.split_infos().values():
            ds = data.split(split.name)
            for batch in ds.to_batches():
                pass

class DataRepository(abc.ABC):
    @abc.abstractmethod
    def keys(self) -> ty.Iterable[str]:
        pass

    @abc.abstractmethod
    def register(self, alias: str, sha: str):
        pass

    @abc.abstractmethod
    def deregister(self, alias: str):
        pass

    @abc.abstractmethod
    def lookup(self, alias_or_sha: str) -> Data | None:
        pass

    @abc.abstractmethod
    @contextlib.contextmanager
    def initialize(self, data_sha: str) -> ty.Iterator[DataWriter]:
        pass

    def fetch(self, source: str | DataSource) -> Data:
        if isinstance(source, str):
            data = self.lookup(source)
        else:
            data = self.lookup(source.sha256)
        if data is not None:
            return data
        if isinstance(source, str):
            raise ValueError(f"Data not found for alias '{source}'")
        data = source.prepare(self)
        assert data.sha256 == source.sha256
        if self.lookup(data.sha256) is None:
            with self.initialize(data.sha256) as writer:
                writer.write(data)
            data = self.lookup(data.sha256)
            assert data is not None
        return data

    @staticmethod
    def default() -> "DataRepository":
        from .fs import FsDataRepository
        return FsDataRepository()

class DataTransform(abc.ABC):
    @property
    @abc.abstractmethod
    def sha256(self) -> str:
        pass

    @abc.abstractmethod
    def transform(self, data: Data, repo: DataRepository | None = None) -> Data:
        pass

# A data pipeline is a source and a sequence of transformations
class DataPipeline(DataSource):
    def __init__(self, source: DataSource,
                 *transformations: DataTransform):
        self.source = source
        self.transformations : ty.Sequence[DataTransform] = transformations

    @staticmethod
    def compose(source: DataSource, *transformations: DataTransform):
        return DataPipeline(source, *transformations)

    @ty.override
    def prepare(self, repo: DataRepository | None = None) -> Data:
        data = self.source.prepare(repo)
        for t in self.transformations:
            data = t.transform(data, repo)
        return data

    @property
    @ty.override
    def sha256(self) -> str:
        combined = (
            self.source.sha256 + "-" +
            "-".join(t.sha256 for t in self.transformations)
        )
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()
