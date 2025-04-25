from huggingface_hub.hf_api import R
from pandas.core.indexes.datetimes import dt
from . import Data, SplitInfo, DataAdapter
from .source import DataSource, DataWriter, SplitWriter, DataRepository
from pathlib import PurePath, Path

from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.dirfs import DirFileSystem

import json
import itertools
import typing as ty
import contextlib
import pyarrow as pa
import pyarrow.parquet as pq

T = ty.TypeVar("T")

class FsData(Data):
    def __init__(self, fs: AbstractFileSystem, sha: str,
                split_fragments: dict[str, list[str]] | None = None,
                # To override the underlying schema metadata
                metadata: dict[str, ty.Any] | None = None):
        self._fs = fs
        self._sha = sha
        self._metadata = metadata

        if split_fragments is None:
            split_fragments = {}
            data_fragment = None
            for fragment in self._fs.glob("**/*.parquet"):
                fragment = PurePath(fragment) # type: ignore
                if fragment.parts[0] == "aux":
                    continue
                data_fragment = fragment
                split_fragments.setdefault(fragment.parts[0], []).append(str(fragment))
        self._split_fragments = split_fragments

    @ty.override
    def split_infos(self) -> dict[str, SplitInfo]:
        splits = {}
        for split in self._split_fragments.keys():
            splits[split] = self.split_info(split)
        return splits

    @ty.override
    def split_info(self, name: str) -> SplitInfo:
        if not name in self._split_fragments:
            raise ValueError(f"Split {name} does not exist")
        fragments = self._split_fragments[name]
        # open the dataset to find the schema
        ds = pq.ParquetDataset(fragments, filesystem=self._fs)
        schema = ds.schema
        if self._metadata is not None:
            schema = schema.with_metadata({
                k.encode("utf-8"): v.encode("utf-8") for k, v in self._metadata.items()
            })
        return SplitInfo(
            name=name,
            size=sum(f.count_rows() for f in ds.fragments),
            content_size=sum(f.count_rows() for f in ds.fragments),
            schema=schema
        )

    def split(self, name: str, adapters: DataAdapter[T] | None = None):
        if not name in self._split_fragments:
            raise ValueError(f"Split {name} does not exist")
        fragments = self._split_fragments[name]
        ds = pq.ParquetDataset(fragments, filesystem=self._fs)
        if self._metadata is not None:
            schema = ds.schema.with_metadata(self._metadata)
            ds = pq.ParquetDataset(fragments, schema=schema, filesystem=self._fs)
        if adapters:
            return adapters(ds)
        return ds

    @property
    def aux(self) -> AbstractFileSystem:
        return DirFileSystem(PurePath("aux"), self._fs)

    @property
    def sha256(self) -> str:
        return self._sha

DEFAULT_LOCAL_DATA = Path.home() / ".cache" / "nanodata"

# 128 Mb per file
MAX_FILE_SIZE = 128 * 1024 * 1024 * 1024

class LocalSplitWriter(SplitWriter):
    def __init__(self, fs: AbstractFileSystem, name: str):
        self._fs = fs
        self._name = name

        self._num_parts = 0
        self._schema = None
        self._current_file = None
        self._current_writer = None

    def _check_writer(self):
        """Check if the current shard needs to be closed and a new one created."""
        if self._current_file and self._current_file.size() > MAX_FILE_SIZE:
            assert self._current_writer is not None
            self._current_writer.close()
            self._current_file.close()
            self._current_file = None
            self._current_writer = None
        if not self._current_file or not self._current_writer:
            self._num_parts += 1
            self._current_file = pa.PythonFile(
                self._fs.open(PurePath(self._name) / f"part-{self._num_parts:05d}.parquet", 'wb')
            )
            self._current_writer = pq.ParquetWriter(
                self._current_file, self._schema
            )

    def write_batch(self, batch: pa.RecordBatch):
        if not self._schema:
            self._schema = batch.schema
        else:
            assert self._schema == batch.schema
        self._check_writer()
        assert self._current_writer is not None
        self._current_writer.write_batch(batch)

    def close(self):
        if self._current_writer:
            self._current_writer.close()
        if self._current_file:
            self._current_file.close()

class FsDataWriter(DataWriter):
    def __init__(self, fs: AbstractFileSystem):
        self._fs = fs

    @ty.override
    @contextlib.contextmanager
    def split(self, name: str) -> ty.Iterator[LocalSplitWriter]:
        if self._fs.exists(name):
            raise FileExistsError(f"Split '{name}' already exists")
        self._fs.mkdir(name)
        writer = LocalSplitWriter(self._fs, name)
        yield writer
        writer.close()

    @ty.override
    @contextlib.contextmanager
    def aux(self) -> ty.Iterator[AbstractFileSystem]:
        if not self._fs.exists("aux"):
            self._fs.mkdir("aux")
        yield DirFileSystem("aux", self._fs)

class FsDataRepository(DataRepository):
    def __init__(self, fs: AbstractFileSystem | Path | str | None = None):
        if fs is None:
            fs = DirFileSystem(DEFAULT_LOCAL_DATA, LocalFileSystem())
        elif isinstance(fs, (str, Path)):
            fs = DirFileSystem(fs, LocalFileSystem())
        self._fs = fs
        self._aliases = {}
        if self._fs.exists('registry.json'):
            with self._fs.open("registry.json") as f:
                self._aliases = json.load(f)

    def keys(self) -> ty.Iterable[str]:
        return self._aliases.keys()

    def register(self, alias: str, sha: str):
        if alias in self._aliases:
            raise ValueError(f"Alias '{alias}' already exists")
        if not self._fs.exists(sha):
            raise FileNotFoundError(f"Data '{sha}' not found")
        # Register the alias
        self._aliases[alias] = sha
        with self._fs.open("registry.json", "w") as f:
            json.dump(self._aliases, f)

    def deregister(self, alias: str):
        if alias not in self._aliases:
            return
        del self._aliases[alias]
        with self._fs.open("registry.json", "w") as f:
            json.dump(self._aliases, f)

    def lookup(self, alias_or_sha: str) -> Data:
        sha = self._aliases.get(alias_or_sha, alias_or_sha)
        if not self._fs.exists(alias_or_sha):
            raise FileNotFoundError(f"Data '{alias_or_sha}' not found")
        return FsData(DirFileSystem(PurePath(sha), self._fs), sha)

    @contextlib.contextmanager
    def initialize(self, data_sha: str) -> ty.Iterator[FsDataWriter]:
        if self._fs.exists(data_sha):
            raise FileExistsError(f"Data '{data_sha}' already exists")
        self._fs.mkdir(data_sha)
        yield FsDataWriter(DirFileSystem(PurePath(data_sha), self._fs))
