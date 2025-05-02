from attr import s
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
import pyarrow.dataset as ds
import pyarrow.parquet as pq

T = ty.TypeVar("T")

class FsData(Data):
    def __init__(self, fs: AbstractFileSystem, sha: str,
                    split_fragments: dict[str, list[str]] | None = None):
        self._fs = fs
        self._sha = sha
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
        return SplitInfo(
            name=name,
            size=sum(f.count_rows() for f in ds.fragments),
            content_size=sum(f.count_rows() for f in ds.fragments),
            schema=ds.schema
        )

    def split(self, name: str, adapter: DataAdapter[T] | None = None) -> ds.Dataset | T | None:
        if not name in self._split_fragments:
            return None
        fragments = self._split_fragments[name]
        ds = pq.ParquetDataset(fragments, filesystem=self._fs)._dataset
        if adapter:
            return adapter(ds)
        return ds

    def splits(self, adapter: DataAdapter[T] | None = None) -> ty.Iterator[tuple[str, ds.Dataset | T]]:
        for name in self._split_fragments:
            yield name, self.split(name, adapter)

    @property
    def aux(self) -> AbstractFileSystem:
        return DirFileSystem(PurePath("aux"), self._fs)

    @property
    def sha256(self) -> str:
        return self._sha

DEFAULT_LOCAL_DATA = Path.home() / ".cache" / "nanodata"

# 128 Mb per file
MAX_FILE_SIZE = 128 * 1024 * 1024 * 1024

class FsSplitWriter(SplitWriter):
    def __init__(self, fs: AbstractFileSystem, name: str):
        self._fs = fs
        self._name = name

        self._num_parts = 0
        self._schema = None
        self._current_file = None
        self._current_writer = None
        self._current_size = 0

    def _check_writer(self):
        """Check if the current shard needs to be closed and a new one created."""
        if self._current_file and self._current_size > MAX_FILE_SIZE:
            assert self._current_writer is not None
            self._current_writer.close()
            self._current_file.close()
            self._current_file = None
            self._current_writer = None
        if not self._current_file or not self._current_writer:
            self._num_parts += 1
            self._current_file = pa.PythonFile(
                self._fs.open(str(PurePath(self._name) / f"{self._name}-{self._num_parts:05d}.parquet"), 'wb')
            )
            self._current_writer = pq.ParquetWriter(
                self._current_file, self._schema
            )
            self._current_size = 0

    def write_batch(self, batch: pa.RecordBatch):
        if not self._schema:
            self._schema = batch.schema
            if not self._schema or not b'mime_type' in self._schema.metadata:
                raise ValueError("Schema must have a mime_type metadata")
            for field in self._schema.names:
                field = self._schema.field(field)
                if not field.metadata or not b'mime_type' in field.metadata:
                    raise ValueError(f"Field {field} must have a mime_type metadata")
        else:
            if not self._schema == batch.schema:
                raise ValueError(f"Schema mismatch: {self._schema} != {batch.schema}")
            # Just in case the metadata is different, cast the batch to a uniform schema
            batch = batch.cast(self._schema)
        self._check_writer()
        assert self._current_writer is not None
        self._current_writer.write_batch(batch)
        self._current_size += batch.nbytes

    def close(self):
        if self._current_writer:
            self._current_writer.close()
        if self._current_file:
            self._current_file.close()

class FsDataWriter(DataWriter):
    def __init__(self, fs: AbstractFileSystem | Path | str, sha: str):
        if isinstance(fs, (str, Path)):
            fs = DirFileSystem(str(fs), LocalFileSystem())
        self._fs = fs
        self._sha = sha

    @ty.override
    @contextlib.contextmanager
    def split(self, name: str) -> ty.Iterator[FsSplitWriter]:
        if self._fs.exists(name):
            raise FileExistsError(f"Split '{name}' already exists")
        self._fs.mkdir(name)
        writer = FsSplitWriter(self._fs, name)
        yield writer
        writer.close()

    @ty.override
    @contextlib.contextmanager
    def aux(self) -> ty.Iterator[AbstractFileSystem]:
        if not self._fs.exists("aux"):
            self._fs.mkdir("aux")
        yield DirFileSystem("aux", self._fs)

    @ty.override
    def close(self) -> Data:
        return FsData(self._fs, self._sha)

class FsDataRepository(DataRepository):
    def __init__(self, fs: AbstractFileSystem | Path | str | None = None):
        if fs is None:
            fs = DirFileSystem(DEFAULT_LOCAL_DATA, LocalFileSystem())
        elif isinstance(fs, (str, Path)):
            fs = DirFileSystem(fs, LocalFileSystem())
        self.fs = fs
        self._aliases = {}
        if self.fs.exists('registry.json'):
            with self.fs.open("registry.json") as f:
                self._aliases = json.load(f)

    def keys(self) -> ty.Iterable[str]:
        return list(self._aliases.keys())

    def register(self, alias: str, sha: str | Data | DataSource):
        if not isinstance(sha, str):
            sha = sha.sha256
        if not self.fs.exists(sha):
            raise FileNotFoundError(f"Data '{sha}' not found")
        # Register the alias
        self._aliases[alias] = sha
        with self.fs.open("registry.json", "w") as f:
            json.dump(self._aliases, f)

    def deregister(self, alias: str):
        if alias not in self._aliases:
            return
        del self._aliases[alias]
        with self.fs.open("registry.json", "w") as f:
            json.dump(self._aliases, f)

    @ty.override
    def gc(self) -> set[str]:
        removed = set()
        shas = set(self._aliases.values())
        for sha in self.fs.listdir("/", detail=False):
            if sha not in shas and sha != "registry.json":
                removed.add(sha)
                self.fs.rm(sha, recursive=True)
        return removed

    @ty.override
    def lookup(self, alias_or_sha: str | Data | DataSource) -> Data | None:
        if not isinstance(alias_or_sha, str):
            sha = alias_or_sha.sha256
        else:
            sha = self._aliases.get(alias_or_sha, alias_or_sha)
        if not self.fs.exists(sha):
            return None
        return FsData(DirFileSystem(PurePath(sha), self.fs), sha)

    @ty.override
    @contextlib.contextmanager
    def init(self, data_sha: str) -> ty.Iterator[FsDataWriter]:
        if self.fs.exists(data_sha):
            raise FileExistsError(f"Data '{data_sha}' already exists")
        self.fs.mkdir(data_sha)
        yield FsDataWriter(DirFileSystem(data_sha, self.fs), data_sha)

class FsDataSource(DataSource):
    def __init__(self, fs: AbstractFileSystem, sha: str):
        self.fs = fs
        self._sha = sha

    def prepare(self, repo: "DataRepository | None" = None) -> Data:
        return FsData(self.fs, self._sha)

    @property
    def sha256(self) -> str:
        return self._sha

    @staticmethod
    def from_path(path: str | Path, sha: str):
        fs = DirFileSystem(str(path), LocalFileSystem())
        return FsDataSource(fs, sha)
