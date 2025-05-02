from dataclasses import dataclass
from fsspec.asyn import AbstractFileSystem
from torch.utils.data import Dataset as TorchDataset
from pathlib import PurePath

import json
import contextlib
import pyarrow.dataset as ds
import pyarrow as pa
import abc
import io
import typing as ty

Metadata : ty.TypeAlias = (
    dict[str, "Metadata"] | list["Metadata"]
    | float | int | str | bool
)

@dataclass(frozen=True)
class SplitInfo:
    name: str
    size: int # size in number of samples
    content_size: int # size in bytes
    schema: pa.Schema

    @property
    def mime_type(self) -> str:
        return self.schema.metadata.get(b"mime_type", b"unknown").decode("utf-8")

T = ty.TypeVar("T", covariant=True)
class DataAdapter(ty.Protocol[T]):
    def __call__(self, dataset: ds.Dataset) -> T:
        ...

class Data(abc.ABC):
    @abc.abstractmethod
    def split_infos(self) -> dict[str, SplitInfo]:
        pass

    @abc.abstractmethod
    def split_info(self, name: str) -> SplitInfo:
        pass

    @ty.overload
    def split(self, name: str, adapter: DataAdapter[T]) -> T | None: ...
    @ty.overload
    def split(self, name: str) -> ds.Dataset | None: ...
    @abc.abstractmethod
    def split(self, name: str, adapter: "DataAdapter[T] | None" = None) -> T | ds.Dataset: ...

    @ty.overload
    def splits(self, adapter: DataAdapter[T]) -> ty.Iterator[tuple[str, T]]: ...
    @ty.overload
    def splits(self) -> ty.Iterator[tuple[str, ds.Dataset]]: ...
    @abc.abstractmethod
    def splits(self, adapter: DataAdapter[T] | None = None) -> ty.Iterator[tuple[str,T | ds.Dataset]]: ...

    # Open an auxiliary file
    # e.g. tokenizer information, etc.
    @property
    @abc.abstractmethod
    def aux(self) -> AbstractFileSystem: ...

    @property
    @abc.abstractmethod
    def sha256(self) -> str:
        pass
