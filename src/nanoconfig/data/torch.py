import abc
import typing as ty
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import torch
import itertools
import torch.utils._pytree as pytree

from torch.utils.data import Dataset, DataLoader, IterableDataset
from . import DataAdapter

T = ty.TypeVar("T")

Converter: ty.TypeAlias = ty.Callable[[ds.Dataset], ty.Iterator[T]]

class SizedDataset(Dataset[T], ty.Generic[T], ty.Sized):
    @abc.abstractmethod
    def loader(self, batch_size: int, *,
                shuffle: bool = False) -> DataLoader[T]:
        ...

    @abc.abstractmethod
    def head(self, n: int) -> T:
        ...

    @property
    @abc.abstractmethod
    def data_sample(self) -> T:
        ...

class StreamingDataset(IterableDataset[T], SizedDataset[T], ty.Generic[T]):
    def __init__(self, data: ds.Dataset, converter: Converter[T], *,
                    batch_size: int | None = None, shuffle: bool = False,
                    _data_sample: T | None = None):
        if _data_sample is None:
            first_batch = next(converter(data))
            _data_sample = pytree.tree_map(lambda x: x[0], first_batch)
        assert _data_sample is not None
        # put back on the iterator
        self._data = data
        self._converter = converter
        self._length = data.count_rows()
        self._data_sample = _data_sample
        self._batch_size = batch_size
        self._shuffle = shuffle

    def head(self, n: int) -> T:
        raise NotImplementedError

    @property
    def data_sample(self) -> T:
        return self._data_sample

    def loader(self, batch_size: int, *, shuffle: bool = False) -> DataLoader[T]:
        return DataLoader(StreamingDataset(self._data, self._converter,
            batch_size=batch_size, shuffle=shuffle,
            _data_sample=self._data_sample
        ))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        raise NotImplementedError

    def __len__(self) -> int:
        return self._length

class InMemoryDataset(SizedDataset[T], ty.Generic[T]):
    def __init__(self, data: T, *, _length: int | None = None):
        self._data = data
        self._data_sample = pytree.tree_map(lambda x: x[0], self._data)
        self._length = pytree.tree_leaves(self._data)[0].shape[0]

    def loader(self, batch_size: int, *, shuffle: bool = True) -> DataLoader[T]:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=
            lambda x: pytree.tree_map(lambda *xs: torch.stack(xs), x))

    def head(self, n: int) -> T:
        return pytree.tree_map(lambda x: x[:n], self._data)

    @property
    def data_sample(self) -> T:
        return self._data_sample

    def __getitem__(self, index: int) -> T:
        sample = pytree.tree_map(lambda x: x[index], self._data)
        print(sample)
        return sample

    def __len__(self) -> int:
        return self._length

    def to(self, device: torch.device) -> "InMemoryDataset[T]":
        data : T = pytree.tree_map(lambda x: x.to(device), self._data)
        return InMemoryDataset(None, None, _data=data, _length=self._length) # type: ignore

class TorchAdapter(DataAdapter[SizedDataset[T]], ty.Generic[T]):
    def __init__(self, force_stream: bool = False):
        self._adapters = {}
        self._force_stream = force_stream

    def register_type(self, mime_type: str,
            convert: Converter[T]):
        self._adapters[mime_type] = convert

    def convert(self, data: ds.Dataset) -> ty.Iterator[T]:
        mime_type = data.schema.metadata.get(b"mime_type", "unknown").decode()
        if mime_type not in self._adapters:
            raise ValueError(f"Unsupported mime type: {mime_type}")
        converter = self._adapters[mime_type]
        return converter(data)

    def __call__(self, data: ds.Dataset) -> SizedDataset[T]:
        # compute the total size of the dataset
        def _size(f):
            if f.filesystem is not None and f.path is not None:
                return f.filesystem.get_file_info(f.path).size
            elif f.buffer is not None:
                return len(f.buffer)
            else:
                raise ValueError(f"Unable to determine size of fragment {f}")
        total_size = data.count_rows()
        # For small datasets, use in memory
        if total_size < 128*1024 and not self._force_stream:
            batches = list(self.convert(data))
            batches = pytree.tree_map(
                lambda *xs: (torch.concatenate(xs)
                    if isinstance(xs[0], torch.Tensor) else xs[0]), *batches
            )
            return InMemoryDataset(batches)
        else:
            return StreamingDataset(data, self.convert)


def as_torch(array: pa.FixedSizeListArray, device: torch.device | str = "cpu") -> torch.Tensor:
    if isinstance(device, str):
        device = torch.device(device)
    shape = []
    type = array.type
    while type.is_list():
        if type.list_size <= 0:
            raise ValueError("Invalid list size, can only use fixed-length lists.")
        shape.append(type.list_size)
        array = array.flatten()
        type = type.value_type
    array = array.to_numpy(zero_copy_only=False).reshape(-1, *shape)
    return torch.tensor(array, device=device)
