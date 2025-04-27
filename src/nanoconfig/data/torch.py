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

Converter: ty.TypeAlias = ty.Callable[[pa.RecordBatch], T]

class SizedDataset(Dataset[T], ty.Generic[T], ty.Sized):
    @abc.abstractmethod
    def loader(self, batch_size: int, *,
                shuffle: bool = False) -> DataLoader[T]:
        ...

    @property
    @abc.abstractmethod
    def data_sample(self) -> T:
        ...

class StreamingDataset(IterableDataset[T], SizedDataset[T], ty.Generic[T]):
    def __init__(self, data: ds.Dataset, converter: Converter[T], *,
                    batch_size: int | None = None, shuffle: bool = False,
                    _data_sample: T | None = None):
        self._data = data
        self._converter = converter
        self._length = sum(
            f.count_rows() for f in data.fragments
        )
        if _data_sample is None:
            _data_sample = data.fragments[0].head(1).to_batches()[0]
            # convert the "batch" to desired format
            _data_sample = converter(_data_sample)
            _data_sample = pytree.tree_map(lambda x: x[0], _data_sample)
            if _data_sample is None:
                raise ValueError("Dataset is empty")
        self._data_sample = _data_sample
        self._batch_size = batch_size
        self._shuffle = shuffle

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

        for sample in itertools.islice(self._data, worker_id, None, num_workers):
            yield self._converter(sample)

    def __len__(self) -> int:
        return self._length

class InMemoryDataset(SizedDataset[T], ty.Generic[T]):
    def __init__(self, data: pa.Table, converter: Converter[T], *,
                    _length: int | None = None, _data: T | None = None):
        converted = [
            converter(batch) for batch in data.to_batches()
        ]
        self._data = pytree.tree_map(
            lambda *xs: (torch.concatenate(xs)
                if isinstance(xs[0], torch.Tensor) else xs[0]), *converted
        ) if _data is None else _data

        self._data_sample = pytree.tree_map(lambda x: x[0], self._data)
        self._length = pytree.tree_leaves(self._data)[0].shape[0]
        self._length = len(data) if _length is None else _length

    def loader(self, batch_size: int, *, shuffle: bool = True) -> DataLoader[T]:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    @property
    def data_sample(self) -> T:
        return self._data_sample

    def __getitem__(self, index: int) -> T:
        return pytree.tree_map(lambda x: x[index], self._data)

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
            convert: ty.Callable[[pa.RecordBatch], T]):
        self._adapters[mime_type] = convert

    def __call__(self, data: ds.Dataset):
        mime_type = data.schema.metadata.get(b"mime_type", "unknown").decode()
        if mime_type not in self._adapters:
            raise ValueError(f"Unsupported mime type: {mime_type}")
        converter = self._adapters[mime_type]

        # compute the total size of the dataset
        def _size(f):
            if f.filesystem is not None and f.path is not None:
                return f.filesystem.get_file_info(f.path).size
            elif f.buffer is not None:
                return len(f.buffer)
            else:
                raise ValueError(f"Unable to determine size of fragment {f}")
        total_size = sum(_size(f) for f in data.fragments)
        # If the size is less than 2GB, use in memory
        if total_size < 2*1024*1024*1024 and not self._force_stream:
            return InMemoryDataset(data.read(), converter)
        else:
            return StreamingDataset(data, converter)
