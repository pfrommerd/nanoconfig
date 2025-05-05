import abc
import typing as ty
from pandas import compat
from plotly.graph_objects import Stream
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

    @abc.abstractmethod
    def limit(self, n: int) -> ty.Self:
        ...

    @property
    @abc.abstractmethod
    def data_sample(self) -> T:
        ...

class StreamingDataset(IterableDataset[T], SizedDataset[T], ty.Generic[T]):
    def __init__(self, data: ds.Dataset, converter: Converter[T], *,
                    batch_size: int | None = None, shuffle: bool = False,
                    _data_sample: T | None = None, _length: int | None = None):
        if _data_sample is None:
            first_batch = next(converter(data))
            _data_sample = pytree.tree_map(lambda x: x[0], first_batch)
        assert _data_sample is not None
        # put back on the iterator
        self._data = data
        self._converter = converter
        self._length = _length or data.count_rows()
        self._data_sample = _data_sample
        self._batch_size = batch_size
        self._shuffle = shuffle

    def head(self, n: int) -> T:
        raise NotImplementedError

    def limit(self, n: int) -> "StreamingDataset[T]":
        return StreamingDataset(self._data, self._converter,
            _data_sample=self._data_sample, _length=min(n, self._length)
        )

    @property
    def data_sample(self) -> T:
        return self._data_sample

    def loader(self, batch_size: int, *, shuffle: bool = False) -> DataLoader[T]:
        return DataLoader(StreamingDataset(self._data, self._converter,
            batch_size=batch_size, shuffle=shuffle,
            _data_sample=self._data_sample, _length=self._length
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
        self._data_sample = pytree.tree_map(lambda x: x[0] if isinstance(x, torch.Tensor) else x, self._data)
        self._length = [
            x for x in pytree.tree_leaves(self._data)
            if isinstance(x, torch.Tensor)
        ][0].shape[0]

    def loader(self, batch_size: int, *, shuffle: bool = True) -> DataLoader[T]:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=
            lambda x: pytree.tree_map(lambda *xs: torch.stack(xs)
                            if isinstance(xs[0], torch.Tensor) else xs[0], *x)
        )

    def head(self, n: int) -> T:
        return pytree.tree_map(lambda x: x[:n], self._data)

    def limit(self, n: int) -> "InMemoryDataset[T]":
        return InMemoryDataset(
            pytree.tree_map(lambda x: x[:n] if isinstance(x, torch.Tensor) else x, self._data),
            _length=min(n, self._length)
        )

    @property
    def data_sample(self) -> T:
        return self._data_sample

    def __getitem__(self, index: int) -> T:
        return pytree.tree_map(
            lambda x: x[index] if isinstance(x, torch.Tensor) else x,
            self._data)

    def __len__(self) -> int:
        return self._length

    def to(self, device: torch.device) -> "InMemoryDataset[T]":
        data : T = pytree.tree_map(lambda x: x.to(device), self._data)
        return InMemoryDataset(None, None, _data=data, _length=self._length) # type: ignore

class TorchAdapter(DataAdapter[SizedDataset[T]], ty.Generic[T]):
    def __init__(self, force_stream: bool = False, override_mime_type: str | None = None,
                        shuffle_load_order: bool = True, load_order_seed: int = 42):
        self._adapters = {}
        self._force_stream = force_stream
        self._force_mime_type = override_mime_type
        self._shuffle_load_order = shuffle_load_order
        self._load_order_seed = load_order_seed

    def register_type(self, mime_type: str,
            convert: Converter[T]):
        self._adapters[mime_type] = convert

    def convert(self, data: ds.Dataset) -> ty.Iterator[T]:
        mime_type = data.schema.metadata.get(b"mime_type", "unknown").decode()
        if self._force_mime_type is not None:
            mime_type = self._force_mime_type
            metadata = data.schema.metadata if data.schema.metadata else {}
            metadata[b"mime_type"] = mime_type.encode()
            data = data.replace_schema(data.schema.with_metadata(metadata))
        if mime_type not in self._adapters:
            # get the longest prefix match
            compatible = list((-len(k), v) for k, v in self._adapters.items() if mime_type.startswith(k))
            compatible.sort(key=lambda x: x[0])
            if not compatible:
                raise ValueError(f"Unsupported mime type: {mime_type}")
            converter = compatible[0][1]
        else:
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
            L = [x for x in pytree.tree_leaves(batches) if isinstance(x, torch.Tensor)][0].shape[0]
            # Shuffle the dataset when loaded in, so that e.g. limit()
            # will perduce a random dataset that is not disk-order dependent
            if self._shuffle_load_order:
                gen = torch.Generator()
                gen.manual_seed(self._load_order_seed)
                perm = torch.randperm(L, generator=gen)
                batches = pytree.tree_map(lambda x: x[perm] if isinstance(x, torch.Tensor) else x, batches)
            return InMemoryDataset(batches)
        else:
            raise NotImplementedError
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
