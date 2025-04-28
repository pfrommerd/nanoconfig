from matplotlib.pyplot import twiny
from nanoconfig.data.huggingface import HfDataSource
from nanoconfig.data.torch import InMemoryDataset, TorchAdapter

from fsspec.implementations.memory import MemoryFileSystem

import pyarrow as pa
import PIL.Image
import io

import pyarrow.dataset as ds
import pyarrow.parquet as pq

import torch
import torchvision.transforms.functional

def test_hf_data_source():
    # Will load the mnist dataset
    mnist_data = HfDataSource.from_repo("ylecun/mnist").prepare()
    actual_schema = mnist_data.split_info("train").schema
    expected_schema = pa.schema([
        pa.field("image", pa.struct([
            pa.field("bytes", pa.binary()),
            pa.field("path", pa.string())
        ])),
        pa.field("label", pa.int64())
    ])
    assert actual_schema == expected_schema
    assert actual_schema.metadata[b"mime_type"] == b"parquet/image+label"

    tinystories_data = HfDataSource.from_repo("roneneldan/TinyStories").prepare()
    actual_schema = tinystories_data.split_info("train").schema
    expected_schema = pa.schema([
        pa.field("text", pa.string()),
    ])
    assert actual_schema == expected_schema
    assert actual_schema.metadata[b"mime_type"] == b"parquet/text"

def test_torch_data():
    adapter = TorchAdapter()

    def read_image(pa_bytes):
        img = PIL.Image.open(io.BytesIO(pa_bytes.as_py()))
        return torchvision.transforms.functional.pil_to_tensor(img)
    def convert_image(batch):
        image_bytes = batch["image"].field("bytes")
        labels = torch.tensor(batch["label"].to_numpy())
        images = torch.stack([
            read_image(b) for b in image_bytes
        ])
        return labels, images
    adapter.register_type("parquet/image+label", convert_image)
    mnist_data = HfDataSource.from_repo("ylecun/mnist").prepare()
    train_data = mnist_data.split("train", adapter)
    assert train_data is not None
    assert len(train_data) == 60000

def test_memory_data_loader():
    data = pa.Table.from_pylist([
        {"data": 2},
        {"data": 4},
        {"data": 5},
        {"data": 100}
    ])
    # fs = MemoryFileSystem()
    # pq.write_to_dataset(data, root_path="data", filesystem=fs)
    # data = pq.ParquetDataset("/data/", filesystem=fs)
    dataset = InMemoryDataset(
        data, lambda x: torch.tensor(
            x["data"].to_numpy(zero_copy_only=False)
        )
    )
    assert dataset[0] == 2
    assert dataset[1] == 4
    assert dataset[2] == 5
    assert dataset[3] == 100
    assert len(dataset) == 4
