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
        pa.field("class", pa.int64())
    ])
    assert actual_schema == expected_schema
    assert actual_schema.metadata[b"mime_type"] == b"data/image+class"

    # tinystories_data = HfDataSource.from_repo("roneneldan/TinyStories").prepare()
    # actual_schema = tinystories_data.split_info("train").schema
    # expected_schema = pa.schema([
    #     pa.field("text", pa.string()),
    # ])
    # assert actual_schema == expected_schema
    # assert actual_schema.metadata[b"mime_type"] == b"data/text"

def test_torch_data():
    adapter = TorchAdapter()

    def read_image(pa_bytes):
        img = PIL.Image.open(io.BytesIO(pa_bytes.as_py()))
        return torchvision.transforms.functional.pil_to_tensor(img)
    def convert_image(dataset):
        for batch in dataset.to_batches():
            image_bytes = batch["image"].field("bytes")
            labels = torch.tensor(batch["class"].to_numpy())
            images = torch.stack([
                read_image(b) for b in image_bytes
            ])
            yield images, labels
    adapter.register_type("data/image+class", convert_image)
    mnist_data = HfDataSource.from_repo("ylecun/mnist").prepare()
    train_data = mnist_data.split("train", adapter)
    assert train_data is not None
    assert len(train_data) == 60000

def test_memory_data_loader():
    # fs = MemoryFileSystem()
    # pq.write_to_dataset(data, root_path="data", filesystem=fs)
    # data = pq.ParquetDataset("/data/", filesystem=fs)
    dataset = InMemoryDataset(
        {"data": torch.tensor([2,4,5,100])}
    )
    assert dataset[0] == {"data": 2}
    assert dataset[1] == {"data": 4}
    assert dataset[2] == {"data": 5}
    assert dataset[3] == {"data": 100}
    assert len(dataset) == 4
