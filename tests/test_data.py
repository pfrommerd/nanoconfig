from matplotlib.pyplot import twiny
from nanoconfig.data.huggingface import HfDataSource

import pyarrow as pa
import PIL.Image
import io

import torch
import torchvision.transforms.functional

def test_hf_data_source():
    # Will load the mnist dataset
    mnist_data = HfDataSource.from_repo("ylecun/mnist", mime_type="pq/image+label").prepare()
    actual_schema = mnist_data.split_info("train").schema
    expected_schema = pa.schema([
        pa.field("image", pa.struct([
            pa.field("bytes", pa.binary()),
            pa.field("path", pa.string())
        ])),
        pa.field("label", pa.int64())
    ])
    assert actual_schema == expected_schema

    tinystories_data = HfDataSource.from_repo("roneneldan/TinyStories", mime_type="pq/text").prepare()
    actual_schema = tinystories_data.split_info("train").schema
    expected_schema = pa.schema([
        pa.field("text", pa.string()),
    ])
    assert actual_schema == expected_schema

def test_torch_data():
    from nanoconfig.data.torch import TorchAdapter
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
    adapter.register_type("pq/image+label", convert_image)

    mnist_data = HfDataSource.from_repo("ylecun/mnist", mime_type="pq/image+label").prepare()
    train_data = mnist_data.split("train", adapter)
    assert len(train_data) == 60000
