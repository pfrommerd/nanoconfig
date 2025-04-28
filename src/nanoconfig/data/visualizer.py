import pyarrow as pa
import pyarrow.dataset as ds
import typing as ty
import rich
import json
import math
import tempfile
import functools
import PIL.Image
import io

from . import Data
from .source import DataRepository
from ..experiment import NestedResult

def image_visualizer(data: pa.RecordBatch):
    labels = data.schema.metadata.get(b"classes", None)
    if labels:
        labels = json.loads(labels.decode())
    for row in data.to_pylist():
        data = {}
        image_data = row["image"]["bytes"]
        image = PIL.Image.open(io.BytesIO(image_data))
        data["image"] = image
        if "label" in row:
            data["label"] = row["label"] if labels is None else labels[row["label"]]
        yield data

def _slice_dataset(dataset: ds.Dataset, visualizer, start: int, stop: int):
    idx = 0
    for batch in dataset.to_batches():
        if idx >= stop:
            return
        if idx + len(batch) >= start:
            for row in visualizer(batch):
                if idx >= stop:
                    return
                if idx >= start:
                    yield row
                idx += 1
        else:
            idx += len(batch)
    return dataset.slice(start=0, stop=100)

class DataVisualizer:
    def __init__(self):
        self._visualizers = {}
        self._visualizers["parquet/image"] = image_visualizer
        self._visualizers["parquet/image+label"] = image_visualizer

    def add_visualizers(self, visualizer: "DataVisualizer"):
        for mime_type, func in visualizer._visualizers.items():
            self._visualizers[mime_type] = func

    def show(self, data: Data):
        import marimo as mo
        import pandas as pd
        splits = data.split_infos().values()
        def load_split(name):
            split = data.split(name)
            assert split is not None, f"Split '{name}' not found"
            visualizer = self._visualizers[split.schema.metadata.get(b"mime_type").decode()]
            rows = []
            for batch in split.to_batches():
                if len(rows) > 200:
                    break
                rows.extend(visualizer(batch))
            df = pd.DataFrame(rows)
            return mo.ui.dataframe(df, page_size=40)
        return mo.ui.tabs({
            split.name: mo.lazy(functools.partial(load_split, split.name)) for split in splits
        })
        # return self._visualizer(data)

    @staticmethod
    def host_marimo_notebook(host: str, port: int,
                visualizer_type: str | ty.Callable | ty.Type,
                data: Data):
        repo = DataRepository.default()
        data = repo.get(data)
        if not isinstance(visualizer_type, str):
            module = visualizer_type.__module__
            name = visualizer_type.__qualname__
        else:
            module, name = visualizer_type.split(":")
        try:
            import marimo
        except ImportError:
            rich.print("Marimo not installed")
            return
        import uvicorn
        from fastapi import FastAPI

        # Generate a marimo notebook
        with tempfile.TemporaryDirectory() as tempdir:
            with open(f"{tempdir}/visualizer.py", "w") as f:
                f.write(NOTEBOOK_TEMPLATE.format(
                    data_sha256=data.sha256, visualizer_module=module, visualizer_name=name
                ))
            server = (
                marimo.create_asgi_app(include_code=True)
                .with_app(path="", root=f"{tempdir}/visualizer.py")
            )
            app = FastAPI()
            app.mount("/", server.build())
            uvicorn.run(app, host=host, port=port)

NOTEBOOK_TEMPLATE = """
import marimo
app = marimo.App(width="medium")

@app.cell
def _():
    from nanoconfig.data.source import DataRepository
    from {visualizer_module} import {visualizer_name}
    repo = DataRepository.default()
    data = repo.lookup("{data_sha256}")
    visualizer = {visualizer_name}()
    return repo, data, visualizer

@app.cell
def _():
    visualizer.show(data)
    return ()

if __name__ == "__main__":
    app.run()
"""
