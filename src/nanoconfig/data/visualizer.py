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
import numpy as np
import pandas as pd

from . import Data, utils
from .source import DataRepository
from ..experiment import NestedResult

def generic_visualizer(data: ds.Dataset):
    classes = data.schema.metadata.get(b"classes", None)
    if classes: classes = json.loads(classes.decode())
    for batch in data.to_batches():
        rows = {}
        for column in batch.column_names:
            if column == "image":
                image_data = batch.column("image").field("bytes")
                rows["image"] = [PIL.Image.open(io.BytesIO(img.as_py())) for img in image_data]
            elif column == "label":
                labels = utils.as_numpy(batch.column("label"))
                rows["label"] = labels if classes is None else np.array([classes[label] for label in labels])
            else:
                rows[column] = utils.as_numpy(batch.column(column))
        N = max(len(v) for v in rows.values()) if rows else 0
        for i in range(N):
            yield {k: v[i] for k, v in rows.items() if len(v) > i}

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

    def add_visualizers(self, mime_type: str, visualizer: ty.Callable):
        self._visualizers[mime_type] = visualizer

    def as_dataframe(self, split: ds.Dataset) -> pd.DataFrame:
        visualizer = self._visualizers.get(split.schema.metadata.get(b"mime_type").decode(), generic_visualizer)
        return pd.DataFrame(list(visualizer(split)))

    def show(self, data: Data):
        import marimo as mo
        import pandas as pd
        splits = data.split_infos().values()
        def load_split(name):
            split = data.split(name)
            assert split is not None, f"Split '{name}' not found"
            df = self.as_dataframe(split)
            return mo.vstack([
                mo.ui.data_explorer(df),
                mo.ui.dataframe(df, page_size=40)
            ])
        return mo.ui.tabs({
            split.name: load_split(split.name) for split in splits
        })
        # return mo.ui.tabs({
        #     split.name: mo.lazy(functools.partial(load_split, split.name)) for split in splits
        # })
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
            args = "()"
        elif "." in visualizer_type:
            idx = visualizer_type.rfind(".")
            module, name = visualizer_type[:idx], visualizer_type[idx+1:].split("(")[0]
            idx = visualizer_type.find("(")
            args = "" if idx < 0 else visualizer_type[idx:]
        else:
            raise ValueError(f"Invalid visualizer: {visualizer_type}")
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
                    data_sha256=data.sha256,
                    visualizer_module=module,
                    visualizer_name=name,
                    visualizer_args=args
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
    visualizer = {visualizer_name}{visualizer_args}
    return repo, data, visualizer

@app.cell
def _():
    visualizer.show(data)
    return ()

if __name__ == "__main__":
    app.run()
"""
