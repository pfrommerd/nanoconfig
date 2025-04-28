import pyarrow as pa
import typing as ty
import rich
import tempfile

from . import Data
from ..experiment import NestedResult

def default_visualizer(data: pa.RecordBatch) -> NestedResult:
    raise NotImplementedError("default_visualizer is not implemented")

class DataVisualizer:
    def __init__(self, visualizer: ty.Callable[[pa.RecordBatch], NestedResult] | None = None):
        if visualizer is None:
            visualizer = default_visualizer
        self._visualizer = visualizer

    def show(self, data: Data) -> NestedResult:
        return {}
        # return self._visualizer(data)

    @staticmethod
    def host_marimo_notebook(host: str, port: int,
                visualizer: str | ty.Callable,
                data: Data):
        if not isinstance(visualizer, str):
            module = visualizer.__module__
            name = visualizer.__qualname__
        else:
            module, name = visualizer.split(":")
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
                marimo.create_asgi_app()
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
    from nanoconfig.data.visualizer import DataVisualizer
    repo = DataRepository.default()
    data = repo.lookup("{data_sha256}")
    return repo, data

@app.cell
def _():
    from {visualizer_module} import {visualizer_name}
    visualizer = DataVisualizer({visualizer_name})
    visualizer.show(data)
    return (visualizer, {visualizer_name},)

if __name__ == "__main__":
    app.run()
"""
