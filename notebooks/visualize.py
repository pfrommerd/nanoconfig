

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from nanoconfig.data.source import DataRepository
    from nanoconfig.data.visualizer import DataVisualizer
    repo = DataRepository.default()
    data = repo.lookup("cifar10")
    return DataVisualizer, data


@app.cell
def _(DataVisualizer, data):
    visualizer = DataVisualizer()
    visualizer.show(data)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
