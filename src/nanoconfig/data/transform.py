import abc
import typing as ty
import hashlib

from . import Data
from .source import DataSource, DataRepository

class DataTransform(abc.ABC):
    @property
    @abc.abstractmethod
    def sha256(self) -> str:
        pass

    @abc.abstractmethod
    def transform(self, data: Data, repo: DataRepository | None = None) -> Data:
        pass

# A data pipeline is a source and a sequence of transformations
class DataPipeline(DataSource):
    def __init__(self, source: DataSource,
                 *transformations: DataTransform):
        self.source = source
        self.transformations : ty.Sequence[DataTransform] = transformations

    @staticmethod
    def compose(source: DataSource, *transformations: DataTransform):
        return DataPipeline(source, *transformations)

    @ty.override
    def prepare(self, repo: DataRepository | None = None) -> Data:
        data = self.source.prepare(repo)
        for t in self.transformations:
            data = t.transform(data, repo)
        return data

    @property
    @ty.override
    def sha256(self) -> str:
        return self.transformed_sha256(self.source, *self.transformations)

    @staticmethod
    def transformed_sha256(data: Data | DataSource, *transforms: DataTransform):
        # Repeatedly hash so that composing pipelines
        # yields the same hash as one big pipeline
        result = hashlib.sha256(data.sha256.encode("utf-8")).hexdigest()
        for t in transforms:
            result = hashlib.sha256((result + "-" + t.sha256).encode("utf-8")).hexdigest()
        return result

class DropColumn(DataTransform):
    def __init__(self, column: str):
        self.column = column

    @property
    @ty.override
    def sha256(self) -> str:
        return hashlib.sha256(("drop" + self.column).encode("utf-8")).hexdigest()

    @ty.override
    def transform(self, data: Data, repo: DataRepository | None = None) -> Data:
        repo = repo or DataRepository.default()
        dest_sha = DataPipeline.transformed_sha256(data, self)
        final = repo.lookup(dest_sha)
        if final is not None:
            return final
        with repo.initialize(dest_sha) as writer:
            for name in data.split_infos().keys():
                split = data.split(name)
                assert split is not None
                with writer.split(name) as split_writer:
                    for batch in split.to_batches():
                        batch = batch.drop_columns(self.column)
                        split_writer.write_batch(batch)
            return writer.close()

drop_label = DropColumn("label")
