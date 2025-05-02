import abc
import typing as ty
import hashlib

from . import Data
from .source import DataSource, DataRepository

class DataTransform(abc.ABC):
    @abc.abstractmethod
    def transform(self, data: Data, repo: DataRepository | None = None) -> Data:
        pass

    @property
    @abc.abstractmethod
    def sha256(self) -> str:
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
        result = data.sha256
        for t in transforms:
            result = hashlib.sha256((result + "-" + t.sha256).encode("utf-8")).hexdigest()
        return result

class DropColumns(DataTransform):
    def __init__(self, *columns: str):
        self.columns = set(columns)

    @property
    @ty.override
    def sha256(self) -> str:
        return hashlib.sha256(("drop-" + "-".join(self.columns)).encode("utf-8")).hexdigest()

    @ty.override
    def transform(self, data: Data, repo: DataRepository | None = None) -> Data:
        repo = repo or DataRepository.default()
        dest_sha = DataPipeline.transformed_sha256(data, self)
        final = repo.lookup(dest_sha)
        if final is not None:
            return final
        with repo.init(dest_sha) as writer:
            for name in data.split_infos().keys():
                split = data.split(name)
                assert split is not None
                # Remove the existing mime_type metadata as it may no longer be valid
                metadata = split.schema.metadata
                metadata[b"mime_type"] = "unknown"
                with writer.split(name) as split_writer:
                    for batch in split.to_batches():
                        batch = batch.drop_columns(list(self.columns))
                        batch = batch.replace_schema_metadata(metadata)
                        split_writer.write_batch(batch)
            return writer.close()

class SetMimeType(DataTransform):
    def __init__(self, mime_type: str):
        self.mime_type = mime_type

    @property
    @ty.override
    def sha256(self) -> str:
        return hashlib.sha256(("set-mime-" + self.mime_type).encode("utf-8")).hexdigest()

    @ty.override
    def transform(self, data: Data, repo: DataRepository | None = None) -> Data:
        repo = repo or DataRepository.default()
        dest_sha = DataPipeline.transformed_sha256(data, self)
        final = repo.lookup(dest_sha)
        if final is not None:
            return final
        with repo.init(dest_sha) as writer:
            for name in data.split_infos().keys():
                split = data.split(name)
                assert split is not None
                # Remove the existing mime_type metadata as it may no longer be valid
                metadata = split.schema.metadata
                metadata[b"mime_type"] = self.mime_type.encode("utf-8")
                with writer.split(name) as split_writer:
                    for batch in split.to_batches():
                        batch = batch.replace_schema_metadata(metadata)
                        split_writer.write_batch(batch)
            return writer.close()

drop_columns = lambda *x: DropColumns(*x)
drop_column = drop_columns
drop_label = DropColumns("label")

set_mime_type = lambda x: SetMimeType(x)
