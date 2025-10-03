import abc
import typing as ty
import hashlib
import pyarrow as pa

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

class ResizeImages(DataTransform):
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    @property
    @ty.override
    def sha256(self) -> str:
        return hashlib.sha256(("resize-images-" + str(self.width) + "-" + str(self.height)).encode("utf-8")).hexdigest()

    @ty.override
    def transform(self, data: Data, repo: DataRepository | None = None) -> Data:
        import PIL.Image
        import io

        def resize_image(image_bytes: bytes) -> bytes:
            img = PIL.Image.open(io.BytesIO(image_bytes))
            img = img.resize((self.width, self.height), PIL.Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return buf.getvalue()

        repo = repo or DataRepository.default()
        dest_sha = DataPipeline.transformed_sha256(data, self)
        final = repo.lookup(dest_sha)
        if final is not None:
            return final
        with repo.init(dest_sha) as writer:
            for name in data.split_infos().keys():
                split = data.split(name)
                assert split is not None
                with writer.split(name) as split_writer:
                    for batch in split.to_batches():
                        image_column = batch.column("image")
                        image_bytes = pa.array((resize_image(image.as_py()) for image in image_column.field("bytes")), type=pa.binary())
                        image_paths = image_column.field("path")
                        new_image_column = pa.StructArray.from_arrays([image_bytes, image_paths], names=["bytes", "path"])

                        idx = batch.schema.get_field_index("image")
                        field = batch.schema.field("image")
                        batch = batch.drop_columns(["image"])
                        batch = batch.set_column(idx, field, new_image_column)
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

resize_images = lambda w, h: ResizeImages(w, h)

set_mime_type = lambda x: SetMimeType(x)
