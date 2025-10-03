from . import Data
from .source import DataRepository, DataSource
from .fs import FsData

from dataclasses import dataclass
from pathlib import PurePath
from rich.progress import Progress

from ..utils import download_url

from fsspec.implementations.dirfs import DirFileSystem

import functools
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import hashlib
import huggingface_hub as hf
import logging
import itertools
import typing as ty
import re
import json

logger = logging.getLogger(__name__)

MIME_TYPE_SCHEMA = {
    "data/image+class": pa.schema([
        pa.field("image", pa.struct([
            pa.field("bytes", pa.binary()),
            pa.field("path", pa.string())
        ]), metadata={"mime_type": "image/encoded"}),
        pa.field("class", pa.int64(), metadata={"mime_type": "class/id"})
    ], metadata={"mime_type": "data/image+class"}),

    "data/text": pa.schema([
        pa.field("text", pa.string(), metadata={"mime_type": "text/plain"}),
    ], metadata={"mime_type": "data/text"}),

}
SCHEMA_CONVERTERS : dict[pa.Schema, ty.Callable[[pa.RecordBatch], pa.RecordBatch]] = {}

def image_class_converter(x: pa.RecordBatch, image_column: str, class_column: str) -> pa.RecordBatch:
    classes = []
    if b"huggingface" in x.schema.metadata:
        classes = (json.loads(x.schema.metadata[b"huggingface"]).get("info", {}).get("features", {})
                .get("label", {}).get("names", []))
    # Rename the columns if necessary
    if image_column != "image" or class_column != "class":
        x = x.rename_columns({image_column: "image", class_column: "class"})
    # Cast to the desired schema, customizing the "class" field
    # metadata to include label names
    final_schema = MIME_TYPE_SCHEMA["data/image+class"]
    if classes:
        class_field = final_schema.field("class").with_metadata(
            {"class_names": json.dumps(classes), "mime_type": "class/id"}
        )
        final_schema = final_schema.remove(1).insert(1, class_field)
    return x.cast(final_schema)

def celeba_converter(x: pa.RecordBatch) -> pa.RecordBatch:
    # remove celeb_id
    # get all of the boolean attributes
    x = x.drop_columns("celeb_id")
    schema = x.schema
    attrs = []
    attr_fields = (schema.field(n) for n in schema.names)
    attr_fields = list(f for f in attr_fields if f.type == pa.bool_())
    for field in attr_fields:
        attrs.append(x.column(field.name).to_numpy(zero_copy_only=False))
    attrs = np.stack(attrs, axis=-1)
    x = x.drop_columns([f.name for f in attr_fields])
    attrs = pa.FixedSizeListArray.from_arrays(
        pa.array(attrs.flatten(), type=pa.bool_()),
        list_size=attrs.shape[-1]
    )
    x = x.append_column("attrs", attrs)
    x = x.cast(pa.schema([
        pa.field("image", pa.struct([
            pa.field("bytes", pa.binary()),
            pa.field("path", pa.string())
        ]), metadata={"mime_type": "image/encoded"}),
        pa.field("attrs", pa.list_(pa.bool_(), len(attr_fields)),
            metadata={"mime_type": "class/attributes"})
    ], metadata={"mime_type": "data/image+attrs"}))
    return x

SCHEMA_CONVERTERS[pa.schema([
    pa.field("img", pa.struct([
        pa.field("bytes", pa.binary()),
        pa.field("path", pa.string())
    ])),
    pa.field("label", pa.int64())
])] = functools.partial(
    image_class_converter, image_column="img", class_column="label"
)
# Override so that we get the label metadata
SCHEMA_CONVERTERS[pa.schema([
    pa.field("image", pa.struct([
        pa.field("bytes", pa.binary()),
        pa.field("path", pa.string())
    ])),
    pa.field("label", pa.int64())
])] = functools.partial(image_class_converter, image_column="image", class_column="label")

SCHEMA_CONVERTERS[
    pa.schema([
        pa.field("image", pa.struct([
            pa.field("bytes", pa.binary()),
            pa.field("path", pa.string())
        ])),
        pa.field("celeb_id", pa.int64())
    ])
] = celeba_converter

def count_fields(child: pa.Schema, parent: pa.Schema) -> int:
    child_fields = {child.field(f) for f in child.names}
    parent_fields = {parent.field(f) for f in parent.names}
    for field in child_fields:
        if field not in parent_fields:
            return 0
    return len(child)

@dataclass
class HfDataSource(DataSource):
    repo: str
    subset: str | None
    rev: str
    content_sha: str
    mime_type: str | None = None

    @staticmethod
    def from_repo(repo: str, subset: str | None = None,
                rev: str | None = None) -> "HfDataSource":
        if rev is None:
            refs = hf.list_repo_refs(repo, repo_type="dataset")
            all_refs = refs.branches + refs.converts
            branches = [r for r in all_refs if r.ref == "refs/convert/parquet"]
            if not branches:
                raise ValueError(f"Ref refs/convert/parquet not found in: {[r.ref for r in all_refs]}")
            branch = branches[0]
            rev = branch.target_commit
        info = hf.dataset_info(repo, revision=rev)
        sha = info.sha
        assert sha is not None
        root_fs = hf.HfFileSystem()
        fs = DirFileSystem(PurePath("datasets") / f"{info.id}@{rev}", root_fs)
        splits = _hf_collect_split_files(fs, subset)
        if not splits:
            raise ValueError("Unable to find parquet files")
        return HfDataSource(repo, subset, rev, sha)

    @ty.override
    def prepare(self, repo: DataRepository | None = None) -> Data:
        if repo is None:
            repo = DataRepository.default()
        data = repo.lookup(self.sha256)
        if data is not None:
            return data
        info = hf.dataset_info(self.repo, revision=self.rev)
        root_fs = hf.HfFileSystem()
        fs = DirFileSystem(PurePath("datasets") / f"{info.id}@{self.rev}", root_fs)
        splits = _hf_collect_split_files(fs, self.subset)
        with repo.init(self.sha256) as writer:
            for split, split_fragments in splits.items():
                ds = pq.ParquetDataset(split_fragments, fs)
                schema = ds.schema
                converter = lambda x: x
                if not schema.metadata.get("mime_type", None):
                    # find the most specific schema
                    schemas = sorted(SCHEMA_CONVERTERS.items(), key=lambda item: count_fields(item[0], schema), reverse=True)
                    target_schema, converter = schemas[0]
                    if count_fields(target_schema, schema) == 0:
                        raise ValueError(f"Unsupported schema: {schema}")
                with writer.split(split) as split_writer:
                    for fragment in ds.fragments:
                        for batch in fragment.to_batches():
                            batch = converter(batch)
                            split_writer.write_batch(batch)
            return writer.close()

    @property
    def id(self):
        return "hf/" + self.repo + "/" + self.sha256

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            (self.repo + "-" + self.rev + "-" + self.content_sha).encode("utf-8")
        ).hexdigest()

FILENAME_REGEX = re.compile(
    r'(?P<split>[^.-]+)(?:-[\w\W]+)?(?:\.parquet)'
)
def _hf_collect_split_files(fs, subset):
    files = fs.ls("/", detail=False)
    if subset is None:
        if not files:
            raise ValueError("No files found in the repository")
        subset = files[0]
    if subset not in files:
        raise ValueError(f"Subset {subset} not found in repository. Valid subsets are: {files}")
    splits = {}
    for f in fs.glob(f"{subset}/**/*.parquet", detail=True):
        split = f.split("/")[1]
        splits.setdefault(split, []).append(f)
    return splits
