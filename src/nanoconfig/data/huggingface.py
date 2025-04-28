from . import Data
from .source import DataRepository, DataSource
from .fs import FsData

from dataclasses import dataclass
from pathlib import PurePath
from rich.progress import Progress

from ..utils import download_url

from fsspec.implementations.dirfs import DirFileSystem

import functools
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
    "parquet/image+label": pa.schema([
        pa.field("image", pa.struct([
            pa.field("bytes", pa.binary()),
            pa.field("path", pa.string())
        ])),
        pa.field("label", pa.int64())
    ]),
    "parquet/text": pa.schema([
        pa.field("text", pa.string()),
    ])
}

def identity_converter(x, mime_type):
    metadata = {} if x.schema.metadata is None else x.schema.metadata
    if "huggingface" in metadata:
        del metadata["huggingface"]
    metadata["mime_type"] = mime_type
    return x.replace_schema_metadata(metadata)

SCHEMA_CONVERTERS : dict[pa.Schema, ty.Callable[[pa.RecordBatch], pa.RecordBatch]] = {
    v: functools.partial(identity_converter, mime_type=k)
    for k, v in MIME_TYPE_SCHEMA.items()
}

def image_label_converter(x: pa.RecordBatch, image_column: str, label_column: str) -> pa.RecordBatch:
    metadata = {} if x.schema.metadata is None else x.schema.metadata
    labels = {}
    if b"huggingface" in metadata:
        classes = (json.loads(metadata[b"huggingface"]).get("info", {}).get("features", {})
                .get("label", {}).get("names", {}))
        if classes:
            metadata["classes"] = json.dumps(classes)
        del metadata[b"huggingface"]
    metadata["mime_type"] = "parquet/image+label"
    if image_column != "image" or label_column != "label":
        x = x.rename_columns({
            image_column: "image", label_column: "label"
        })
    return x.replace_schema_metadata(metadata)

SCHEMA_CONVERTERS[pa.schema([
    pa.field("img", pa.struct([
        pa.field("bytes", pa.binary()),
        pa.field("path", pa.string())
    ])),
    pa.field("label", pa.int64())
])] = functools.partial(
    image_label_converter, image_column="img", label_column="label"
)
# Override so that we get the label metadata
SCHEMA_CONVERTERS[MIME_TYPE_SCHEMA["parquet/image+label"]] = functools.partial(
    image_label_converter, image_column="image", label_column="label"
)

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
        logger.info(f"Using data from {info.id}")
        logger.info(f"  downloads (30 days) : {info.downloads}")
        logger.info(f"  hash                : {info.sha}")
        for split, files in splits.items():
            logger.info(f"Split: {split}")
            for file in files:
                logger.info(f"  {file}")
        logger.info(f"Using data from {info.id}")
        logger.info(f"  downloads (30 days) : {info.downloads}")
        logger.info(f"  hash                : {info.sha}")
        for split, files in splits.items():
            logger.info(f"Split: {split}")
            for file in files:
                logger.info(f"  {file}")

        with repo.initialize(self.sha256) as writer:
            for split, split_fragments in splits.items():
                ds = pq.ParquetDataset(split_fragments, fs)
                schema = ds.schema
                converter = lambda x: x
                if not schema.metadata.get("mime_type", None):
                    converter = None
                    for other_schema, other_converter in SCHEMA_CONVERTERS.items():
                        if other_schema == schema:
                            converter = other_converter
                    if converter is None:
                        raise ValueError(f"Unsupported schema: {schema}")
                with writer.split(split) as split_writer:
                    for fragment in ds.fragments:
                        for batch in fragment.to_batches():
                            batch = converter(batch)
                            split_writer.write_batch(batch)
        return FsData(fs, self.sha256, splits)

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
