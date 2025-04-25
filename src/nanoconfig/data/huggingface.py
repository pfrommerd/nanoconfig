from click.core import V
from . import Data
from .source import DataRepository, DataSource
from .fs import FsData

from dataclasses import dataclass
from pathlib import PurePath
from rich.progress import Progress

from ..utils import download_url

from fsspec.implementations.dirfs import DirFileSystem

import pyarrow as pa
import pyarrow.parquet as pq
import hashlib
import huggingface_hub as hf
import logging
import itertools
import typing as ty
import re

logger = logging.getLogger(__name__)

@dataclass
class HfDataSource(DataSource):
    repo: str
    revision: str
    content_sha: str
    mime_type: str | None = None

    @staticmethod
    def from_repo(repo: str, rev: str | None = None, mime_type: str | None = None) -> "HfDataSource":
        if rev is None:
            refs = hf.list_repo_refs(repo, repo_type="dataset")
            main_branch = refs.branches[0]
            revision = main_branch.ref
        else:
            revision = rev
        info = hf.dataset_info(repo, revision=revision)
        sha = info.sha
        assert sha is not None

        return HfDataSource(repo, revision, sha, mime_type)

    @ty.override
    def prepare(self, repo: DataRepository | None = None) -> Data:
        info = hf.dataset_info(self.repo, revision=self.revision)
        root_fs = hf.HfFileSystem()
        fs = DirFileSystem(PurePath("datasets") / info.id, root_fs)
        splits = _hf_collect_split_files(fs)
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

        # Read in the schema to get the metadata
        all_fragments = list(itertools.chain.from_iterable(splits.values()))
        if not all_fragments:
            raise ValueError("No fragments found")
        with fs.open(all_fragments[0], mode="rb") as f:
            schema = pq.ParquetFile(pa.PythonFile(f)).schema.to_arrow_schema()
        metadata = schema.metadata
        if metadata is not None:
            metadata = {k.decode("utf-8"): v.decode("utf-8") for k, v in metadata.items()}
            if "huggingface" in metadata:
                del metadata["huggingface"]
        else: metadata = {}
        if not "mime_type" in metadata:
            if self.mime_type is None:
                raise ValueError("mime_type is not set")
            metadata["mime_type"] = self.mime_type
        return FsData(fs, self.sha256, splits, metadata)

    @property
    def id(self):
        return "hf/" + self.repo + "/" + self.sha256

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            (self.repo + "-" + self.revision + "-" + self.content_sha).encode("utf-8")
        ).hexdigest()


FILENAME_REGEX = re.compile(
    r'(?P<split>[^.-]+)(?:-[\w\W]+)?(?:\.parquet)'
)
def _hf_collect_split_files(fs):
    data_path = "/"
    if fs.exists("data"):
        data_path = "data"
    splits = {}
    for f in fs.glob("**/*.parquet", detail=True):
        file_name = f.split("/")[-1]
        match = FILENAME_REGEX.match(file_name)
        if match is None:
            logger.debug(f"Skipping file {file_name} as it does not match the expected pattern.")
            continue
        split = match.group("split")
        splits.setdefault(split, []).append(f)
    return splits
