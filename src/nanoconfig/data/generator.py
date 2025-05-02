from sys import stderr
from torch._vendor.packaging.version import LocalType
from .source import DataSource, DataRepository
from .fs import FsData
from . import Data

from dataclasses import dataclass

from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem

import typing as ty
import hashlib
import tempfile
import logging
import subprocess

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class GeneratorSource(DataSource):
    executable: str
    args: ty.Sequence[str]
    sha: str

    @staticmethod
    def from_command(executable: str, *args: str) -> "GeneratorSource":
        # Run with --sha256 to generate just a SHA256 hash of the output
        output = subprocess.run((executable,) + tuple(args) + ("--sha256",),
                            capture_output=True)
        out = output.stdout.decode() + output.stderr.decode()
        if output.returncode != 0:
            raise ValueError(f"Command ({executable} {" ".join(args)}) failed: {out}")
        sha = out.strip()
        if len(sha) != len(hashlib.sha256(b"").hexdigest()):
            raise ValueError(f"Invalid SHA256 hash: {sha}")
        return GeneratorSource(executable, args, sha)

    def prepare(self, repo: DataRepository | None = None) -> Data:
        if repo is None:
            repo = DataRepository.default()

        with tempfile.TemporaryDirectory() as tempdir:
            logger.info(f"Executing: {self.executable} {' '.join(self.args)} --output-dir {str(tempdir)}")
            res = subprocess.run(
                (self.executable,) + tuple(self.args) + ("--output-dir", str(tempdir)),
                stdout=None, stderr=subprocess.STDOUT,
            )
            res.check_returncode()
            fs = DirFileSystem(str(tempdir), LocalFileSystem())
            data = FsData(fs, self.sha256)
            with repo.init(self.sha256) as w:
                w.write(data)
                data = w.close()
                return data

    @property
    def sha256(self) -> str:
        return self.sha
