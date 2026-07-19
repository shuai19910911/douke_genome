from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_CODE_LUT = np.zeros(256, dtype=np.uint8)
_VALID_LUT = np.zeros(256, dtype=np.bool_)
for _symbol, _code in ((b"A", 0), (b"C", 1), (b"G", 2), (b"T", 3)):
    _CODE_LUT[_symbol[0]] = _code
    _VALID_LUT[_symbol[0]] = True


@dataclass(frozen=True)
class PackedSequenceStoreResult:
    store_dir: Path
    base_count: int
    packed_size_bytes: int
    packed_sha256: str
    manifest_sha256: str


class PackedSequenceStoreWriter:
    def __init__(self, store_dir: Path, identity: dict[str, object]) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.parent.mkdir(parents=True, exist_ok=True)
        self.staging = Path(
            tempfile.mkdtemp(prefix=f".{self.store_dir.name}.tmp.", dir=self.store_dir.parent)
        )
        self.identity = dict(identity)
        self.packed_path = self.staging / "sequence.2bit"
        self.handle = self.packed_path.open("wb")
        self.pending = np.empty(0, dtype=np.uint8)
        self.base_count = 0
        self.contigs: list[dict[str, object]] = []
        self.callable_intervals: list[dict[str, int]] = []
        self.current_name: str | None = None
        self.current_start = 0
        self.current_contig_index = -1
        self.open_callable_start: int | None = None
        self.names: set[str] = set()
        self.closed = False

    def start_contig(self, name: str) -> None:
        if self.closed:
            raise RuntimeError("sequence store writer is closed")
        name = name.strip()
        if not name:
            raise ValueError("contig name is empty")
        if name in self.names:
            raise ValueError(f"duplicate contig name: {name}")
        self._finish_contig()
        self.names.add(name)
        self.current_name = name
        self.current_start = self.base_count
        self.current_contig_index += 1

    def _append_callable(self, start: int, end: int) -> None:
        if end > start:
            self.callable_intervals.append(
                {
                    "contig_index": self.current_contig_index,
                    "start": start,
                    "length": end - start,
                }
            )

    def _update_callable(self, valid: np.ndarray) -> None:
        chunk_start = self.base_count
        if valid.size == 0:
            return
        padded = np.empty(valid.size + 2, dtype=np.int8)
        padded[0] = 0
        padded[-1] = 0
        padded[1:-1] = valid
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        if self.open_callable_start is not None:
            if bool(valid[0]):
                starts = starts.copy()
                starts[0] = self.open_callable_start - chunk_start
            else:
                self._append_callable(self.open_callable_start, chunk_start)
            self.open_callable_start = None
        for start, end in zip(starts.tolist(), ends.tolist()):
            absolute_start = chunk_start + start
            absolute_end = chunk_start + end
            if end == valid.size and bool(valid[-1]):
                self.open_callable_start = absolute_start
            else:
                self._append_callable(absolute_start, absolute_end)

    def _write_codes(self, codes: np.ndarray) -> None:
        if self.pending.size:
            codes = np.concatenate((self.pending, codes))
        full_length = codes.size - (codes.size % 4)
        if full_length:
            full = codes[:full_length].reshape(-1, 4)
            packed = (
                (full[:, 0] << 6)
                | (full[:, 1] << 4)
                | (full[:, 2] << 2)
                | full[:, 3]
            ).astype(np.uint8, copy=False)
            self.handle.write(packed.tobytes())
        self.pending = codes[full_length:].copy()

    def add_bases(self, bases: bytes) -> None:
        if self.closed:
            raise RuntimeError("sequence store writer is closed")
        if self.current_name is None:
            raise ValueError("bases cannot be added before a contig header")
        normalized = bases.upper()
        raw = np.frombuffer(normalized, dtype=np.uint8)
        valid = _VALID_LUT[raw]
        codes = _CODE_LUT[raw]
        self._update_callable(valid)
        self._write_codes(codes)
        self.base_count += int(raw.size)

    def _finish_contig(self) -> None:
        if self.current_name is None:
            return
        if self.open_callable_start is not None:
            self._append_callable(self.open_callable_start, self.base_count)
            self.open_callable_start = None
        self.contigs.append(
            {
                "name": self.current_name,
                "offset": self.current_start,
                "length": self.base_count - self.current_start,
            }
        )
        self.current_name = None

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def finalize(self) -> PackedSequenceStoreResult:
        if self.closed:
            raise RuntimeError("sequence store writer is closed")
        if self.current_name is None and not self.contigs:
            raise ValueError("cannot finalize an empty sequence store")
        self._finish_contig()
        if self.pending.size:
            padded = np.zeros(4, dtype=np.uint8)
            padded[: self.pending.size] = self.pending
            packed = int(padded[0]) << 6 | int(padded[1]) << 4 | int(padded[2]) << 2 | int(padded[3])
            self.handle.write(bytes((packed,)))
            self.pending = np.empty(0, dtype=np.uint8)
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        packed_size = self.packed_path.stat().st_size
        expected_size = (self.base_count + 3) // 4
        if packed_size != expected_size:
            self.abort()
            raise RuntimeError("packed sequence size mismatch")
        packed_sha = self._sha256(self.packed_path)
        manifest = {
            "schema_version": "1.0",
            "state": "READY",
            "encoding": "two_bit_acgt_non_acgt_encoded_as_a",
            "identity": self.identity,
            "base_count": self.base_count,
            "packed_size_bytes": packed_size,
            "packed_sha256": packed_sha,
            "contigs": self.contigs,
            "callable_intervals": self.callable_intervals,
        }
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        manifest_path = self.staging / "manifest.json"
        with manifest_path.open("wb") as handle:
            handle.write(manifest_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        ready = self.staging / "READY"
        with ready.open("w", encoding="ascii") as handle:
            handle.write(manifest_sha + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if self.store_dir.exists():
            self.abort()
            raise FileExistsError(f"sequence store target already exists: {self.store_dir}")
        os.replace(self.staging, self.store_dir)
        self.closed = True
        return PackedSequenceStoreResult(
            self.store_dir,
            self.base_count,
            packed_size,
            packed_sha,
            manifest_sha,
        )

    def abort(self) -> None:
        if not self.handle.closed:
            self.handle.close()
        if self.staging.exists():
            shutil.rmtree(self.staging)
        self.closed = True


class PackedSequenceStore:
    def __init__(self, store_dir: Path) -> None:
        self.store_dir = Path(store_dir)
        manifest_bytes = (self.store_dir / "manifest.json").read_bytes()
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        if (self.store_dir / "READY").read_text(encoding="ascii") != manifest_sha + "\n":
            raise ValueError("sequence store READY digest mismatch")
        self.manifest = json.loads(manifest_bytes)
        if self.manifest.get("state") != "READY":
            raise ValueError("sequence store is not READY")
        self.contigs = self.manifest["contigs"]
        self.callable_intervals = self.manifest["callable_intervals"]
        self.base_count = int(self.manifest["base_count"])
        self._packed = np.memmap(self.store_dir / "sequence.2bit", dtype=np.uint8, mode="r")

    def decode(self, start: int, length: int) -> np.ndarray:
        if start < 0 or length < 0 or start + length > self.base_count:
            raise ValueError("decode interval is outside sequence store")
        if length == 0:
            return np.empty(0, dtype=np.uint8)
        positions = np.arange(start, start + length, dtype=np.int64)
        packed = self._packed[positions // 4]
        shifts = 6 - 2 * (positions % 4)
        return ((packed >> shifts) & 0b11).astype(np.uint8)
