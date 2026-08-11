"""Shared safety, hashing, environment, and memory-budget helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import locale
import mmap
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
from typing import Any, BinaryIO


UINT64_MAX = (1 << 64) - 1
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_HEADER_BYTES = 256 * 1024 * 1024
DEFAULT_STAGING_BYTES = 1024 * 1024
MIN_STAGING_BYTES = 4096


class CompilerError(RuntimeError):
    exit_code = 4


class InvalidPlanError(CompilerError):
    exit_code = 2


class SourceVerificationError(CompilerError):
    exit_code = 3


class DataError(CompilerError):
    exit_code = 4


class OutputError(CompilerError):
    exit_code = 5


class ReproducibilityError(CompilerError):
    exit_code = 6


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidPlanError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path, maximum_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    try:
        size = path.stat().st_size
        if size > maximum_bytes:
            raise InvalidPlanError(f"JSON exceeds {maximum_bytes} bytes: {path}")
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=reject_duplicate_keys)
    except CompilerError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InvalidPlanError(f"cannot parse JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise InvalidPlanError(f"JSON root must be an object: {path}")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe_relative_path(value: object, description: str) -> PurePosixPath:
    text = str(value)
    if not text or "\\" in text or "\x00" in text:
        raise InvalidPlanError(f"unsafe {description}: {text!r}")
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or any(
        part in ("", ".", "..") for part in path.parts
    ):
        raise InvalidPlanError(f"unsafe {description}: {text!r}")
    return path


def checked_shape(value: object, description: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise InvalidPlanError(f"{description} must be an array")
    shape: list[int] = []
    elements = 1
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
            raise InvalidPlanError(f"{description} contains an invalid dimension")
        if dimension != 0 and elements > UINT64_MAX // dimension:
            raise InvalidPlanError(f"{description} product exceeds uint64")
        elements *= dimension
        shape.append(dimension)
    return tuple(shape)


DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def tensor_bytes(dtype: str, shape: tuple[int, ...], description: str) -> int:
    if dtype not in DTYPE_BYTES:
        raise InvalidPlanError(f"unsupported dtype {dtype!r}: {description}")
    size = DTYPE_BYTES[dtype]
    for dimension in shape:
        if dimension != 0 and size > UINT64_MAX // dimension:
            raise InvalidPlanError(f"tensor byte count exceeds uint64: {description}")
        size *= dimension
    return size


def current_rss_bytes() -> int:
    status = Path("/proc/self/status")
    if status.is_file():
        try:
            for line in status.read_text(encoding="ascii").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except (OSError, UnicodeError, ValueError):
            pass
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, ValueError):
        return 0


def peak_rss_bytes() -> int:
    status = Path("/proc/self/status")
    if status.is_file():
        try:
            for line in status.read_text(encoding="ascii").splitlines():
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
        except (OSError, UnicodeError, ValueError):
            pass
    return current_rss_bytes()


@dataclass
class BoundedWorkspace:
    host_memory_cap_bytes: int
    staging_bytes: int = DEFAULT_STAGING_BYTES

    def __post_init__(self) -> None:
        if self.host_memory_cap_bytes <= 0:
            raise InvalidPlanError("--max-host-memory must be positive")
        if self.staging_bytes < MIN_STAGING_BYTES:
            raise InvalidPlanError(
                f"staging buffer must be at least {MIN_STAGING_BYTES} bytes"
            )
        self.baseline_rss_bytes = current_rss_bytes()
        initial_peak = peak_rss_bytes()
        if max(self.baseline_rss_bytes, initial_peak) >= self.host_memory_cap_bytes:
            raise InvalidPlanError(
                "host-memory cap is not above compiler baseline/peak RSS: "
                f"baseline={self.baseline_rss_bytes} peak={initial_peak} "
                f"cap={self.host_memory_cap_bytes}"
            )
        available = self.host_memory_cap_bytes - max(
            self.baseline_rss_bytes, initial_peak
        )
        if self.staging_bytes > available:
            raise InvalidPlanError(
                "staging buffer exceeds memory available under host cap: "
                f"staging={self.staging_bytes} available={available}"
            )
        self.buffer = bytearray(self.staging_bytes)
        # Touch one byte per page without allocating a second staging-sized object.
        for offset in range(0, self.staging_bytes, 4096):
            self.buffer[offset] = 0
        self.max_observed_rss_bytes = max(
            self.baseline_rss_bytes, current_rss_bytes(), peak_rss_bytes()
        )
        self.max_header_bytes = 0
        self.max_tensor_bytes = 0
        self.max_mapped_window_bytes = 0
        self.check("staging-buffer allocation")

    def check(self, operation: str) -> None:
        current = current_rss_bytes()
        peak = peak_rss_bytes()
        observed = max(current, peak)
        self.max_observed_rss_bytes = max(self.max_observed_rss_bytes, observed)
        if observed > self.host_memory_cap_bytes:
            raise DataError(
                f"host-memory cap exceeded during {operation}: "
                f"rss={current} peak={peak} cap={self.host_memory_cap_bytes}"
            )

    def record_header(self, byte_count: int, operation: str) -> None:
        self.max_header_bytes = max(self.max_header_bytes, byte_count)
        self.check(operation)

    def record_tensor(self, byte_count: int) -> None:
        self.max_tensor_bytes = max(self.max_tensor_bytes, byte_count)

    def hash_range(self, path: Path, offset: int, length: int) -> str:
        digest = hashlib.sha256()
        try:
            with path.open("rb", buffering=0) as stream:
                stream.seek(offset)
                remaining = length
                view = memoryview(self.buffer)
                while remaining:
                    requested = min(remaining, len(view))
                    read = stream.readinto(view[:requested])
                    if read != requested:
                        raise SourceVerificationError(
                            f"short read while hashing {path}: "
                            f"expected {requested}, got {read}"
                        )
                    digest.update(view[:read])
                    remaining -= read
                    self.check(f"hashing {path.name}")
        except CompilerError:
            raise
        except OSError as error:
            raise SourceVerificationError(f"cannot hash {path}: {error}") from error
        return digest.hexdigest()

    def _mapped_tensor_range(
        self, path: Path, offset: int, length: int, output: BinaryIO | None
    ) -> tuple[str, str]:
        source_digest = hashlib.sha256()
        output_digest = hashlib.sha256()
        self.record_tensor(length)
        try:
            file_size = path.stat().st_size
            if offset < 0 or length < 0 or offset > file_size - length:
                raise DataError(
                    f"mapped tensor range is outside {path}: "
                    f"offset={offset} length={length} size={file_size}"
                )
            with path.open("rb", buffering=0) as stream:
                position = offset
                remaining = length
                while remaining:
                    requested = min(remaining, self.staging_bytes)
                    map_offset = (
                        position // mmap.ALLOCATIONGRANULARITY
                    ) * mmap.ALLOCATIONGRANULARITY
                    delta = position - map_offset
                    map_length = delta + requested
                    self.max_mapped_window_bytes = max(
                        self.max_mapped_window_bytes, map_length
                    )
                    with mmap.mmap(
                        stream.fileno(),
                        length=map_length,
                        access=mmap.ACCESS_READ,
                        offset=map_offset,
                    ) as mapped:
                        view = memoryview(mapped)[delta : delta + requested]
                        try:
                            source_digest.update(view)
                            output_digest.update(view)
                            if output is not None:
                                output.write(view)
                            self.check(f"mapping tensor bytes from {path.name}")
                        finally:
                            view.release()
                    position += requested
                    remaining -= requested
        except CompilerError:
            raise
        except (OSError, ValueError) as error:
            raise DataError(f"cannot map tensor range from {path}: {error}") from error
        return source_digest.hexdigest(), output_digest.hexdigest()

    def hash_tensor_range(self, path: Path, offset: int, length: int) -> str:
        source_hash, _ = self._mapped_tensor_range(path, offset, length, None)
        return source_hash

    def copy_range(
        self,
        source: Path,
        offset: int,
        length: int,
        output: BinaryIO,
        *,
        track_tensor: bool = True,
    ) -> tuple[str, str]:
        source_digest = hashlib.sha256()
        output_digest = hashlib.sha256()
        if track_tensor:
            return self._mapped_tensor_range(source, offset, length, output)
        try:
            with source.open("rb", buffering=0) as stream:
                stream.seek(offset)
                remaining = length
                view = memoryview(self.buffer)
                while remaining:
                    requested = min(remaining, len(view))
                    read = stream.readinto(view[:requested])
                    if read != requested:
                        raise DataError(
                            f"short file read from {source}: "
                            f"expected {requested}, got {read}"
                        )
                    chunk = view[:read]
                    output.write(chunk)
                    source_digest.update(chunk)
                    output_digest.update(chunk)
                    remaining -= read
                    self.check(f"copying {source.name}")
        except CompilerError:
            raise
        except OSError as error:
            raise OutputError(f"cannot stream tensor from {source}: {error}") from error
        return source_digest.hexdigest(), output_digest.hexdigest()

    def telemetry(self) -> dict[str, int]:
        self.check("telemetry")
        return {
            "host_memory_cap_bytes": self.host_memory_cap_bytes,
            "baseline_rss_bytes": self.baseline_rss_bytes,
            "peak_rss_bytes": self.max_observed_rss_bytes,
            "staging_buffer_bytes": self.staging_bytes,
            "maximum_header_bytes": self.max_header_bytes,
            "maximum_tensor_bytes": self.max_tensor_bytes,
            "maximum_mapped_window_bytes": self.max_mapped_window_bytes,
        }


def write_file_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("xb") as stream:
            os.chmod(temporary, 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except FileExistsError as error:
        raise OutputError(f"temporary output already exists: {temporary}") from error
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise OutputError(f"cannot write {path}: {error}") from error


def fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise OutputError(f"cannot fsync directory {path}: {error}") from error


def git_compiler_identity(repository_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise InvalidPlanError(f"cannot determine compiler git identity: {error}") from error
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise InvalidPlanError(f"invalid compiler git commit: {commit!r}")
    return commit, dirty


def environment_identity() -> dict[str, str]:
    try:
        active_locale = locale.setlocale(locale.LC_CTYPE)
    except locale.Error:
        active_locale = "unknown"
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "byteorder": sys.byteorder,
        "locale": active_locale,
    }
