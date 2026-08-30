"""Deterministic, resumable prediction shards for long detector evaluations.

Each shard is a canonical JSONL stream compressed with a filename-free gzip
header whose timestamp is fixed to zero. The first record binds all portable
experiment identities and the shard's ordered image selection. Every remaining
record contains exactly one image ID and one raw JSON-compatible detector
output; camera provenance is intentionally not repeated at row granularity.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .._canonical import canonical_sha256, freeze_json_value, json_value, nfc_string

_FORMAT = "phycam_prediction_jsonl_gzip_v1"
_HEADER_TYPE = "prediction_shard_header"
_ROW_TYPE = "prediction_shard_row"
_RECEIPT_TYPE = "prediction_shard_receipt"
_INDEX_TYPE = "prediction_shard_index"
_HEADER_KEYS = {
    "schema_version",
    "record_type",
    "format",
    "run",
    "dataset",
    "model",
    "camera_profile_sha256",
    "condition",
    "shard",
    "header_sha256",
}
_ROW_KEYS = {"schema_version", "record_type", "image_id", "prediction"}
_RECEIPT_KEYS = {
    "schema_version",
    "record_type",
    "format",
    "shard_file",
    "sha256",
    "compressed_bytes",
    "prediction_count",
    "jsonl_record_count",
    "header_sha256",
    "receipt_sha256",
}


def _strict_json(value: Any, *, label: str) -> Any:
    """Normalize an already JSON-compatible value without accepting arrays."""

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite float")
        return 0.0 if value == 0.0 else value
    if isinstance(value, str):
        return nfc_string(value, field_name=label)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise TypeError(f"{label} mappings require string keys")
            key = nfc_string(raw_key, field_name=f"{label} key")
            if key in normalized:
                raise ValueError(f"{label} has keys that collide after NFC normalization")
            normalized[key] = _strict_json(item, label=f"{label}.{key}")
        return dict(sorted(normalized.items(), key=lambda pair: pair[0].encode("utf-8")))
    if isinstance(value, (list, tuple)):
        return [_strict_json(item, label=f"{label} item") for item in value]
    raise TypeError(f"{label} must be JSON-compatible; got {type(value).__name__}")


def _mapping(value: Any, *, label: str, nonempty: bool = True) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    normalized = _strict_json(value, label=label)
    assert isinstance(normalized, dict)
    if nonempty and not normalized:
        raise ValueError(f"{label} must not be empty")
    return normalized


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _image_id(value: Any, *, label: str = "image_id") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _image_ids(values: Sequence[int], *, label: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be an ordered sequence")
    result = tuple(_image_id(value, label=f"{label} item") for value in values)
    if not result:
        raise ValueError(f"{label} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicate image IDs")
    return result


def _selection_record(image_ids: Sequence[int]) -> dict[str, Any]:
    ordered = _image_ids(image_ids, label="ordered_image_ids")
    payload = {"ordered_image_ids": list(ordered)}
    return {
        "count": len(ordered),
        "first": ordered[0],
        "last": ordered[-1],
        **payload,
        "selection_sha256": canonical_sha256(payload),
    }


def make_prediction_shard_header(
    *,
    run: Mapping[str, Any],
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
    camera_profile_sha256: str,
    condition: Mapping[str, Any],
    image_ids: Sequence[int],
) -> dict[str, Any]:
    """Create one self-validating schema-v2 prediction-shard header."""

    profile_hash = _sha256(camera_profile_sha256, label="camera_profile_sha256")
    normalized_condition = _mapping(condition, label="condition")
    fixed_profile = normalized_condition.get("fixed_profile_sha256")
    if fixed_profile is not None and fixed_profile != profile_hash:
        raise ValueError("condition is bound to a different camera profile")
    payload = {
        "schema_version": 2,
        "record_type": _HEADER_TYPE,
        "format": _FORMAT,
        "run": _mapping(run, label="run"),
        "dataset": _mapping(dataset, label="dataset"),
        "model": _mapping(model, label="model"),
        "camera_profile_sha256": profile_hash,
        "condition": normalized_condition,
        "shard": _selection_record(image_ids),
    }
    return {**payload, "header_sha256": canonical_sha256(payload)}


def _validated_header(value: Mapping[str, Any]) -> dict[str, Any]:
    record = _mapping(value, label="prediction shard header")
    if set(record) != _HEADER_KEYS:
        raise ValueError("prediction shard header has missing or unknown fields")
    if record.get("schema_version") != 2 or record.get("record_type") != _HEADER_TYPE:
        raise ValueError("prediction shard header requires schema version 2 and the header type")
    if record.get("format") != _FORMAT:
        raise ValueError("prediction shard header format is unsupported")
    shard = record.get("shard")
    if not isinstance(shard, Mapping):
        raise TypeError("prediction shard header requires a shard selection mapping")
    image_ids = shard.get("ordered_image_ids")
    rebuilt = make_prediction_shard_header(
        run=record["run"],
        dataset=record["dataset"],
        model=record["model"],
        camera_profile_sha256=record["camera_profile_sha256"],
        condition=record["condition"],
        image_ids=image_ids,
    )
    if record != rebuilt:
        raise ValueError("prediction shard header identity or selection metadata does not match")
    return rebuilt


def make_prediction_record(image_id: int, prediction: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap one raw detector output without adding repeated camera provenance."""

    raw = _mapping(prediction, label="prediction", nonempty=False)
    if "camera_provenance" in raw:
        raise ValueError("camera provenance belongs in the shard identity, not prediction rows")
    return {
        "schema_version": 2,
        "record_type": _ROW_TYPE,
        "image_id": _image_id(image_id),
        "prediction": raw,
    }


def _validated_prediction_record(value: Mapping[str, Any]) -> dict[str, Any]:
    record = _mapping(value, label="prediction record")
    if set(record) != _ROW_KEYS:
        raise ValueError("prediction record has missing or unknown fields")
    if record.get("schema_version") != 2 or record.get("record_type") != _ROW_TYPE:
        raise ValueError("prediction record requires schema version 2 and the row type")
    rebuilt = make_prediction_record(record["image_id"], record["prediction"])
    if record != rebuilt:
        raise ValueError("prediction record is not canonical")
    return rebuilt


def _json_line(value: Mapping[str, Any]) -> bytes:
    normalized = _strict_json(value, label="JSONL record")
    return (
        json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _parse_json_line(line: bytes, *, label: str) -> dict[str, Any]:
    if not line.endswith(b"\n") or line == b"\n":
        raise ValueError(f"{label} must be one nonempty newline-terminated JSON record")
    try:
        decoded = line[:-1].decode("utf-8")
        value = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    record = _mapping(value, label=label)
    if _json_line(record) != line:
        raise ValueError(f"{label} is not in canonical JSONL form")
    return record


def prediction_shard_receipt_path(path: str | Path) -> Path:
    """Return the deterministic receipt sidecar path for one shard."""

    shard = Path(path)
    return shard.with_name(f"{shard.name}.receipt.json")


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_temporary_no_clobber(temporary: Path, path: Path) -> bool:
    """Atomically publish ``temporary`` without replacing a concurrent winner."""

    try:
        # Both files live in the same directory.  A hard-link publication is
        # atomic and, unlike os.replace, fails if another worker already won.
        os.link(temporary, path)
    except FileExistsError:
        return False
    _fsync_directory(path.parent)
    return True


def _file_digest_and_size(path: Path) -> tuple[str, int]:
    """Hash one regular file without following a publication-race symlink."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"failed to inspect concurrent publication: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
    except OSError as exc:
        os.close(descriptor)
        raise RuntimeError(f"failed to inspect concurrent publication: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise RuntimeError(f"concurrent publication is not a regular file: {path}")
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(descriptor, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"failed to read concurrent publication: {path}") from exc
    return digest.hexdigest(), size


def _require_identical_concurrent_winner(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> None:
    observed_sha256, observed_bytes = _file_digest_and_size(path)
    if observed_bytes != expected_bytes or observed_sha256 != expected_sha256:
        raise RuntimeError("concurrent publication diverged from the attempted deterministic bytes")


def _atomic_write_bytes(path: Path, payload: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        published = _publish_temporary_no_clobber(temporary, path)
        if not published:
            _require_identical_concurrent_winner(
                path,
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                expected_bytes=len(payload),
            )
        return published
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_gzip(path: Path, lines: Sequence[bytes]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw,
                mtime=0,
            ) as compressed:
                for line in lines:
                    compressed.write(line)
            raw.flush()
            os.fsync(raw.fileno())
        attempted_sha256, attempted_bytes = _file_digest_and_size(temporary)
        published = _publish_temporary_no_clobber(temporary, path)
        if not published:
            _require_identical_concurrent_winner(
                path,
                expected_sha256=attempted_sha256,
                expected_bytes=attempted_bytes,
            )
        return published
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    return _file_digest_and_size(path)[0]


def _receipt(
    *,
    shard_path: Path,
    header_sha256: str,
    prediction_count: int,
) -> dict[str, Any]:
    payload = {
        "schema_version": 2,
        "record_type": _RECEIPT_TYPE,
        "format": _FORMAT,
        "shard_file": nfc_string(shard_path.name, field_name="prediction shard filename"),
        "sha256": _sha256_file(shard_path),
        "compressed_bytes": shard_path.stat().st_size,
        "prediction_count": prediction_count,
        "jsonl_record_count": prediction_count + 1,
        "header_sha256": header_sha256,
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def write_prediction_shard(
    path: str | Path,
    *,
    header: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    receipt_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Atomically write one new shard and its content receipt.

    Existing targets are never overwritten. Call
    :func:`validate_existing_prediction_shard` first to safely resume a run.
    """

    shard_path = Path(path)
    sidecar = (
        prediction_shard_receipt_path(shard_path) if receipt_path is None else Path(receipt_path)
    )
    if shard_path == sidecar:
        raise ValueError("prediction shard and receipt paths must differ")
    if shard_path.exists() or shard_path.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError("prediction shard or receipt already exists; validate it for resume")
    normalized_header = _validated_header(header)
    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        raise TypeError("predictions must be an ordered sequence")
    records = tuple(_validated_prediction_record(record) for record in predictions)
    expected_ids = tuple(normalized_header["shard"]["ordered_image_ids"])
    observed_ids = tuple(record["image_id"] for record in records)
    if observed_ids != expected_ids:
        if len(observed_ids) != len(expected_ids):
            raise ValueError("prediction cardinality does not match the shard image selection")
        if len(observed_ids) != len(set(observed_ids)):
            raise ValueError("prediction records contain duplicate image IDs")
        raise ValueError("prediction record order does not match the shard image selection")

    lines = (_json_line(normalized_header), *(_json_line(record) for record in records))
    _atomic_write_gzip(shard_path, lines)
    receipt = _receipt(
        shard_path=shard_path,
        header_sha256=normalized_header["header_sha256"],
        prediction_count=len(records),
    )
    # A concurrent worker may have won either hard-link publication.  Validate
    # the winning pair rather than replacing it or trusting filenames.  A
    # crash between publications still leaves a shard without a receipt, which
    # resume validation deliberately rejects as incomplete.
    _atomic_write_bytes(sidecar, _json_line(receipt))
    validated = validate_prediction_shard(
        shard_path,
        receipt_path=sidecar,
        expected_header=normalized_header,
    )
    return validated.receipt


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = freeze_json_value(value)
    if not isinstance(frozen, MappingProxyType):
        raise TypeError("record must be a mapping")
    return frozen


@dataclass(frozen=True, slots=True)
class PredictionShard:
    """Validated immutable header, ordered prediction rows, and receipt."""

    header: Mapping[str, Any]
    predictions: tuple[Mapping[str, Any], ...]
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "header", _immutable_mapping(self.header))
        object.__setattr__(
            self,
            "predictions",
            tuple(_immutable_mapping(record) for record in self.predictions),
        )
        object.__setattr__(self, "receipt", _immutable_mapping(self.receipt))

    @property
    def image_ids(self) -> tuple[int, ...]:
        return tuple(int(record["image_id"]) for record in self.predictions)


def _read_receipt(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"failed to read prediction shard receipt: {path}") from exc
    record = _parse_json_line(raw, label="prediction shard receipt")
    if set(record) != _RECEIPT_KEYS:
        raise ValueError("prediction shard receipt has missing or unknown fields")
    if record.get("schema_version") != 2 or record.get("record_type") != _RECEIPT_TYPE:
        raise ValueError("prediction shard receipt requires schema version 2 and the receipt type")
    if record.get("format") != _FORMAT:
        raise ValueError("prediction shard receipt format is unsupported")
    supplied_hash = record.get("receipt_sha256")
    payload = {key: value for key, value in record.items() if key != "receipt_sha256"}
    if supplied_hash != canonical_sha256(payload):
        raise ValueError("prediction shard receipt identity does not match")
    _sha256(record.get("sha256"), label="prediction shard receipt sha256")
    _sha256(record.get("header_sha256"), label="prediction shard receipt header_sha256")
    for field in ("compressed_bytes", "prediction_count", "jsonl_record_count"):
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"prediction shard receipt {field} must be a nonnegative integer")
    if record["jsonl_record_count"] != record["prediction_count"] + 1:
        raise ValueError("prediction shard receipt record counts disagree")
    return record


def validate_prediction_shard(
    path: str | Path,
    *,
    receipt_path: str | Path | None = None,
    expected_header: Mapping[str, Any] | None = None,
) -> PredictionShard:
    """Validate receipt, bytes, canonical JSONL, identities, and row coverage."""

    shard_path = Path(path)
    sidecar = (
        prediction_shard_receipt_path(shard_path) if receipt_path is None else Path(receipt_path)
    )
    if not shard_path.is_file():
        raise FileNotFoundError(f"prediction shard is missing: {shard_path}")
    if not sidecar.is_file():
        raise FileNotFoundError(f"prediction shard receipt is missing: {sidecar}")
    receipt = _read_receipt(sidecar)
    shard_name = nfc_string(shard_path.name, field_name="prediction shard filename")
    if receipt["shard_file"] != shard_name:
        raise ValueError("prediction shard receipt names a different shard file")
    if receipt["compressed_bytes"] != shard_path.stat().st_size:
        raise ValueError("prediction shard byte count does not match its receipt")
    if receipt["sha256"] != _sha256_file(shard_path):
        raise ValueError("prediction shard SHA-256 does not match its receipt")

    compressed = shard_path.read_bytes()
    if len(compressed) < 10 or compressed[:3] != b"\x1f\x8b\x08":
        raise ValueError("prediction shard is not a gzip stream")
    if compressed[3:10] != b"\0\0\0\0\0\x02\xff":
        raise ValueError("prediction shard gzip header is not deterministic level-9 mtime=0 form")
    try:
        uncompressed = gzip.decompress(compressed)
    except (OSError, EOFError) as exc:
        raise ValueError("prediction shard gzip payload is invalid") from exc
    lines = uncompressed.splitlines(keepends=True)
    if not lines or b"".join(lines) != uncompressed or not uncompressed.endswith(b"\n"):
        raise ValueError("prediction shard must contain newline-terminated JSONL records")
    header = _validated_header(_parse_json_line(lines[0], label="prediction shard header"))
    predictions = tuple(
        _validated_prediction_record(_parse_json_line(line, label=f"prediction shard row {index}"))
        for index, line in enumerate(lines[1:], start=1)
    )
    expected_ids = tuple(header["shard"]["ordered_image_ids"])
    observed_ids = tuple(record["image_id"] for record in predictions)
    if observed_ids != expected_ids:
        if len(observed_ids) != len(expected_ids):
            raise ValueError("prediction shard row cardinality does not match its header")
        if len(observed_ids) != len(set(observed_ids)):
            raise ValueError("prediction shard contains duplicate image IDs")
        raise ValueError("prediction shard row order does not match its header")
    if receipt["prediction_count"] != len(predictions):
        raise ValueError("prediction shard count does not match its receipt")
    if receipt["jsonl_record_count"] != len(lines):
        raise ValueError("prediction shard JSONL line count does not match its receipt")
    if receipt["header_sha256"] != header["header_sha256"]:
        raise ValueError("prediction shard header does not match its receipt")
    if expected_header is not None and header != _validated_header(expected_header):
        raise ValueError("existing prediction shard header does not match the expected run shard")
    return PredictionShard(header, predictions, receipt)


def validate_existing_prediction_shard(
    path: str | Path,
    *,
    expected_header: Mapping[str, Any],
    receipt_path: str | Path | None = None,
) -> PredictionShard | None:
    """Return a valid completed shard for resume, or ``None`` when absent."""

    shard_path = Path(path)
    sidecar = (
        prediction_shard_receipt_path(shard_path) if receipt_path is None else Path(receipt_path)
    )
    shard_exists = shard_path.exists()
    receipt_exists = sidecar.exists()
    if not shard_exists and not receipt_exists:
        return None
    if shard_exists != receipt_exists:
        raise RuntimeError(
            "incomplete prediction shard publication; shard and receipt must both exist"
        )
    return validate_prediction_shard(
        shard_path,
        receipt_path=sidecar,
        expected_header=expected_header,
    )


@dataclass(frozen=True, slots=True)
class PredictionShardMerge:
    """Deterministically ordered merge result and content-addressed shard index."""

    header: Mapping[str, Any]
    predictions: tuple[Mapping[str, Any], ...]
    index: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "header", _immutable_mapping(self.header))
        object.__setattr__(
            self,
            "predictions",
            tuple(_immutable_mapping(record) for record in self.predictions),
        )
        object.__setattr__(self, "index", _immutable_mapping(self.index))


def _shared_identity(header: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: json_value(header[key])
        for key in ("run", "dataset", "model", "camera_profile_sha256", "condition")
    }


def merge_prediction_shards(
    paths: Sequence[str | Path],
    *,
    expected_image_ids: Sequence[int],
) -> PredictionShardMerge:
    """Validate and merge a complete set of nonoverlapping compatible shards."""

    if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
        raise TypeError("paths must be an ordered sequence")
    shard_paths = tuple(Path(path) for path in paths)
    if not shard_paths:
        raise ValueError("at least one prediction shard is required")
    if len({path.resolve() for path in shard_paths}) != len(shard_paths):
        raise ValueError("prediction shard paths must be unique")
    expected = _image_ids(expected_image_ids, label="expected_image_ids")
    expected_ordinal = {image_id: ordinal for ordinal, image_id in enumerate(expected)}
    shards = tuple(validate_prediction_shard(path) for path in shard_paths)
    identity = _shared_identity(shards[0].header)
    identity_sha256 = canonical_sha256(identity)
    by_image: dict[int, Mapping[str, Any]] = {}
    index_entries: list[dict[str, Any]] = []
    for shard in shards:
        candidate_identity = _shared_identity(shard.header)
        if candidate_identity != identity:
            raise ValueError(
                "prediction shard run, dataset, model, profile, or condition identity drifted"
            )
        ordinals: list[int] = []
        for record in shard.predictions:
            image_id = int(record["image_id"])
            if image_id not in expected_ordinal:
                raise ValueError(f"prediction shard contains unexpected image ID {image_id}")
            if image_id in by_image:
                raise ValueError(f"prediction shards overlap at image ID {image_id}")
            by_image[image_id] = record
            ordinals.append(expected_ordinal[image_id])
        if ordinals != sorted(ordinals):
            raise ValueError("a prediction shard selection is out of global expected order")
        index_entries.append(
            {
                "header_sha256": shard.header["header_sha256"],
                "selection_sha256": shard.header["shard"]["selection_sha256"],
                "ordered_image_ids": list(shard.image_ids),
                "expected_ordinals": ordinals,
                "sha256": shard.receipt["sha256"],
                "compressed_bytes": shard.receipt["compressed_bytes"],
                "prediction_count": shard.receipt["prediction_count"],
            }
        )
    missing = [image_id for image_id in expected if image_id not in by_image]
    if missing:
        raise ValueError(f"prediction shards leave {len(missing)} expected image ID gaps")
    merged_predictions = tuple(by_image[image_id] for image_id in expected)
    index_entries.sort(
        key=lambda entry: (
            entry["expected_ordinals"][0],
            entry["selection_sha256"],
        )
    )
    merged_header = make_prediction_shard_header(
        run=identity["run"],
        dataset=identity["dataset"],
        model=identity["model"],
        camera_profile_sha256=identity["camera_profile_sha256"],
        condition=identity["condition"],
        image_ids=expected,
    )
    index_payload = {
        "schema_version": 2,
        "record_type": _INDEX_TYPE,
        "format": _FORMAT,
        "shared_identity_sha256": identity_sha256,
        "shared_identity": identity,
        "merged_header_sha256": merged_header["header_sha256"],
        "expected_selection": _selection_record(expected),
        "shards": index_entries,
    }
    index = {**index_payload, "index_sha256": canonical_sha256(index_payload)}
    return PredictionShardMerge(merged_header, merged_predictions, index)


__all__ = [
    "PredictionShard",
    "PredictionShardMerge",
    "make_prediction_record",
    "make_prediction_shard_header",
    "merge_prediction_shards",
    "prediction_shard_receipt_path",
    "validate_existing_prediction_shard",
    "validate_prediction_shard",
    "write_prediction_shard",
]
