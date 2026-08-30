"""Cache-path-independent model and artifact provenance."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .._canonical import canonical_sha256, json_value, nfc_string


def _nonempty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = nfc_string(value, field_name=field_name)
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _execution_devices_match(requested: str, actual: str) -> bool:
    """Return whether two portable execution-device spellings can agree.

    Torch permits an omitted CUDA index to resolve to the current device and
    commonly reports ``cpu``/``mps`` with or without index zero.  Unknown
    backend-neutral spellings must match exactly; an attestation boolean may
    not override contradictory requested and observed devices.
    """

    if requested == actual:
        return True

    def parse(value: str) -> tuple[str, int | None] | None:
        pieces = value.split(":")
        if len(pieces) not in {1, 2} or pieces[0] not in {"cpu", "mps", "cuda"}:
            return None
        if len(pieces) == 1:
            return pieces[0], None
        if not pieces[1].isdigit():
            return None
        return pieces[0], int(pieces[1])

    expected = parse(requested)
    observed = parse(actual)
    if expected is None or observed is None or expected[0] != observed[0]:
        return False
    device_type, expected_index = expected
    actual_index = observed[1]
    if device_type in {"cpu", "mps"}:
        return expected_index in {None, 0} and actual_index in {None, 0}
    return expected_index is None or expected_index == actual_index


def _stream_artifact_identity(
    path: str | Path,
    *,
    chunk_size: int,
) -> tuple[int, str]:
    artifact = Path(path).expanduser()
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    size = 0
    try:
        with artifact.open("rb") as stream:
            mode = os.fstat(stream.fileno()).st_mode
            if not stat.S_ISREG(mode):
                raise FileNotFoundError(f"artifact is missing or is not a regular file: {artifact}")
            for chunk in iter(lambda: stream.read(chunk_size), b""):
                size += len(chunk)
                digest.update(chunk)
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError) as exc:
        raise FileNotFoundError(
            f"artifact is missing or is not a regular file: {artifact}"
        ) from exc
    return size, digest.hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a regular file without loading the whole artifact into memory."""

    _, digest = _stream_artifact_identity(path, chunk_size=chunk_size)
    return digest


def read_artifact_bytes(path: str | Path) -> bytes:
    """Read one regular artifact through a single opened file description.

    Callers that construct a model from a local artifact should retain this
    immutable value and give these exact bytes to the backend loader.  That
    keeps the bytes used for provenance and construction identical even if the
    caller-visible path is replaced after this read completes.
    """

    artifact = Path(path).expanduser()
    try:
        with artifact.open("rb") as stream:
            mode = os.fstat(stream.fileno()).st_mode
            if not stat.S_ISREG(mode):
                raise FileNotFoundError(f"artifact is missing or is not a regular file: {artifact}")
            return stream.read()
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError) as exc:
        raise FileNotFoundError(
            f"artifact is missing or is not a regular file: {artifact}"
        ) from exc


def describe_artifact_bytes(
    payload: bytes,
    *,
    published_name: str,
) -> dict[str, Any]:
    """Describe one immutable artifact payload without consulting a path."""

    if not isinstance(payload, bytes):
        raise TypeError("artifact payload must be immutable bytes")
    name = nfc_string(published_name, field_name="published_name")
    if not name:
        raise ValueError("published_name must not be empty")
    return {
        "name": name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def describe_artifact(path: str | Path, *, published_name: str | None = None) -> dict[str, Any]:
    """Describe a local artifact without serializing its machine-specific path."""

    artifact = Path(path).expanduser()
    name = (
        artifact.name
        if published_name is None
        else nfc_string(published_name, field_name="published_name")
    )
    if not name:
        raise ValueError("published_name must not be empty")
    size, digest = _stream_artifact_identity(artifact, chunk_size=1024 * 1024)
    return {"name": name, "bytes": size, "sha256": digest}


def verify_artifact_bytes(
    payload: bytes,
    expected: Mapping[str, Any],
    *,
    published_name: str,
) -> dict[str, Any]:
    """Return payload identity after failing closed on size or digest drift."""

    if not isinstance(expected, Mapping):
        raise TypeError("expected artifact identity must be a mapping")
    for key in ("bytes", "sha256"):
        if key not in expected:
            raise ValueError(f"expected artifact identity requires {key!r}")
    actual = describe_artifact_bytes(payload, published_name=published_name)
    mismatches = [key for key in ("bytes", "sha256") if actual[key] != expected[key]]
    if mismatches:
        details = ", ".join(
            f"{key}={actual[key]!r} (expected {expected[key]!r})" for key in mismatches
        )
        raise RuntimeError(f"artifact identity mismatch for {actual['name']}: {details}")
    return actual


def verify_artifact(
    path: str | Path,
    expected: Mapping[str, Any],
    *,
    published_name: str | None = None,
) -> dict[str, Any]:
    """Return actual identity after failing closed on size or digest drift."""

    if not isinstance(expected, Mapping):
        raise TypeError("expected artifact identity must be a mapping")
    for key in ("bytes", "sha256"):
        if key not in expected:
            raise ValueError(f"expected artifact identity requires {key!r}")
    actual = describe_artifact(path, published_name=published_name)
    mismatches = [key for key in ("bytes", "sha256") if actual[key] != expected[key]]
    if mismatches:
        details = ", ".join(
            f"{key}={actual[key]!r} (expected {expected[key]!r})" for key in mismatches
        )
        raise RuntimeError(f"artifact identity mismatch for {actual['name']}: {details}")
    return actual


def model_identity(
    *,
    backend: str,
    model_id: str,
    revision: str | None,
    artifacts: Sequence[Mapping[str, Any]],
    implementation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a content-addressed, portable detector/model contract."""

    backend = nfc_string(backend, field_name="backend")
    model_id = nfc_string(model_id, field_name="model_id")
    if not backend or not model_id:
        raise ValueError("backend and model_id must not be empty")
    if revision is not None:
        revision = nfc_string(revision, field_name="revision")
        if not revision:
            raise ValueError("revision must not be empty when supplied")
    normalized_artifacts: list[dict[str, Any]] = []
    names: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise TypeError("artifacts must contain mappings")
        try:
            name = nfc_string(artifact["name"], field_name="artifact name")
            size = artifact["bytes"]
            digest = artifact["sha256"]
        except KeyError as exc:
            raise ValueError("each artifact requires name, bytes, and sha256") from exc
        if not name or name in names:
            raise ValueError("artifact names must be non-empty and unique")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("artifact bytes must be a nonnegative integer")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
        ):
            raise ValueError("artifact sha256 must be a lowercase SHA-256 digest")
        names.add(name)
        normalized_artifacts.append({"name": name, "bytes": size, "sha256": digest})
    if not normalized_artifacts:
        raise ValueError("at least one verified artifact is required")
    normalized_artifacts.sort(key=lambda artifact: artifact["name"].encode("utf-8"))
    payload: dict[str, Any] = {
        "backend": backend,
        "model_id": model_id,
        "revision": revision,
        "artifacts": normalized_artifacts,
        "implementation": None if implementation is None else dict(implementation),
        "identity_verified": True,
    }
    return {**payload, "model_sha256": canonical_sha256(payload)}


def validate_model_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute a portable model identity from its embedded artifact records."""

    if not isinstance(record, Mapping):
        raise TypeError("model identity must be a mapping")
    normalized = json_value(record)
    expected = model_identity(
        backend=normalized.get("backend"),
        model_id=normalized.get("model_id"),
        revision=normalized.get("revision"),
        artifacts=normalized.get("artifacts", ()),
        implementation=normalized.get("implementation"),
    )
    if normalized.get("model_sha256") != expected["model_sha256"]:
        raise ValueError("model_sha256 does not match embedded model provenance")
    if normalized != expected:
        raise ValueError("model identity contains fields outside the portable contract")
    return expected


def detector_execution_identity(
    *,
    model: Mapping[str, Any],
    implementation_id: str,
    requested_device: str,
    actual_device: str,
    device_attestation: Mapping[str, Any],
    inference_execution: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one detector implementation and attested execution to its model.

    The model identity covers checkpoint bytes and backend configuration.  This
    complementary record covers the code path that executed those bytes, the
    requested and observed device, and the deterministic-inference contract.
    It is intentionally backend-neutral so lightweight test detectors and
    production Torch adapters use the same fail-closed envelope.
    """

    validated_model = validate_model_identity(model)
    requested = _nonempty(requested_device, field_name="requested_device")
    actual = _nonempty(actual_device, field_name="actual_device")
    if not _execution_devices_match(requested, actual):
        raise ValueError("requested_device and actual_device contradict the device attestation")
    if not isinstance(device_attestation, Mapping):
        raise TypeError("device_attestation must be a mapping")
    attestation = json_value(device_attestation)
    if not isinstance(attestation.get("method"), str) or not attestation["method"]:
        raise ValueError("device_attestation requires a nonempty method")
    if attestation.get("matches") is not True:
        raise ValueError("device_attestation must affirm requested/actual device agreement")
    if not isinstance(inference_execution, Mapping):
        raise TypeError("inference_execution must be a mapping")
    inference = json_value(inference_execution)
    if not isinstance(inference.get("method"), str) or not inference["method"]:
        raise ValueError("inference_execution requires a nonempty method")
    if inference.get("attested") is not True:
        raise ValueError("inference_execution must affirm deterministic execution attestation")
    payload = {
        "schema_version": 3,
        "record_type": "detector_execution_identity",
        "model_sha256": validated_model["model_sha256"],
        "implementation_id": _nonempty(
            implementation_id,
            field_name="detector implementation_id",
        ),
        "requested_device": requested,
        "actual_device": actual,
        "device_attestation": attestation,
        "inference_execution": inference,
    }
    return {**payload, "detector_execution_sha256": canonical_sha256(payload)}


def validate_detector_execution_identity(
    record: Mapping[str, Any],
    *,
    model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute a detector execution identity and optionally bind its model."""

    if not isinstance(record, Mapping):
        raise TypeError("detector execution identity must be a mapping")
    normalized = json_value(record)
    if normalized.get("schema_version") != 3:
        raise ValueError("detector execution identity requires schema_version 3")
    if normalized.get("record_type") != "detector_execution_identity":
        raise ValueError("detector execution identity has an unsupported record_type")
    model_sha256 = normalized.get("model_sha256")
    if model is None:
        if (
            not isinstance(model_sha256, str)
            or len(model_sha256) != 64
            or any(char not in "0123456789abcdef" for char in model_sha256)
        ):
            raise ValueError("detector execution requires a lowercase model_sha256")
        # Rebuild the payload directly when only the already-validated model
        # digest is available (for standalone manifest validation).
        attestation = normalized.get("device_attestation")
        inference = normalized.get("inference_execution")
        if not isinstance(attestation, Mapping) or not isinstance(inference, Mapping):
            raise ValueError("detector execution requires attestation mappings")
        if not isinstance(attestation.get("method"), str) or not attestation["method"]:
            raise ValueError("device_attestation requires a nonempty method")
        if attestation.get("matches") is not True:
            raise ValueError("device_attestation must affirm requested/actual device agreement")
        if not isinstance(inference.get("method"), str) or not inference["method"]:
            raise ValueError("inference_execution requires a nonempty method")
        if inference.get("attested") is not True:
            raise ValueError("inference_execution must affirm deterministic execution attestation")
        requested = _nonempty(
            normalized.get("requested_device"),
            field_name="requested_device",
        )
        actual = _nonempty(
            normalized.get("actual_device"),
            field_name="actual_device",
        )
        if not _execution_devices_match(requested, actual):
            raise ValueError("requested_device and actual_device contradict the device attestation")
        payload = {
            "schema_version": 3,
            "record_type": "detector_execution_identity",
            "model_sha256": model_sha256,
            "implementation_id": _nonempty(
                normalized.get("implementation_id"),
                field_name="detector implementation_id",
            ),
            "requested_device": requested,
            "actual_device": actual,
            "device_attestation": json_value(attestation),
            "inference_execution": json_value(inference),
        }
        expected = {
            **payload,
            "detector_execution_sha256": canonical_sha256(payload),
        }
    else:
        expected = detector_execution_identity(
            model=model,
            implementation_id=normalized.get("implementation_id"),
            requested_device=normalized.get("requested_device"),
            actual_device=normalized.get("actual_device"),
            device_attestation=normalized.get("device_attestation"),
            inference_execution=normalized.get("inference_execution"),
        )
    if normalized.get("detector_execution_sha256") != expected["detector_execution_sha256"]:
        raise ValueError("detector_execution_sha256 does not match embedded execution provenance")
    if normalized != expected:
        raise ValueError("detector execution identity contains fields outside the contract")
    return expected


def local_model_identity(
    path: str | Path,
    *,
    backend: str,
    model_id: str | None = None,
    revision: str | None = None,
    implementation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Content-address one local checkpoint while hiding its cache path."""

    artifact = describe_artifact(path)
    return model_identity(
        backend=backend,
        model_id=artifact["name"] if model_id is None else model_id,
        revision=revision,
        artifacts=[artifact],
        implementation=implementation,
    )


__all__ = [
    "detector_execution_identity",
    "describe_artifact",
    "describe_artifact_bytes",
    "local_model_identity",
    "model_identity",
    "read_artifact_bytes",
    "sha256_file",
    "validate_model_identity",
    "validate_detector_execution_identity",
    "verify_artifact",
    "verify_artifact_bytes",
]
