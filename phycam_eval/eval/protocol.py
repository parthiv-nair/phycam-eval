"""Fail-closed identities and validation for versioned experiments.

This module deliberately contains no detector or camera-model implementation.
It binds already-computed artifacts, selections, conditions, and stage-aware
camera provenance into portable records using the repository canonical hash.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import platform
import re
import stat
import subprocess
from argparse import ArgumentParser, Namespace
from collections.abc import Iterable, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .._canonical import canonical_sha256, json_value, nfc_string
from ..domains import DataMode, Domain, is_legal_transition
from .model_provenance import (
    validate_detector_execution_identity,
    validate_model_identity,
)

MAX_SEED = 2**64 - 1
DEFAULT_INFERENCE_SEED = 20_260_715
DEFAULT_RUNTIME_DISTRIBUTIONS = (
    "phycam-eval",
    "annotated-doc",
    "anyio",
    "certifi",
    "charset-normalizer",
    "contourpy",
    "cycler",
    "filelock",
    "fonttools",
    "fsspec",
    "h11",
    "hf-xet",
    "httpcore",
    "httpx",
    "huggingface-hub",
    "idna",
    "jinja2",
    "kiwisolver",
    "markdown-it-py",
    "markupsafe",
    "matplotlib",
    "mdurl",
    "mpmath",
    "networkx",
    "numpy",
    "opencv-python",
    "packaging",
    "pillow",
    "pip",
    "polars",
    "polars-runtime-32",
    "psutil",
    "pycocotools",
    "pygments",
    "pyparsing",
    "python-dateutil",
    "pyyaml",
    "regex",
    "requests",
    "rich",
    "safetensors",
    "scipy",
    "setuptools",
    "shellingham",
    "six",
    "sympy",
    "timm",
    "tokenizers",
    "torch",
    "torchvision",
    "tqdm",
    "transformers",
    "typer",
    "typing-extensions",
    "ultralytics",
    "ultralytics-thop",
    "urllib3",
)
RUNTIME_IDENTITY_METHOD = "phycam.runtime_reproducibility_identity.v5"
ANALYSIS_RUNTIME_IDENTITY_METHOD = "phycam.analysis_runtime_reproducibility_identity.v4"
WORKTREE_STATE_METHOD = "git_head_index_worktree_status_and_nonignored_untracked_content_sha256.v1"
INFERENCE_EXECUTION_METHOD = "phycam.torch_deterministic_inference.v2"
PROJECT_INSTALLATION_IDENTITY_METHOD = "python.importlib.metadata.direct_url_json.v1"
PACKAGE_IDENTITY_METHOD = "python.importlib.metadata.complete_distribution_universe.v2"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_DISTRIBUTION_NAME_RE = re.compile(r"[-_.]+")
_NORMALIZED_DISTRIBUTION_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


def _nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    result = nfc_string(value, field_name=label)
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return 0.0 if result == 0.0 else result


def result_run_identity() -> dict[str, object]:
    """Return the optional orchestration binding inherited by a process."""

    profile = os.environ.get("PHYCAM_RUN_PROFILE", "ad_hoc")
    run_id = os.environ.get("PHYCAM_RUN_ID")
    if profile == "archival" and not run_id:
        raise RuntimeError("PHYCAM_RUN_PROFILE=archival requires PHYCAM_RUN_ID")
    return {
        "run_profile": profile,
        "run_id": run_id,
        "archival_eligible": profile == "archival" and bool(run_id),
    }


def _git_environment() -> dict[str, str]:
    """Return a deterministic Git environment that cannot be redirected."""

    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "LANG": "C",
            "LC_ALL": "C",
            "PAGER": "cat",
        }
    )
    return environment


def _run_git(
    directory: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            "git",
            "-c",
            "color.ui=false",
            "-c",
            "core.quotepath=false",
            "-c",
            "status.renames=false",
            "-C",
            os.fspath(directory),
            *arguments,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=check,
        env=_git_environment(),
    )


def _hash_component(digest: Any, label: str, value: bytes) -> None:
    label_bytes = label.encode("ascii")
    digest.update(len(label_bytes).to_bytes(4, "big"))
    digest.update(label_bytes)
    digest.update(len(value).to_bytes(16, "big"))
    digest.update(value)


def _hash_untracked_file(digest: Any, repository: Path, relative_bytes: bytes) -> None:
    relative_text = os.fsdecode(relative_bytes)
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("Git returned a non-portable untracked path")
    path = repository / relative
    try:
        before = path.lstat()
    except OSError as exc:
        raise RuntimeError("an untracked file could not be inspected") from exc

    _hash_component(digest, "untracked_path", relative_bytes)
    if stat.S_ISREG(before.st_mode):
        _hash_component(digest, "untracked_type", b"regular")
        _hash_component(
            digest,
            "untracked_executable",
            b"1" if before.st_mode & 0o111 else b"0",
        )
        _hash_component(digest, "untracked_size", str(before.st_size).encode("ascii"))
        digest.update(b"untracked_content\x00")
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise RuntimeError("an untracked file could not be hashed") from exc
    elif stat.S_ISLNK(before.st_mode):
        _hash_component(digest, "untracked_type", b"symlink")
        try:
            target = os.fsencode(os.readlink(path))
        except OSError as exc:
            raise RuntimeError("an untracked symlink could not be inspected") from exc
        _hash_component(digest, "untracked_symlink_target", target)
    else:
        raise RuntimeError("unsupported non-regular untracked worktree entry")

    try:
        after = path.lstat()
    except OSError as exc:
        raise RuntimeError("an untracked file changed while it was hashed") from exc
    before_identity = (before.st_mode, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_mode, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError("an untracked file changed while it was hashed")


def _unavailable_git_identity(reason: str) -> dict[str, object]:
    return {
        "available": False,
        "unavailable_reason": reason,
        "commit": None,
        "branch": None,
        "detached_head": None,
        "dirty": None,
        "untracked_file_count": None,
        "ignored_files_included": False,
        "worktree_state_method": WORKTREE_STATE_METHOD,
        "worktree_state_sha256": None,
    }


def _git_repository_identity(repository_root: str | os.PathLike[str] | None) -> dict[str, object]:
    if repository_root is None:
        return _unavailable_git_identity("repository_root_not_supplied")
    candidate = Path(repository_root)
    if not candidate.is_dir():
        return _unavailable_git_identity("repository_root_unavailable")
    try:
        root_result = _run_git(candidate, "rev-parse", "--show-toplevel", check=False)
    except FileNotFoundError:
        return _unavailable_git_identity("git_executable_unavailable")
    if root_result.returncode != 0:
        return _unavailable_git_identity("not_a_git_worktree")

    repository = Path(os.fsdecode(root_result.stdout.rstrip(b"\r\n")))
    commit_result = _run_git(repository, "rev-parse", "--verify", "HEAD", check=False)
    commit = os.fsdecode(commit_result.stdout.strip()) if commit_result.returncode == 0 else None
    branch_result = _run_git(
        repository,
        "symbolic-ref",
        "--quiet",
        "--short",
        "HEAD",
        check=False,
    )
    branch = os.fsdecode(branch_result.stdout.strip()) if branch_result.returncode == 0 else None
    status = _run_git(
        repository,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
    ).stdout
    diff_options = (
        "--binary",
        "--full-index",
        "--no-color",
        "--no-ext-diff",
        "--no-indent-heuristic",
        "--no-renames",
        "--no-textconv",
        "--diff-algorithm=myers",
    )
    cached_arguments = ["diff", "--cached", *diff_options]
    if commit is not None:
        cached_arguments.append("HEAD")
    cached_arguments.append("--")
    index_diff = _run_git(repository, *cached_arguments).stdout
    worktree_diff = _run_git(repository, "diff", *diff_options, "--").stdout
    untracked_payload = _run_git(
        repository,
        "ls-files",
        "--others",
        "--exclude-standard",
        "--full-name",
        "-z",
    ).stdout
    untracked_paths = sorted(path for path in untracked_payload.split(b"\x00") if path)

    digest = hashlib.sha256()
    _hash_component(digest, "method", WORKTREE_STATE_METHOD.encode("ascii"))
    _hash_component(digest, "head", b"" if commit is None else commit.encode("ascii"))
    _hash_component(digest, "branch", b"" if branch is None else os.fsencode(branch))
    _hash_component(digest, "status", status)
    _hash_component(digest, "index_diff", index_diff)
    _hash_component(digest, "worktree_diff", worktree_diff)
    _hash_component(digest, "untracked_count", str(len(untracked_paths)).encode("ascii"))
    for relative_bytes in untracked_paths:
        _hash_untracked_file(digest, repository, relative_bytes)

    status_after = _run_git(
        repository,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
    ).stdout
    if status_after != status:
        raise RuntimeError("Git worktree state changed while its identity was computed")
    return {
        "available": True,
        "unavailable_reason": None,
        "commit": commit,
        "branch": branch,
        "detached_head": branch is None and commit is not None,
        "dirty": bool(status),
        "untracked_file_count": len(untracked_paths),
        "ignored_files_included": False,
        "worktree_state_method": WORKTREE_STATE_METHOD,
        "worktree_state_sha256": digest.hexdigest(),
    }


def validate_project_installation_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the path-free provenance of the installed project distribution."""

    if not isinstance(value, Mapping):
        raise TypeError("project installation identity must be a mapping")
    record = json_value(value)
    if set(record) != {"method", "kind", "editable", "wheel_sha256"}:
        raise ValueError("project installation identity has missing or unknown fields")
    if record["method"] != PROJECT_INSTALLATION_IDENTITY_METHOD:
        raise ValueError("project installation identity method is unsupported")
    if record["kind"] not in {"editable", "local_wheel", "index_or_unknown", "direct_url_other"}:
        raise ValueError("project installation kind is unsupported")
    if not isinstance(record["editable"], bool):
        raise TypeError("project installation editable flag must be bool")
    wheel_sha256 = record["wheel_sha256"]
    if wheel_sha256 is not None and (
        not isinstance(wheel_sha256, str) or _SHA256_RE.fullmatch(wheel_sha256) is None
    ):
        raise ValueError("project wheel identity must be null or a lowercase SHA-256 digest")
    expected = {
        "editable": (True, None),
        "local_wheel": (False, wheel_sha256),
        "index_or_unknown": (False, None),
        "direct_url_other": (False, None),
    }[record["kind"]]
    if (record["editable"], wheel_sha256) != expected or (
        record["kind"] == "local_wheel" and wheel_sha256 is None
    ):
        raise ValueError("project installation fields are internally inconsistent")
    return record


def project_installation_identity() -> dict[str, Any]:
    """Return path-free provenance for the installed project distribution."""

    try:
        distribution = metadata.distribution("phycam-eval")
    except metadata.PackageNotFoundError:
        return validate_project_installation_identity(
            {
                "method": PROJECT_INSTALLATION_IDENTITY_METHOD,
                "kind": "index_or_unknown",
                "editable": False,
                "wheel_sha256": None,
            }
        )
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        kind = "index_or_unknown"
        editable = False
        wheel_sha256 = None
    else:
        try:
            direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("installed project direct_url.json cannot be parsed") from exc
        if not isinstance(direct_url, Mapping):
            raise RuntimeError("installed project direct_url.json must be a mapping")
        directory = direct_url.get("dir_info")
        editable = bool(isinstance(directory, Mapping) and directory.get("editable") is True)
        archive = direct_url.get("archive_info")
        url = direct_url.get("url")
        parsed = urlsplit(url) if isinstance(url, str) else None
        hashes = archive.get("hashes") if isinstance(archive, Mapping) else None
        candidate = hashes.get("sha256") if isinstance(hashes, Mapping) else None
        is_local_wheel = bool(
            parsed is not None
            and parsed.scheme == "file"
            and unquote(parsed.path).lower().endswith(".whl")
            and isinstance(candidate, str)
            and _SHA256_RE.fullmatch(candidate) is not None
        )
        if editable:
            kind = "editable"
            wheel_sha256 = None
        elif is_local_wheel:
            kind = "local_wheel"
            wheel_sha256 = candidate
        else:
            kind = "direct_url_other"
            wheel_sha256 = None
    return validate_project_installation_identity(
        {
            "method": PROJECT_INSTALLATION_IDENTITY_METHOD,
            "kind": kind,
            "editable": editable,
            "wheel_sha256": wheel_sha256,
        }
    )


def _normalized_distribution_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError("installed distribution has no package name")
    normalized = _DISTRIBUTION_NAME_RE.sub("-", value).lower()
    if _NORMALIZED_DISTRIBUTION_NAME_RE.fullmatch(normalized) is None:
        raise RuntimeError("installed distribution has an invalid package name")
    return normalized


def installed_distribution_universe() -> dict[str, str]:
    """Return every installed distribution as a path-free PEP 503 name/version map."""

    versions: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = _normalized_distribution_name(distribution.metadata.get("Name"))
        version = distribution.version
        if not isinstance(version, str) or not version:
            raise RuntimeError(f"installed distribution {name} has no version")
        previous = versions.get(name)
        if previous is not None and previous != version:
            raise RuntimeError(
                f"multiple installed versions were discovered for distribution {name}"
            )
        versions[name] = version
    return dict(sorted(versions.items()))


def _runtime_package_versions() -> dict[str, object]:
    versions: dict[str, str | None] = {}
    for distribution in DEFAULT_RUNTIME_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "method": PACKAGE_IDENTITY_METHOD,
        "versions": versions,
        "installed_distribution_universe": installed_distribution_universe(),
        "project_installation": project_installation_identity(),
    }


def _parse_torch_device_spelling(value: object) -> tuple[str, int | None] | None:
    text = str(value)
    pieces = text.split(":")
    if len(pieces) not in {1, 2} or pieces[0] not in {"cpu", "mps", "cuda"}:
        return None
    if len(pieces) == 1:
        return pieces[0], None
    if not pieces[1] or not pieces[1].isdigit():
        return None
    return pieces[0], int(pieces[1])


def torch_device_attestation_matches(requested: object, actual: object) -> bool:
    """Compare device spellings without importing Torch or probing hardware."""

    expected = _parse_torch_device_spelling(requested)
    observed = _parse_torch_device_spelling(actual)
    if expected is None or observed is None or expected[0] != observed[0]:
        return False
    device_type, expected_index = expected
    observed_index = observed[1]
    if device_type in {"cpu", "mps"}:
        return expected_index in {None, 0} and observed_index in {None, 0}
    return expected_index is None or observed_index == expected_index


def validate_torch_device(requested: str) -> str:
    """Return an available canonical Torch device, failing before inference."""

    if not isinstance(requested, str) or not requested.strip():
        raise ValueError("device must be a non-empty PyTorch device string")
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Torch device validation requires the optional eval dependency") from exc
    try:
        device = torch.device(requested)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"invalid PyTorch device {requested!r}") from exc
    if device.type == "cpu":
        if device.index is not None:
            raise ValueError("CPU device must not specify an index")
    elif device.type == "mps":
        if device.index is not None:
            raise ValueError("MPS device must not specify an index")
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise RuntimeError("requested MPS device is not available in this runtime")
    elif device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested CUDA device is not available in this runtime")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(f"requested CUDA device index {index} is not available")
        device = torch.device("cuda", index)
    else:
        raise ValueError(f"unsupported PyTorch device type {device.type!r}")
    try:
        torch.empty(1, device=device)
    except (RuntimeError, TypeError) as exc:
        raise RuntimeError(f"requested PyTorch device {device} cannot allocate a tensor") from exc
    return str(device)


def verify_actual_torch_device(requested: str, actual: object) -> str:
    """Fail on requested/observed execution-device drift."""

    expected = validate_torch_device(requested)
    if not torch_device_attestation_matches(expected, actual):
        raise RuntimeError(
            f"detector executed on {actual}, but the requested device was {expected}"
        )
    parsed = _parse_torch_device_spelling(actual)
    assert parsed is not None
    return parsed[0] if parsed[0] in {"cpu", "mps"} else str(actual)


def deterministic_inference_execution_contract(
    seed: int = DEFAULT_INFERENCE_SEED,
) -> dict[str, Any]:
    """Return the frozen seed and backend controls for detector inference."""

    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError("inference seed must be an integer")
    normalized_seed = int(seed)
    if not 0 <= normalized_seed <= MAX_SEED:
        raise ValueError(f"inference seed must be in [0, {MAX_SEED}]")
    return {
        "implementation_id": INFERENCE_EXECUTION_METHOD,
        "seed": normalized_seed,
        "seed_scope": "python_numpy_torch_before_first_detector_inference",
        "numpy_seed_modulus": 2**32,
        "torch_deterministic_algorithms": True,
        "torch_deterministic_debug_mode": "error",
        "torch_float32_matmul_precision": "highest",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cublas_workspace_config": ":4096:8",
        "mps_fallback_allowed": False,
        "mps_fast_math": False,
        "mps_prefer_metal": False,
    }


def _attest_deterministic_inference_execution(
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(expected, Mapping):
        raise TypeError("expected inference execution contract must be a mapping")
    normalized = json_value(expected)
    canonical = deterministic_inference_execution_contract(normalized.get("seed"))
    if normalized != canonical:
        raise ValueError("inference execution contract is unsupported or noncanonical")
    try:
        import torch
    except ImportError as exc:
        raise ImportError("deterministic detector execution requires Torch") from exc
    checks = {
        "torch_initial_seed": int(torch.initial_seed()) == canonical["seed"],
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "torch_deterministic_debug_mode": int(torch.get_deterministic_debug_mode()) == 2,
        "torch_float32_matmul_precision": (
            torch.get_float32_matmul_precision() == canonical["torch_float32_matmul_precision"]
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark) == canonical["cudnn_benchmark"],
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic)
        == canonical["cudnn_deterministic"],
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32)
        == canonical["cuda_matmul_allow_tf32"],
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32) == canonical["cudnn_allow_tf32"],
        "cublas_workspace_config": (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") == canonical["cublas_workspace_config"]
        ),
        "mps_fallback_allowed": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0")
        not in {"1", "true", "TRUE", "yes", "YES"},
        "mps_fast_math": (
            os.environ.get("PYTORCH_MPS_FAST_MATH") == ("1" if canonical["mps_fast_math"] else "0")
        ),
        "mps_prefer_metal": (
            os.environ.get("PYTORCH_MPS_PREFER_METAL")
            == ("1" if canonical["mps_prefer_metal"] else "0")
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "deterministic detector execution attestation failed: " + ", ".join(failed)
        )
    return canonical


def configure_deterministic_inference(
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seed Python/NumPy/Torch and enable fail-closed deterministic kernels."""

    import random

    import numpy as np

    contract = (
        deterministic_inference_execution_contract()
        if expected is None
        else deterministic_inference_execution_contract(expected.get("seed"))
    )
    if expected is not None and json_value(expected) != contract:
        raise ValueError("inference execution contract is unsupported or noncanonical")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = contract["cublas_workspace_config"]
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["PYTORCH_MPS_FAST_MATH"] = "0"
    os.environ["PYTORCH_MPS_PREFER_METAL"] = "0"
    try:
        import torch
    except ImportError as exc:
        raise ImportError("deterministic detector execution requires Torch") from exc
    seed = int(contract["seed"])
    random.seed(seed)
    np.random.seed(seed % int(contract["numpy_seed_modulus"]))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_deterministic_debug_mode("error")
    torch.set_float32_matmul_precision(contract["torch_float32_matmul_precision"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return _attest_deterministic_inference_execution(contract)


def _physical_memory_bytes() -> int | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    result = page_size * page_count
    return result if result > 0 else None


def _macos_sysctl(name: str) -> str | None:
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            env={"LANG": "C", "LC_ALL": "C"},
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.decode("utf-8", errors="strict").strip()
    return value or None


def _hardware_identity() -> dict[str, Any]:
    """Return non-secret CPU, memory, accelerator, and Torch build facts."""

    try:
        import torch
    except ImportError:
        torch_record: dict[str, Any] | None = None
    else:
        cuda_devices = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                cuda_devices.append(
                    {
                        "index": index,
                        "name": properties.name,
                        "total_memory_bytes": int(properties.total_memory),
                        "compute_capability": [int(properties.major), int(properties.minor)],
                    }
                )
        torch_record = {
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "intraop_thread_count": int(torch.get_num_threads()),
            "interop_thread_count": int(torch.get_num_interop_threads()),
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_devices": cuda_devices,
            "build_config_sha256": hashlib.sha256(
                torch.__config__.show().encode("utf-8")
            ).hexdigest(),
        }
    return {
        "logical_cpu_count": os.cpu_count(),
        "physical_memory_bytes": _physical_memory_bytes(),
        "platform_processor": platform.processor() or None,
        "macos_hardware_model": _macos_sysctl("hw.model"),
        "macos_cpu_brand": _macos_sysctl("machdep.cpu.brand_string"),
        "macos_physical_cpu_count": _macos_sysctl("hw.physicalcpu"),
        "torch": torch_record,
    }


def _runtime_environment_payload(
    repository_root: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    return {
        "run_identity": result_run_identity(),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "operating_system": {
            "name": os.name,
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(aliased=False, terse=False),
            "machine": platform.machine(),
        },
        "hardware": _hardware_identity(),
        "packages": _runtime_package_versions(),
        "git": _git_repository_identity(repository_root),
    }


def analysis_runtime_reproducibility_identity(
    *,
    repository_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Bind the source, package, process, and hardware state used for statistics."""

    payload = {
        "schema_version": 2,
        "record_type": "analysis_runtime_reproducibility_identity",
        "method": ANALYSIS_RUNTIME_IDENTITY_METHOD,
        **_runtime_environment_payload(repository_root),
    }
    normalized = json_value(payload)
    return {**normalized, "runtime_identity_sha256": canonical_sha256(normalized)}


def runtime_reproducibility_identity(
    *,
    requested_detector_device: str,
    actual_detector_device: object,
    repository_root: str | os.PathLike[str] | None = None,
    expected_inference_execution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the bounded runtime and source state of a detector run.

    Git paths, diffs, untracked filenames, and file contents are consumed only
    by the aggregate worktree hash and are never embedded in the returned
    record. Ignored files are excluded so datasets, weights, caches, and local
    secrets covered by repository ignore rules do not enter the identity.
    """

    requested = _nonempty(requested_detector_device, label="requested_detector_device")
    resolved_requested = validate_torch_device(requested)
    resolved_actual = verify_actual_torch_device(resolved_requested, actual_detector_device)
    reported_actual = str(actual_detector_device)
    parsed_actual = _parse_torch_device_spelling(reported_actual)
    if parsed_actual is None:
        raise RuntimeError("detector reported an invalid execution device")
    if parsed_actual[0] == "cuda" and parsed_actual[1] is not None:
        resolved_actual = f"cuda:{parsed_actual[1]}"
    inference_execution = (
        None
        if expected_inference_execution is None
        else _attest_deterministic_inference_execution(expected_inference_execution)
    )

    payload: dict[str, Any] = {
        "schema_version": 2,
        "record_type": "runtime_reproducibility_identity",
        "method": RUNTIME_IDENTITY_METHOD,
        **_runtime_environment_payload(repository_root),
        "detector_device": {
            "requested": requested,
            "resolved_requested": resolved_requested,
            "actual": reported_actual,
            "resolved_actual": resolved_actual,
            "attestation": {
                "method": "torch_requested_actual_device_match.v1",
                "matches": True,
            },
        },
        "inference_execution": inference_execution,
    }
    normalized = json_value(payload)
    return {**normalized, "runtime_identity_sha256": canonical_sha256(normalized)}


def ordered_integer_selection_identity(
    values: Iterable[int], *, selection_kind: str
) -> dict[str, object]:
    """Return the historical compact identity of an ordered integer axis.

    This six-field shape is embedded in schema-v2 COCO/study records and is
    therefore immutable.  It commits to the complete ordered values through
    ``selection_sha256`` but does not serialize them a second time.  New
    standalone result manifests that must recompute an axis without external
    context use :func:`embedded_integer_selection_identity` instead.
    """

    kind = _nonempty(selection_kind, label="selection_kind")
    raw = list(values)
    if not raw:
        raise ValueError(f"{kind} selection must contain at least one value")
    if any(not isinstance(value, numbers.Integral) or isinstance(value, bool) for value in raw):
        raise TypeError(f"{kind} selection must contain only integers")
    normalized = [int(value) for value in raw]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{kind} selection must not contain duplicates")
    payload = {
        "kind": kind,
        "ordered_values": normalized,
        "encoding": "repository canonical bytes v1",
    }
    return {
        "kind": kind,
        "count": len(normalized),
        "first": normalized[0],
        "last": normalized[-1],
        "selection_sha256": canonical_sha256(payload),
        "encoding": payload["encoding"],
    }


def embedded_integer_selection_identity(
    values: Iterable[int], *, selection_kind: str
) -> dict[str, object]:
    """Serialize and content-address an integer axis for standalone records."""

    materialized = list(values)
    compact = ordered_integer_selection_identity(
        materialized,
        selection_kind=selection_kind,
    )
    normalized = [int(value) for value in materialized]
    payload = {
        "schema_version": 1,
        "record_type": "embedded_ordered_integer_selection",
        **compact,
        "ordered_values": normalized,
    }
    return {**payload, "record_sha256": canonical_sha256(payload)}


def validate_embedded_integer_selection_identity(
    record: Mapping[str, Any],
    *,
    selection_kind: str,
) -> dict[str, object]:
    """Recompute a versioned embedded integer selection from its values."""

    if not isinstance(record, Mapping):
        raise TypeError("selection identity must be a mapping")
    normalized = json_value(record)
    expected = embedded_integer_selection_identity(
        normalized.get("ordered_values", ()),
        selection_kind=selection_kind,
    )
    if normalized != expected:
        raise ValueError(f"{selection_kind} selection identity does not match ordered values")
    return expected


def source_content_identity(content: Mapping[str, Any]) -> dict[str, Any]:
    """Content-address a portable description of one exact source image.

    ``content`` normally embeds a raw-file artifact record (name, byte count,
    and SHA-256), decoded shape, and decode contract.  The helper remains
    format-neutral so scientific arrays can use an exact array-byte record.
    """

    if not isinstance(content, Mapping):
        raise TypeError("source content must be a mapping")
    payload = {
        "schema_version": 3,
        "record_type": "evaluation_source_content",
        "content": json_value(content),
    }
    return {**payload, "source_sha256": canonical_sha256(payload)}


def validate_source_content_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute one embedded source-content identity."""

    if not isinstance(record, Mapping):
        raise TypeError("source identity must be a mapping")
    normalized = json_value(record)
    expected = source_content_identity(normalized.get("content"))
    if normalized != expected:
        raise ValueError("source_sha256 does not match embedded source content")
    return expected


def target_annotation_identity(content: Mapping[str, Any]) -> dict[str, Any]:
    """Content-address annotations/targets associated with one source image."""

    if not isinstance(content, Mapping):
        raise TypeError("target annotation content must be a mapping")
    payload = {
        "schema_version": 3,
        "record_type": "evaluation_target_annotations",
        "content": json_value(content),
    }
    return {**payload, "target_sha256": canonical_sha256(payload)}


def validate_target_annotation_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute one embedded annotation/target identity."""

    if not isinstance(record, Mapping):
        raise TypeError("target identity must be a mapping")
    normalized = json_value(record)
    expected = target_annotation_identity(normalized.get("content"))
    if normalized != expected:
        raise ValueError("target_sha256 does not match embedded target annotations")
    return expected


def evaluation_source_record(
    *,
    image_id: int,
    source_identity: Mapping[str, Any],
    target_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a dataset integer key to exact image and target content."""

    if isinstance(image_id, bool) or not isinstance(image_id, numbers.Integral):
        raise TypeError("image_id must be an integer")
    payload = {
        "schema_version": 3,
        "record_type": "evaluation_source_record",
        "image_id": int(image_id),
        "source_identity": validate_source_content_identity(source_identity),
        "target_identity": validate_target_annotation_identity(target_identity),
    }
    return {**payload, "source_record_sha256": canonical_sha256(payload)}


def validate_evaluation_source_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a complete per-image source/target record."""

    if not isinstance(record, Mapping):
        raise TypeError("evaluation source record must be a mapping")
    normalized = json_value(record)
    expected = evaluation_source_record(
        image_id=normalized.get("image_id"),
        source_identity=normalized.get("source_identity"),
        target_identity=normalized.get("target_identity"),
    )
    if normalized != expected:
        raise ValueError("source_record_sha256 does not match embedded source record")
    return expected


def ordered_source_selection_identity(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Content-address an ordered source axis including image and target bytes."""

    if isinstance(records, (Mapping, str, bytes, bytearray, set, frozenset)):
        raise TypeError("source selection requires an explicitly ordered iterable")
    normalized = [validate_evaluation_source_record(record) for record in records]
    if not normalized:
        raise ValueError("evaluation_source selection must contain at least one value")
    image_ids = [record["image_id"] for record in normalized]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("evaluation_source selection must not contain duplicate image IDs")
    payload = {
        "kind": "evaluation_source",
        "ordered_values": normalized,
        "encoding": "repository canonical bytes v1",
    }
    return {
        "kind": payload["kind"],
        "count": len(normalized),
        "first": normalized[0]["source_record_sha256"],
        "last": normalized[-1]["source_record_sha256"],
        "ordered_values": normalized,
        "selection_sha256": canonical_sha256(payload),
        "encoding": payload["encoding"],
    }


def validate_ordered_source_selection_identity(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute a serialized source selection from all embedded records."""

    if not isinstance(record, Mapping):
        raise TypeError("source selection identity must be a mapping")
    normalized = json_value(record)
    expected = ordered_source_selection_identity(normalized.get("ordered_values", ()))
    if normalized != expected:
        raise ValueError("source selection identity does not match ordered source records")
    return expected


def image_selection_identity(image_ids: Iterable[int]) -> dict[str, object]:
    """Return a portable identity for the actual ordered image selection."""

    return ordered_integer_selection_identity(image_ids, selection_kind="image_id")


def realization_selection_identity(realization_ids: Iterable[int]) -> dict[str, object]:
    """Bind the ordered stochastic realization IDs used at every condition."""

    return ordered_integer_selection_identity(realization_ids, selection_kind="realization_id")


def condition_identity(
    *,
    family: str,
    coordinate_name: str,
    coordinate_unit: str,
    coordinate_value: float,
    baseline_type: str,
    fixed_profile_sha256: str,
    observed_severity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a one-knob condition record and its canonical content identity."""

    profile_hash = _nonempty(fixed_profile_sha256, label="fixed_profile_sha256")
    if len(profile_hash) != 64 or any(ch not in "0123456789abcdef" for ch in profile_hash):
        raise ValueError("fixed_profile_sha256 must be a lowercase SHA-256 digest")
    baseline = _nonempty(baseline_type, label="baseline_type")
    if baseline not in {"untouched_input", "modeled_neutral"}:
        raise ValueError("baseline_type must be 'untouched_input' or 'modeled_neutral'")
    payload: dict[str, Any] = {
        "family": _nonempty(family, label="family"),
        "coordinate": {
            "name": _nonempty(coordinate_name, label="coordinate_name"),
            "unit": _nonempty(coordinate_unit, label="coordinate_unit"),
            "value": _finite(coordinate_value, label="coordinate_value"),
        },
        "baseline_type": baseline,
        "fixed_profile_sha256": profile_hash,
        "observed_severity": None if observed_severity is None else json_value(observed_severity),
    }
    return {**payload, "condition_sha256": canonical_sha256(payload)}


def parse_float_grid(
    text: str,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> tuple[float, ...]:
    """Parse an ordered, unique, finite comma-separated coordinate grid."""

    name = _nonempty(label, label="label")
    if not isinstance(text, str):
        raise TypeError(f"{name} must be a comma-separated string")
    tokens = [token.strip() for token in text.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError(f"{name} must be a non-empty comma-separated grid")
    try:
        values = tuple(float(token) for token in tokens)
    except ValueError as exc:
        raise ValueError(f"{name} must contain only finite numbers") from exc
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite numbers")
    normalized = tuple(0.0 if value == 0.0 else value for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate values")
    for value in normalized:
        if minimum is not None:
            valid = value >= minimum if minimum_inclusive else value > minimum
            if not valid:
                operator = ">=" if minimum_inclusive else ">"
                raise ValueError(f"{name} values must be {operator} {minimum:g}")
        if maximum is not None:
            valid = value <= maximum if maximum_inclusive else value < maximum
            if not valid:
                operator = "<=" if maximum_inclusive else "<"
                raise ValueError(f"{name} values must be {operator} {maximum:g}")
    return normalized


def parse_int_grid(text: str, *, label: str, minimum: int | None = None) -> tuple[int, ...]:
    """Parse canonical base-10 integer spellings without lossy coercion."""

    name = _nonempty(label, label="label")
    if not isinstance(text, str):
        raise TypeError(f"{name} must be a comma-separated string")
    tokens = [token.strip() for token in text.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError(f"{name} must be a non-empty comma-separated grid")
    try:
        values = tuple(int(token) for token in tokens)
    except ValueError as exc:
        raise ValueError(f"{name} must contain only integers") from exc
    if any(str(value) != token and f"+{value}" != token for token, value in zip(tokens, values)):
        raise ValueError(f"{name} must contain canonical base-10 integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate values")
    if minimum is not None and any(value < minimum for value in values):
        raise ValueError(f"{name} values must be >= {minimum}")
    return values


def validate_common_args(
    parser: ArgumentParser,
    args: Namespace,
    *,
    minimum_bootstrap_iters: int = 2,
) -> None:
    """Reject invalid run sizes, seeds, and devices before expensive setup."""

    for name in ("max_images", "image_size"):
        if hasattr(args, name) and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if hasattr(args, "image_offset") and args.image_offset < 0:
        parser.error("--image-offset must be non-negative")
    if hasattr(args, "bootstrap_iters") and args.bootstrap_iters < minimum_bootstrap_iters:
        parser.error(f"--bootstrap-iters must be at least {minimum_bootstrap_iters}")
    for name in ("bootstrap_seed", "noise_seed"):
        if not hasattr(args, name):
            continue
        value = getattr(args, name)
        if not isinstance(value, numbers.Integral) or isinstance(value, bool):
            parser.error(f"--{name.replace('_', '-')} must be an integer")
        if not 0 <= int(value) <= MAX_SEED:
            parser.error(f"--{name.replace('_', '-')} must be in [0, {MAX_SEED}]")
    if hasattr(args, "device"):
        try:
            args.device = validate_torch_device(args.device)
        except (ImportError, TypeError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))


def preprocessing_identity(
    *, name: str, implementation_id: str, parameters: Mapping[str, Any]
) -> dict[str, Any]:
    """Content-address detector preprocessing separately from the camera graph."""

    if not isinstance(parameters, Mapping):
        raise TypeError("preprocessing parameters must be a mapping")
    payload = {
        "name": _nonempty(name, label="preprocessing name"),
        "implementation_id": _nonempty(implementation_id, label="preprocessing implementation_id"),
        "parameters": json_value(parameters),
    }
    return {**payload, "preprocessing_sha256": canonical_sha256(payload)}


def require_positive_finite(value: object, *, label: str) -> float:
    """Validate a denominator without accepting booleans or masked zeros."""

    try:
        result = _finite(value, label=label)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc
    if result <= 0.0:
        raise RuntimeError(f"{label} must be finite and positive; got {value!r}")
    return result


def require_probability_metric(value: object, *, label: str) -> float:
    """Return a finite metric in ``(0, 1]`` suitable as a denominator."""

    result = require_positive_finite(value, label=label)
    if result > 1.0:
        raise RuntimeError(f"{label} must be at most 1; got {value!r}")
    return result


def camera_stage_graph_signature(
    record: Mapping[str, Any],
) -> tuple[tuple[Any, ...], ...]:
    """Validate and return the complete executable camera-stage signature.

    Domain/unit continuity alone is insufficient: changing a stage name,
    deterministic declaration, implementation ID, or neutral semantics also
    changes what was executed.  This signature deliberately includes every
    serialized :class:`~phycam_eval.pipeline.StageSpec` field.
    """

    if not isinstance(record, Mapping):
        raise TypeError("camera provenance must be a mapping")
    try:
        mode = DataMode(record.get("data_mode"))
    except (TypeError, ValueError) as exc:
        raise ValueError("camera provenance has an unsupported data_mode") from exc
    graph = record.get("stage_graph")
    if not isinstance(graph, Sequence) or isinstance(graph, (str, bytes, bytearray)):
        raise ValueError("camera provenance requires an embedded stage_graph array")
    if not graph:
        raise ValueError("camera stage_graph must not be empty")
    expected_fields = {
        "name",
        "input_domain",
        "output_domain",
        "input_units",
        "output_units",
        "deterministic",
        "implementation_id",
        "neutral_condition",
    }
    signature: list[tuple[Any, ...]] = []
    names: set[str] = set()
    previous_domain: Domain | None = None
    previous_units: str | None = None
    for index, stage in enumerate(graph):
        if not isinstance(stage, Mapping):
            raise ValueError("every stage_graph entry must be a mapping")
        if set(stage) != expected_fields:
            raise ValueError(f"camera stage {index} lacks its complete StageSpec contract")
        name = _nonempty(stage.get("name"), label=f"camera stage {index} name")
        implementation = _nonempty(
            stage.get("implementation_id"),
            label=f"camera stage {index} implementation_id",
        )
        input_units = _nonempty(
            stage.get("input_units"),
            label=f"camera stage {index} input_units",
        )
        output_units = _nonempty(
            stage.get("output_units"),
            label=f"camera stage {index} output_units",
        )
        if name in names:
            raise ValueError("camera stage names must be unique")
        names.add(name)
        try:
            source = Domain(stage.get("input_domain"))
            target = Domain(stage.get("output_domain"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"camera stage {index} has an unsupported domain") from exc
        if not is_legal_transition(mode, source, target):
            raise ValueError(f"camera stage {index} has an illegal domain transition")
        if previous_domain is not None and (
            source is not previous_domain or input_units != previous_units
        ):
            raise ValueError(f"camera stage {index} drifted from the previous boundary")
        deterministic = stage.get("deterministic")
        if not isinstance(deterministic, bool):
            raise ValueError(f"camera stage {index} deterministic flag must be bool")
        neutral = stage.get("neutral_condition")
        if neutral is not None:
            neutral = _nonempty(
                neutral,
                label=f"camera stage {index} neutral_condition",
            )
        signature.append(
            (
                name,
                source.value,
                target.value,
                input_units,
                output_units,
                deterministic,
                implementation,
                neutral,
            )
        )
        previous_domain = target
        previous_units = output_units
    allowed_starts = (
        {Domain.DISPLAY_RGB.value}
        if mode is DataMode.LDR_REDEGRADATION
        else {
            Domain.SCENE_SPECTRAL.value,
            Domain.OVERSAMPLED_SCENE_LINEAR_WITH_DECLARED_SPECTRAL_ADAPTER.value,
        }
    )
    if signature[0][1] not in allowed_starts:
        raise ValueError("camera stage graph starts at an incomplete mode topology")
    if signature[-1][2] != Domain.DISPLAY_RGB.value:
        raise ValueError("camera stage graph must terminate in DISPLAY_RGB")
    input_frame = record.get("input_frame")
    if input_frame is not None:
        if not isinstance(input_frame, Mapping):
            raise ValueError("camera input_frame must be a mapping or null")
        if input_frame.get("domain") != signature[0][1]:
            raise ValueError("camera input_frame domain drifted from the first stage")
        metadata = input_frame.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("units") != signature[0][3]:
            raise ValueError("camera input_frame units drifted from the first stage")
        if metadata.get("data_mode") != mode.value:
            raise ValueError("camera input_frame data mode drifted from provenance")
    output_frame = record.get("output_frame")
    if output_frame is not None:
        if not isinstance(output_frame, Mapping):
            raise ValueError("camera output_frame must be a mapping or null")
        if output_frame.get("domain") != signature[-1][2]:
            raise ValueError("camera output_frame domain drifted from the final stage")
        metadata = output_frame.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("units") != signature[-1][4]:
            raise ValueError("camera output_frame units drifted from the final stage")
        if metadata.get("data_mode") != mode.value:
            raise ValueError("camera output_frame data mode drifted from provenance")
    return tuple(signature)


def validate_camera_provenance(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a complete :class:`CameraPipeline` record.

    Hashes are recomputed from embedded content.  A schema-v1 record or a
    record whose profile, data mode, or stage graph drifted fails closed.
    """

    if not isinstance(record, Mapping):
        raise TypeError("camera provenance must be a mapping")
    if record.get("schema_version") != 2:
        raise ValueError("camera provenance requires schema_version 2")
    profile = record.get("camera_profile")
    if not isinstance(profile, Mapping):
        raise ValueError("camera provenance requires an embedded camera_profile")
    if profile.get("schema_version") != 2:
        raise ValueError("embedded camera_profile requires schema_version 2")
    graph = record.get("stage_graph")
    signature = camera_stage_graph_signature(record)
    expected_profile_hash = canonical_sha256(profile)
    if record.get("camera_profile_sha256") != expected_profile_hash:
        raise ValueError("camera_profile_sha256 does not match embedded profile")
    expected_graph_hash = canonical_sha256(graph)
    if record.get("stage_graph_sha256") != expected_graph_hash:
        raise ValueError("stage_graph_sha256 does not match embedded stage graph")
    if record.get("data_mode") != profile.get("data_mode"):
        raise ValueError("camera provenance data_mode does not match camera profile")
    if not isinstance(record.get("deterministic"), bool):
        raise ValueError("camera provenance deterministic flag must be bool")
    if record.get("deterministic") != all(item[5] for item in signature):
        raise ValueError("camera provenance deterministic flag disagrees with stage graph")
    capture_condition = record.get("capture_condition")
    capture_hash = record.get("capture_condition_sha256")
    if (capture_condition is None) != (capture_hash is None):
        raise ValueError("camera capture condition and hash must be declared together")
    if capture_condition is not None:
        if not isinstance(capture_condition, Mapping):
            raise ValueError("camera capture_condition must be a mapping")
        if capture_hash != canonical_sha256(capture_condition):
            raise ValueError("capture_condition_sha256 does not match embedded capture condition")
    renderer_condition = record.get("renderer_capture_condition")
    renderer_hash = record.get("renderer_capture_condition_sha256")
    if (renderer_condition is None) != (renderer_hash is None):
        raise ValueError("renderer capture condition and hash must be declared together")
    if renderer_condition is not None:
        if not isinstance(renderer_condition, Mapping):
            raise ValueError("renderer_capture_condition must be a mapping")
        if renderer_hash != canonical_sha256(renderer_condition):
            raise ValueError(
                "renderer_capture_condition_sha256 does not match embedded renderer condition"
            )
    return json_value(record)


def validate_result_manifest(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a schema-v3 evaluation envelope and its optional result hash."""

    if not isinstance(record, Mapping):
        raise TypeError("result manifest must be a mapping")
    if record.get("schema_version") != 3:
        raise ValueError("result manifest requires schema_version 3")
    required_mappings = (
        "camera_provenance",
        "condition",
        "detector_execution",
        "image_selection",
        "model",
        "preprocessing",
        "realization_selection",
        "run_identity",
    )
    for key in required_mappings:
        if not isinstance(record.get(key), Mapping):
            raise ValueError(f"result manifest requires mapping field {key!r}")
    validate_camera_provenance(record["camera_provenance"])
    model = validate_model_identity(record["model"])
    validate_detector_execution_identity(record["detector_execution"], model=model)
    preprocessing = record["preprocessing"]
    preprocessing_payload = {
        key: value for key, value in preprocessing.items() if key != "preprocessing_sha256"
    }
    if preprocessing.get("preprocessing_sha256") != canonical_sha256(preprocessing_payload):
        raise ValueError("preprocessing_sha256 does not match embedded preprocessing contract")
    condition = record["condition"]
    condition_payload = {
        key: value for key, value in condition.items() if key != "condition_sha256"
    }
    if condition.get("condition_sha256") != canonical_sha256(condition_payload):
        raise ValueError("condition_sha256 does not match embedded condition")
    if record["camera_provenance"].get("capture_condition_sha256") != condition.get(
        "condition_sha256"
    ):
        raise ValueError(
            "camera capture_condition_sha256 does not match selected manifest condition"
        )
    profile_hash = record["camera_provenance"]["camera_profile_sha256"]
    if condition.get("fixed_profile_sha256") != profile_hash:
        raise ValueError("condition is bound to a different camera profile")
    image_selection = record.get("image_selection")
    if not isinstance(image_selection, Mapping):
        raise ValueError("image_selection is missing its selection identity")
    validate_ordered_source_selection_identity(image_selection)
    validate_embedded_integer_selection_identity(
        record["realization_selection"],
        selection_kind="realization_id",
    )
    supplied_hash = record.get("result_sha256")
    normalized = json_value(record)
    if supplied_hash is not None:
        payload = {key: value for key, value in normalized.items() if key != "result_sha256"}
        if supplied_hash != canonical_sha256(payload):
            raise ValueError("result_sha256 does not match result manifest")
    return normalized


def attach_result_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a manifest and return a copy carrying ``result_sha256``."""

    payload = {key: value for key, value in record.items() if key != "result_sha256"}
    normalized = validate_result_manifest(payload)
    return {**normalized, "result_sha256": canonical_sha256(normalized)}


__all__ = [
    "ANALYSIS_RUNTIME_IDENTITY_METHOD",
    "DEFAULT_INFERENCE_SEED",
    "DEFAULT_RUNTIME_DISTRIBUTIONS",
    "INFERENCE_EXECUTION_METHOD",
    "MAX_SEED",
    "PACKAGE_IDENTITY_METHOD",
    "PROJECT_INSTALLATION_IDENTITY_METHOD",
    "RUNTIME_IDENTITY_METHOD",
    "WORKTREE_STATE_METHOD",
    "attach_result_identity",
    "analysis_runtime_reproducibility_identity",
    "camera_stage_graph_signature",
    "condition_identity",
    "configure_deterministic_inference",
    "deterministic_inference_execution_contract",
    "embedded_integer_selection_identity",
    "evaluation_source_record",
    "image_selection_identity",
    "installed_distribution_universe",
    "ordered_integer_selection_identity",
    "ordered_source_selection_identity",
    "parse_float_grid",
    "parse_int_grid",
    "preprocessing_identity",
    "project_installation_identity",
    "realization_selection_identity",
    "require_positive_finite",
    "require_probability_metric",
    "result_run_identity",
    "runtime_reproducibility_identity",
    "source_content_identity",
    "target_annotation_identity",
    "torch_device_attestation_matches",
    "validate_common_args",
    "validate_camera_provenance",
    "validate_evaluation_source_record",
    "validate_embedded_integer_selection_identity",
    "validate_ordered_source_selection_identity",
    "validate_result_manifest",
    "validate_source_content_identity",
    "validate_target_annotation_identity",
    "validate_project_installation_identity",
    "validate_torch_device",
    "verify_actual_torch_device",
]
