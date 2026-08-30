"""Fail-closed advisory locks for official study and publication transactions."""

from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _lock_path(target: str | Path, *, purpose: str) -> Path:
    path = Path(target)
    if not path.name:
        raise ValueError("lock target must have a final path component")
    return path.parent / f".{path.name}.{purpose}.lock"


@contextmanager
def advisory_target_lock(
    target: str | Path,
    *,
    purpose: str,
    exclusive: bool,
) -> Iterator[Path]:
    """Lock a sibling inode without following a pre-existing symbolic link.

    Official detector workers take a shared ``study-layout`` lock, while the
    analyzer/promoter takes it exclusively. Publication writers use an
    exclusive ``publication`` lock. The sibling location keeps lock metadata
    outside the exact artifact sets being validated.
    """

    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - publication CI is POSIX
        raise RuntimeError("official study publication requires POSIX flock support") from exc

    lock_path = _lock_path(target, purpose=purpose)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise RuntimeError(f"failed to open safe {purpose} lock: {lock_path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError(f"{purpose} lock is not a private regular file: {lock_path}")
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(descriptor, operation)
        # Recheck after acquisition so a concurrently replaced lock pathname
        # cannot split cooperating processes across different inodes.
        path_metadata = os.lstat(lock_path)
        if (
            not stat.S_ISREG(path_metadata.st_mode)
            or path_metadata.st_nlink != 1
            or (path_metadata.st_dev, path_metadata.st_ino) != (metadata.st_dev, metadata.st_ino)
        ):
            raise RuntimeError(f"{purpose} lock pathname changed during acquisition")
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


__all__ = ["advisory_target_lock"]
