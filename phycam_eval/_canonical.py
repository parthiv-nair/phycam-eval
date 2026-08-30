"""Canonical value freezing and byte encoding used by profile identity."""

from __future__ import annotations

import hashlib
import math
import struct
import unicodedata
from collections.abc import Mapping, Sequence
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np


def nfc_string(value: str, *, field_name: str = "string") -> str:
    """Validate and normalize a string to Unicode NFC."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return unicodedata.normalize("NFC", value)


def finite_float(value: object, *, field_name: str) -> float:
    """Return a finite binary64 value, normalizing negative zero."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field_name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be a real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return 0.0 if result == 0.0 else result


def positive_float(value: object, *, field_name: str, allow_zero: bool = False) -> float:
    """Return a finite positive (or nonnegative) binary64 value."""

    result = finite_float(value, field_name=field_name)
    if result < 0.0 or (result == 0.0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{field_name} must be {qualifier}")
    return result


def positive_int(value: object, *, field_name: str, allow_zero: bool = False) -> int:
    """Validate an integer without accepting booleans or lossy coercions."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field_name} must be an integer")
    result = int(value)
    if result < 0 or (result == 0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{field_name} must be {qualifier}")
    return result


def _normalized_mapping_items(value: Mapping[Any, Any]) -> list[tuple[str, Any]]:
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("canonical mappings require string keys")
        normalized_key = nfc_string(key, field_name="mapping key")
        if normalized_key in normalized:
            raise ValueError("mapping contains keys that collide after Unicode NFC normalization")
        normalized[normalized_key] = item
    return sorted(normalized.items(), key=lambda pair: pair[0].encode("utf-8"))


def freeze_json_value(value: Any) -> Any:
    """Defensively freeze a JSON-like value into immutable normalized objects.

    NumPy arrays/scalars are accepted for ergonomic profile construction, but
    are copied into tuples/Python scalars.  Set-like collections are rejected
    because they do not have a declared array order.
    """

    if isinstance(value, Enum):
        return freeze_json_value(value.value)
    if value is None or isinstance(value, (bool, np.bool_)):
        return None if value is None else bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return finite_float(value, field_name="mapping value")
    if isinstance(value, str):
        return nfc_string(value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays are not valid profile values")
        return freeze_json_value(value.tolist())
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: freeze_json_value(item) for key, item in _normalized_mapping_items(value)}
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        raise TypeError("unordered collections are not valid profile values")
    raise TypeError(f"unsupported profile value type: {type(value).__name__}")


def json_value(value: Any) -> Any:
    """Convert frozen profile values into ordinary JSON-compatible values."""

    if isinstance(value, Enum):
        return json_value(value.value)
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float):
            return finite_float(value, field_name="JSON float")
        return value
    if isinstance(value, Mapping):
        return {key: json_value(item) for key, item in _normalized_mapping_items(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_value(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return json_value(to_dict())
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def _encode(value: Any) -> bytes:
    """Encode using the repository's self-delimiting canonical grammar.

    Grammar (ASCII tags, UTF-8 string payloads)::

        null       n;
        boolean    b0; | b1;
        integer    i<base-10>;
        binary64   f<16 lowercase IEEE-754 hex digits>;
        string     s<UTF-8-byte-count>:<UTF-8 bytes>
        array      a<item-count>[<items>]
        mapping    m<pair-count>{<string-key><value>...}

    Mapping keys are NFC-normalized and sorted by UTF-8 bytes.  Arrays retain
    their declared order.  Type tags and length prefixes make collisions
    between, for example, a float and a string containing its hex value
    impossible.
    """

    if isinstance(value, Enum):
        return _encode(value.value)
    if value is None:
        return b"n;"
    if isinstance(value, (bool, np.bool_)):
        return b"b1;" if bool(value) else b"b0;"
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return b"i" + str(int(value)).encode("ascii") + b";"
    if isinstance(value, (float, np.floating)):
        number = finite_float(value, field_name="canonical float")
        bits = struct.unpack(">Q", struct.pack(">d", number))[0]
        return b"f" + f"{bits:016x}".encode("ascii") + b";"
    if isinstance(value, str):
        payload = nfc_string(value).encode("utf-8")
        return b"s" + str(len(payload)).encode("ascii") + b":" + payload
    if isinstance(value, Mapping):
        items = _normalized_mapping_items(value)
        payload = b"".join(_encode(key) + _encode(item) for key, item in items)
        return b"m" + str(len(items)).encode("ascii") + b"{" + payload + b"}"
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays cannot be canonically encoded")
        return _encode(value.tolist())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        payload = b"".join(_encode(item) for item in value)
        return b"a" + str(len(value)).encode("ascii") + b"[" + payload + b"]"
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _encode(to_dict())
    raise TypeError(f"unsupported canonical value type: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    """Return deterministic, language-independent identity bytes for ``value``."""

    return _encode(value)


def canonical_sha256(value: Any) -> str:
    """Return the lowercase SHA-256 hex digest of canonical identity bytes."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()
