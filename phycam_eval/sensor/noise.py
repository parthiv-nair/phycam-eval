"""Stateless electron-domain shot, dark, and read-noise reference."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtri
from scipy.stats import poisson as poisson_distribution

_NOISE_NAMESPACE_DOMAIN = b"phycam-eval/noise-namespace/v1"
_OPEN_UNIFORM_BITS = 52
_OPEN_UNIFORM_SHIFT = 64 - _OPEN_UNIFORM_BITS
_OPEN_UNIFORM_HALF_STEP = np.float64(2.0**-53)


def _length_prefixed(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return len(encoded).to_bytes(8, "big") + encoded


def _derived_noise_source(parent: str, child: str) -> str:
    """Return a domain-separated digest of the complete stream namespace."""

    digest = sha256()
    digest.update(_NOISE_NAMESPACE_DOMAIN)
    digest.update(_length_prefixed(parent))
    digest.update(_length_prefixed(child))
    return f"derived-sha256-v1:{digest.hexdigest()}"


@dataclass(frozen=True, slots=True)
class RNGKey:
    """Complete stochastic condition identity, excluding photosite counter."""

    profile_hash: str
    seed: int
    image_id: str
    coupling_id: str
    realization: int
    noise_source: str

    def __post_init__(self) -> None:
        for name in ("profile_hash", "image_id", "coupling_id", "noise_source"):
            value = str(getattr(self, name))
            if not value:
                raise ValueError(f"{name} must be nonempty")
            object.__setattr__(self, name, value)
        for name in ("seed", "realization"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            value = int(value)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)

    def for_source(self, noise_source: str) -> RNGKey:
        """Derive a child stream while preserving the caller's namespace.

        The parent and child strings are length-prefixed and hashed with an
        explicit domain tag.  This avoids delimiter ambiguity and, critically,
        prevents internal names such as ``"shot"`` from replacing the base
        namespace supplied by the caller.  Equal derivation paths are stable;
        different parent namespaces or child labels produce independent keys
        up to SHA-256 collision resistance.
        """

        child = str(noise_source)
        if not child:
            raise ValueError("noise_source must be nonempty")
        return replace(self, noise_source=_derived_noise_source(self.noise_source, child))

    def digest_seed(self) -> np.uint64:
        digest = sha256()
        for value in (
            self.profile_hash,
            str(self.seed),
            self.image_id,
            self.coupling_id,
            str(self.realization),
            self.noise_source,
        ):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return np.uint64(int.from_bytes(digest.digest()[:8], "little"))


class StatelessRNG:
    """SplitMix64 counter generator with inverse-CDF distribution transforms.

    The algorithm is intentionally simple and fully indexed: each sample is a
    pure function of the serialized key and its global photosite counter.
    Generating an image in different worker partitions therefore gives the
    exact same result.
    """

    algorithm = "splitmix64_top52_open_midpoint_inverse_cdf_v2"

    def uniform(
        self,
        shape: tuple[int, ...],
        key: RNGKey,
        *,
        counters: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        if any(int(size) < 0 for size in shape):
            raise ValueError("shape dimensions must be nonnegative")
        count = int(np.prod(shape, dtype=np.int64))
        counter_array = _counters(count, shape, counters)
        raw = _splitmix64(counter_array, key.digest_seed())
        return _uint64_to_open_uniform(raw).reshape(shape)

    def normal(
        self,
        mean: ArrayLike,
        sigma: ArrayLike,
        key: RNGKey,
        *,
        counters: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        mean_array, sigma_array = np.broadcast_arrays(
            np.asarray(mean, dtype=np.float64), np.asarray(sigma, dtype=np.float64)
        )
        if not np.all(np.isfinite(mean_array)):
            raise ValueError("normal mean must be finite")
        if not np.all(np.isfinite(sigma_array)) or np.any(sigma_array < 0.0):
            raise ValueError("normal sigma must be finite and nonnegative")
        uniform = self.uniform(mean_array.shape, key, counters=counters)
        return mean_array + sigma_array * ndtri(uniform)

    def poisson(
        self,
        expectation: ArrayLike,
        key: RNGKey,
        *,
        counters: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        expectation_array = np.asarray(expectation, dtype=np.float64)
        if expectation_array.size == 0:
            return np.empty_like(expectation_array)
        if not np.all(np.isfinite(expectation_array)) or np.any(expectation_array < 0.0):
            raise ValueError("Poisson expectation must be finite and nonnegative")
        uniform = self.uniform(expectation_array.shape, key, counters=counters)
        samples = poisson_distribution.ppf(uniform, expectation_array)
        samples = np.where(expectation_array == 0.0, 0.0, samples)
        if not np.all(np.isfinite(samples)):
            raise RuntimeError("Poisson inverse CDF produced nonfinite output")
        return samples.astype(np.float64, copy=False)


@dataclass(frozen=True, slots=True)
class ElectronCapture:
    photoelectrons: NDArray[np.float64]
    dark_electrons: NDArray[np.float64]
    well_electrons: NDArray[np.float64]
    read_noise_electrons: NDArray[np.float64]
    electrons: NDArray[np.float64]


def capture_electrons(
    expected_photoelectrons: ArrayLike,
    *,
    full_well_electrons: float,
    read_noise_electrons: ArrayLike,
    expected_dark_electrons: ArrayLike | float = 0.0,
    rng: StatelessRNG,
    key: RNGKey,
    counters: ArrayLike | None = None,
) -> ElectronCapture:
    """Apply exact shot/dark Poisson, full well, then input-referred read noise."""

    photo_expectation = np.asarray(expected_photoelectrons, dtype=np.float64)
    dark_expectation = np.broadcast_to(
        np.asarray(expected_dark_electrons, dtype=np.float64), photo_expectation.shape
    )
    read_sigma = np.broadcast_to(
        np.asarray(read_noise_electrons, dtype=np.float64), photo_expectation.shape
    )
    full_well = float(full_well_electrons)
    if photo_expectation.size == 0:
        raise ValueError("expected_photoelectrons must be nonempty")
    if not np.all(np.isfinite(photo_expectation)) or np.any(photo_expectation < 0.0):
        raise ValueError("photoelectron expectation must be finite and nonnegative")
    if not np.all(np.isfinite(dark_expectation)) or np.any(dark_expectation < 0.0):
        raise ValueError("dark expectation must be finite and nonnegative")
    if not np.all(np.isfinite(read_sigma)) or np.any(read_sigma < 0.0):
        raise ValueError("read noise must be finite and nonnegative")
    if not np.isfinite(full_well) or full_well <= 0.0:
        raise ValueError("full_well_electrons must be finite and positive")

    photo = rng.poisson(photo_expectation, key.for_source("shot"), counters=counters)
    dark = rng.poisson(dark_expectation, key.for_source("dark"), counters=counters)
    well = np.minimum(photo + dark, full_well)
    read = rng.normal(
        np.zeros_like(photo_expectation),
        read_sigma,
        key.for_source("read"),
        counters=counters,
    )
    electrons = well + read
    return ElectronCapture(photo, dark, well, read, electrons)


def _counters(count: int, shape: tuple[int, ...], counters: ArrayLike | None) -> NDArray[np.uint64]:
    if counters is None:
        return np.arange(count, dtype=np.uint64).reshape(shape)
    array = np.asarray(counters)
    if array.shape != shape:
        if array.size != count:
            raise ValueError("counters must have the requested shape or size")
        array = array.reshape(shape)
    if not np.issubdtype(array.dtype, np.integer) or np.any(array < 0):
        raise ValueError("counters must be nonnegative integers")
    return array.astype(np.uint64, copy=False)


def _splitmix64(counters: NDArray[np.uint64], seed: np.uint64) -> NDArray[np.uint64]:
    with np.errstate(over="ignore"):
        values = counters + seed + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return values ^ (values >> np.uint64(31))


def _uint64_to_open_uniform(raw: ArrayLike) -> NDArray[np.float64]:
    """Map uint64 words to exact binary64 midpoints strictly inside ``(0, 1)``.

    The top 52 random bits choose one of ``2**52`` equal-width bins and the
    returned value is that bin's exact midpoint.  Using 52 rather than 53 bits
    is intentional: every midpoint, including the uppermost ``1 - 2**-53``,
    is representable in binary64.  A top-53 half-step formula rounds its
    mathematical upper midpoint to exactly ``1.0`` under ties-to-even.
    """

    words = np.asarray(raw, dtype=np.uint64)
    bins = words >> np.uint64(_OPEN_UNIFORM_SHIFT)
    return np.ldexp(bins.astype(np.float64), -_OPEN_UNIFORM_BITS) + _OPEN_UNIFORM_HALF_STEP
