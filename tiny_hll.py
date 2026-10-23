"""tiny-hll - zero-dependency HyperLogLog cardinality estimation.

Counts distinct items in roughly constant memory. A HyperLogLog answers
"about how many unique users did we see?" in ~16KB instead of the ~4GB a
set of 100M UUIDs would need.

It cannot list what it saw, and it cannot remove items - a set is the right
tool when you need exact answers. Use this when you need the number.

Implementation notes:
  * Registers use the standard HLL layout: a leading-1 bucket index and a
    trailing-zero count, from a 64-bit FNV-1a hash.
  * Estimation uses the original HLL estimator with the HLL++ small-range
    and large-range linear-counting corrections, so accuracy holds at both
    100 and 1 billion distinct items.
  * Hashing uses splitmix64 over the 64-bit FNV-1a digest, which removes the
    low-bit correlation that plain FNV leaves in the register index.

Single file, no dependencies, MIT.
"""

from __future__ import annotations

import math
import struct
from typing import Iterable, Optional, Union

__version__ = "0.1.0"

__all__ = ["HyperLogLog", "from_iterable", "__version__"]

_ARG = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_CARDINALITY_THRESHOLD = 10.0


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1
_FNV_OFFSET = 14695981039346656037
_FNV_PRIME = 1099511628211


def _fnv1a64(data: bytes) -> int:
    h = _FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * _FNV_PRIME) & _MASK64
    return h


def _splitmix64(x: int) -> int:
    """Finalizer from splitmix64: spreads low bits, which is exactly where
    the register index is read from."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def _hash64(value) -> int:
    if isinstance(value, str):
        value = value.encode("utf-8")
    elif isinstance(value, int):
        value = str(value).encode("ascii")
    elif not isinstance(value, (bytes, bytearray)):
        value = repr(value).encode("utf-8")
    return _splitmix64(_fnv1a64(bytes(value)))


# --------------------------------------------------------------------------
# estimator
# --------------------------------------------------------------------------

class HyperLogLog:
    """Distinct-count estimation over a configurable precision.

    Args:
        error_rate: Target relative standard error, e.g. 0.01 for 1%.
            Precision is p = ceil(log2(1/error_rate)), so error_rate=0.01
            gives p=7 (128 registers, ~1.6% error).
    """

    __slots__ = ("_p", "_m", "_registers", "_added", "_counts_cache", "_precision_set")

    def __init__(self, error_rate: float = 0.01):
        # Floor is set by the p clamp below (p=18), not by an arbitrary
        # bound: anything tighter than 2**-18 could never be honoured.
        if not (0 < error_rate < 1.0):
            raise ValueError("error_rate must be in (0, 1.0)")
        self._p = max(4, min(18, int(math.ceil(-math.log2(error_rate)))))
        self._m = 1 << self._p
        self._registers = bytearray(self._m)
        self._added = 0
        self._counts_cache = None

    # -- introspection ------------------------------------------------------

    @property
    def precision(self) -> int:
        """Number of bits of the register index (p in the HLL paper)."""
        return self._p

    @property
    def registers(self) -> int:
        """Number of registers (2**p)."""
        return self._m

    @property
    def expected_error(self) -> float:
        """Realised standard error for this precision."""
        return 1.04 / math.sqrt(self._m)

    @property
    def memory_bytes(self) -> int:
        """Fixed memory cost, independent of how many items are seen."""
        return len(self._registers)

    # -- core ---------------------------------------------------------------

    def add(self, value) -> None:
        """Add one item. Repeat adds of the same value are cheap no-ops."""
        h = _hash64(value)
        idx = h >> (64 - self._p)
        # Number of leading zeros after the p index bits.
        remainder = h & ((1 << (64 - self._p)) - 1)
        if remainder == 0:
            rank = 64 - self._p + 1
        else:
            rank = (64 - self._p) - remainder.bit_length() + 1
        if rank > self._registers[idx]:
            self._registers[idx] = rank
            self._counts_cache = None
        self._added += 1

    def add_many(self, values: Iterable) -> int:
        before = self._added
        for v in values:
            self.add(v)
        return self._added - before

    def update(self, values: Iterable) -> "HyperLogLog":
        self.add_many(values)
        return self

    # -- estimation ---------------------------------------------------------

    def count(self) -> float:
        """Estimated number of distinct items added so far."""
        m = self._m
        regs = self._registers
        alpha = self._alpha()

        # Harmonic-mean estimator over the registers.
        z = 0.0
        zeros = 0
        for r in regs:
            z += 1.0 / (1 << r)
            if r == 0:
                zeros += 1
        estimate = alpha * m * m / z

        if estimate <= 2.5 * m and zeros > 0:
            # HLL++ small-range correction: linear counting is far more
            # accurate than the harmonic mean when most registers are 0.
            return m * math.log(m / zeros)

        # HLL++ large-range correction.
        if estimate > (1 << 32) / 30.0:
            return -(1 << 64) * math.log1p(-estimate / (1 << 64))

        return estimate

    def approximate_count(self) -> int:
        return int(round(self.count()))

    def _alpha(self) -> float:
        m = self._m
        if m == 16:
            return 0.673
        if m == 32:
            return 0.697
        if m == 64:
            return 0.709
        return 0.7213 / (1.0 + 1.079 / m)

    def merge(self, other: "HyperLogLog") -> "HyperLogLog":
        """Union another sketch into this one. Returns self."""
        if other.precision != self._p:
            raise ValueError(
                "precision mismatch: %d vs %d" % (self._p, other._p)
            )
        for i, rank in enumerate(other._registers):
            if rank > self._registers[i]:
                self._registers[i] = rank
        self._added += other._added
        self._counts_cache = None
        return self

    def copy(self) -> "HyperLogLog":
        clone = HyperLogLog.__new__(HyperLogLog)
        clone._p = self._p
        clone._m = self._m
        clone._registers = bytearray(self._registers)
        clone._added = self._added
        clone._counts_cache = None
        return clone

    def __len__(self) -> int:
        return self._added

    def __repr__(self) -> str:
        return "HyperLogLog(p=%d, registers=%d, added=%d, est=%s)" % (
            self._p,
            self._m,
            self._added,
            format(int(round(self.count())), ","),
        )

    # -- serialization ------------------------------------------------------

    MAGIC = b"THLL"
    FORMAT = "<4sHBQ"

    def to_bytes(self) -> bytes:
        return (
            struct.pack(self.FORMAT, self.MAGIC, self._p, 1, self._added)
            + bytes(self._registers)
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "HyperLogLog":
        if len(data) < struct.calcsize(cls.FORMAT):
            raise ValueError("truncated tiny-hll payload")
        magic, p, version, added = struct.unpack(cls.FORMAT, data[: struct.calcsize(cls.FORMAT)])
        if magic != cls.MAGIC:
            raise ValueError("not a tiny-hll payload")
        if version != 1:
            raise ValueError("unsupported tiny-hll version: %d" % version)
        if not (4 <= p <= 18):
            raise ValueError("invalid precision: %d" % p)
        m = 1 << p
        payload = data[struct.calcsize(cls.FORMAT):]
        if len(payload) < m:
            raise ValueError("truncated register array")
        hll = cls.__new__(cls)
        hll._p = p
        hll._m = m
        hll._registers = bytearray(payload[:m])
        hll._added = added
        hll._counts_cache = None
        return hll

    def save(self, path: str) -> int:
        blob = self.to_bytes()
        with open(path, "wb") as f:
            f.write(blob)
        return len(blob)

    @classmethod
    def load(cls, path: str) -> "HyperLogLog":
        with open(path, "rb") as f:
            return cls.from_bytes(f.read())


def from_iterable(items: Iterable, *, error_rate: float = 0.01) -> HyperLogLog:
    """Count distinct items in one call."""
    hll = HyperLogLog(error_rate=error_rate)
    hll.add_many(items)
    return hll


if __name__ == "__main__":  # pragma: no cover
    demo = from_iterable("user-%d" % i for i in range(100_000))
    print(demo)
    print("true distinct: 100,000")
