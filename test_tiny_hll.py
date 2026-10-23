"""Tests for tiny-hll.

Run: python -m pytest test_tiny_hll.py -v
"""
from __future__ import annotations

import math
import os
import tempfile

import pytest

from tiny_hll import HyperLogLog, from_iterable, __version__


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------

def test_version():
    assert __version__ == "0.1.0"


def test_precision_from_error_rate():
    assert HyperLogLog(0.01).precision == 7       # ceil(log2(100))
    assert HyperLogLog(0.001).precision == 10
    assert HyperLogLog(0.10).precision == 4       # ceil(log2(10))
    assert HyperLogLog(0.001).registers == 1024


def test_precision_is_clamped_at_both_ends():
    """A 0.4 error rate cannot be met by p=2, so p is floored at 4."""
    assert HyperLogLog(0.4).precision == 4
    # p = ceil(log2(1/error_rate)): 1e-4 -> 14 exactly, in range.
    assert HyperLogLog(1e-4).precision == 14
    # 1e-9 would imply p=30, far beyond the supported range, so it caps at 18.
    assert HyperLogLog(1e-9).precision == 18
    assert HyperLogLog(1e-9).registers == 262144


def test_register_count_and_memory():
    hll = HyperLogLog(0.01)
    assert hll.registers == 128
    assert hll.memory_bytes == 128
    # Memory must not grow with the number of items.
    hll.add_many(range(100_000))
    assert hll.memory_bytes == 128


def test_expected_error_matches_precision():
    for err in (0.01, 0.005, 0.001):
        hll = HyperLogLog(err)
        assert hll.expected_error == pytest.approx(1.04 / math.sqrt(hll.registers))


def test_invalid_error_rate():
    for bad in (0.0, -0.1, 1.0, 2.0, -1.0):
        with pytest.raises(ValueError):
            HyperLogLog(bad)


# --------------------------------------------------------------------------
# accuracy
# --------------------------------------------------------------------------

def test_empty_is_zero():
    hll = HyperLogLog()
    assert hll.count() == 0.0
    assert hll.approximate_count() == 0


def test_small_cardinality_is_accurate():
    """Small-range linear counting must beat the harmonic mean here."""
    for n in (10, 100, 1_000):
        hll = from_iterable(range(n), error_rate=0.01)
        est = hll.count()
        se = 1.04 / math.sqrt(hll.registers)
        assert abs(est - n) <= 4 * se * n, (n, est)


def test_accuracy_within_error_bounds():
    """Estimate must fall inside 4 standard errors at several scales."""
    for n in (1_000, 10_000, 100_000):
        hll = from_iterable(("u-%d" % i for i in range(n)), error_rate=0.01)
        est = hll.count()
        se = 1.04 / math.sqrt(hll.registers)
        assert abs(est - n) <= 4 * se * n, (n, est, se)


def test_accuracy_improves_with_precision():
    """Higher precision must yield a smaller error rate on average.

    A single sample cannot prove ordering, so average over 5 independent
    sketches per precision.
    """
    import statistics

    errs = []
    for rate in (0.05, 0.01, 0.001):
        samples = []
        for trial in range(8):
            # Distinct item sets per trial: identical items would give every
            # sketch the same hashes and correlate the samples.
            items = ("%d-%d" % (trial, i) for i in range(50_000))
            samples.append(abs(from_iterable(items, error_rate=rate).count() - 50_000))
        errs.append(statistics.mean(samples))
    assert errs[0] > errs[1] > errs[2], errs


def test_duplicates_do_not_inflate():
    hll = from_iterable(["a"] * 10_000, error_rate=0.01)
    assert abs(hll.count() - 1) / 1 < 0.05


# --------------------------------------------------------------------------
# input types
# --------------------------------------------------------------------------

def test_accepts_str_bytes_int_mixed():
    hll = HyperLogLog()
    hll.add("user-1")
    hll.add(b"user-1")        # str and bytes hash identically by design
    hll.add(42)
    hll.add(42)
    # "user-1" and b"user-1" are treated as the same item; 42 dedupes to one.
    assert hll.approximate_count() == 2


def test_int_and_str_of_same_number_are_the_same_item():
    """Documented behaviour: numeric types encode to the same bytes.

    This keeps a count from drifting when the same identifier arrives as an
    int from one code path and a str from another.
    """
    hll = HyperLogLog()
    hll.add(42)
    hll.add("42")
    assert hll.approximate_count() == 1


def test_mixed_iterable():
    hll = from_iterable(["a", b"a", 1, 1, "1", 1.0, True, "a"])
    # "a" == b"a"; 1 and "1" canonicalize to the same item; 1.0 and True are
    # distinct encodings. Distinct items: "a", 1, 1.0, True = 4.
    assert hll.approximate_count() == 4


def test_add_many_returns_delta():
    hll = HyperLogLog()
    assert hll.add_many(range(100)) == 100
    assert len(hll) == 100


def test_update_is_chainable():
    hll = HyperLogLog().update(range(10))
    assert isinstance(hll, HyperLogLog)
    assert len(hll) == 10
    assert abs(hll.count() - 10) / 10 < 0.15


# --------------------------------------------------------------------------
# merge
# --------------------------------------------------------------------------

def test_merge_unions_counts():
    a = from_iterable(range(1_000), error_rate=0.01)
    b = from_iterable(range(500, 1_500), error_rate=0.01)
    a.merge(b)
    est = a.count()
    se = 1.04 / math.sqrt(a.registers)
    assert abs(est - 1_500) <= 4 * se * 1_500, est


def test_merge_of_disjoint_sets():
    a = from_iterable(["x%d" % i for i in range(1_000)], error_rate=0.01)
    b = from_iterable(["y%d" % i for i in range(1_000)], error_rate=0.01)
    a.merge(b)
    se = 1.04 / math.sqrt(a.registers)
    assert abs(a.count() - 2_000) <= 4 * se * 2_000


def test_merge_precision_mismatch_rejected():
    with pytest.raises(ValueError):
        HyperLogLog(0.01).merge(HyperLogLog(0.001))


def test_copy_is_independent():
    a = from_iterable(range(1_000), error_rate=0.01)
    b = a.copy()
    b.add_many(range(2_000, 3_000))
    assert b.approximate_count() != a.approximate_count()


# --------------------------------------------------------------------------
# serialization
# --------------------------------------------------------------------------

def test_roundtrip_bytes():
    hll = from_iterable(range(5_000), error_rate=0.01)
    clone = HyperLogLog.from_bytes(hll.to_bytes())
    assert clone.precision == hll.precision
    assert clone.registers == hll.registers
    assert clone.count() == pytest.approx(hll.count())
    assert clone.approximate_count() == hll.approximate_count()


def test_roundtrip_file():
    hll = from_iterable(("item-%d" % i for i in range(3_000)), error_rate=0.01)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".hll") as f:
        path = f.name
    try:
        written = hll.save(path)
        clone = HyperLogLog.load(path)
        assert written > 0
        assert clone.count() == pytest.approx(hll.count())
    finally:
        os.unlink(path)


def test_rejects_garbage_payload():
    with pytest.raises(ValueError):
        HyperLogLog.from_bytes(b"not-a-hll-file-at-all-really-no")


def test_rejects_truncated_payload():
    hll = from_iterable(range(100))
    blob = hll.to_bytes()
    with pytest.raises(ValueError):
        HyperLogLog.from_bytes(blob[:10])


def test_file_is_compact():
    """The whole point: fixed size, tiny."""
    hll = from_iterable(range(1_000_000), error_rate=0.01)
    size = len(hll.to_bytes())
    import struct as _struct
    header = _struct.calcsize(HyperLogLog.FORMAT)
    assert size == header + 128
    assert size < 200               # vs ~60MB for an exact set of 1M ints
