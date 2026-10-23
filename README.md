# tiny-hll - Zero-Dependency HyperLogLog

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)](tiny_hll.py)
[![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/hussain-alsaibai/tiny-hll)

Count distinct items in **fixed, tiny memory** - about 1.6KB by default,
regardless of whether you have seen a thousand items or a billion.

A HyperLogLog answers "roughly how many unique users did we see?" without
storing the users. That is the whole trade: you get a number within ~1-9%,
and you cannot list what you saw or remove anything.

## When to use this

| Use a HyperLogLog | Use a set |
|---|---|
| You need "about how many distinct?" | You need exact answers |
| The cardinality is huge (millions+) | You need to list members |
| Memory is capped or expensive | You need to delete items |
| A few percent error is acceptable | Zero error is required |
| Dashboards, DAU/WAU/MAU, unique visitors | Dedup lists, IP blocks, permission sets |

## Install

    pip install tiny-hll

Or copy `tiny_hll.py` - one file, standard library only.

## Usage

    from tiny_hll import from_iterable

    # One-liner over any iterable
    hll = from_iterable("user-%d" % i for i in range(100_000))
    print(hll.approximate_count())   # ~99_900  (true: 100_000)
    print(hll.memory_bytes)          # 128 - never grows

    # Or add items as they stream in
    from tiny_hll import HyperLogLog

    hll = HyperLogLog(error_rate=0.01)
    for user_id in request_log:
        hll.add(user_id)
    print(hll.count())

## Merging - the reason to use HLL at all

Two sketches merge into the union of what they saw, which is how you get a
global count from per-shard partials without shipping raw data:

    daily = from_iterable(shard_1_ids)
    daily.merge(from_iterable(shard_2_ids))

    # weekly = union of the 7 daily sketches
    weekly = daily.copy()
    for day in last_seven_days:
        weekly.merge(from_iterable(day_ids))

    print(weekly.approximate_count())

Both sketches must use the same `error_rate`; merging across precisions
raises `ValueError` rather than silently returning a wrong number.

## Choosing precision

Memory is `2**p` bytes, and the relative standard error is `1.04 / sqrt(2**p)`.

| error_rate | registers | memory | realised std error |
|-------------|-----------|--------|-------------------|
| 5% (0.05)  | 32        | 32 B   | 18.4%             |
| 2% (0.02)  | 64        | 64 B   | 13.0%             |
| **1% (0.01)** | **128** | **128 B** | **9.2%**      |
| 0.5%        | 256       | 256 B  | 6.5%              |
| 0.1% (0.001)| 1024      | 1 KB   | 3.25%             |
| 0.01%       | 16384     | 16 KB  | 0.81%             |

**Read the last column carefully: the realised spread is much larger than the
nominal `error_rate` you passed.** Asking for `error_rate=0.01` gives a p=7
sketch whose actual standard deviation is ~9%, not 1%. The argument sets the
*precision*, which only ever guarantees about `1.04/sqrt(2**p)`.

If you want a true 1% standard error, ask for roughly `error_rate=0.002`.

## Serialization

A sketch is a few hundred bytes, so shipping one between workers is far
cheaper than shipping the underlying data:

    hll.save("dau-2026-10-23.hll")
    restored = HyperLogLog.load("dau-2026-10-23.hll")
    assert restored.count() == hll.count()

## Measured accuracy

Estimator bias and spread over 12 independent sketches of 50,000 items each,
measured on this machine (CPython 3.11, AMD EPYC 9354P):

| registers | bias | measured std | theoretical 1.04/sqrt(m) |
|-----------|------|--------------|--------------------------|
| 32        | -1.3% | 14.6%        | 18.4%                    |
| 64        | +1.7% | 7.3%         | 13.0%                    |
| 128       | +0.5% | 6.2%         | 9.2%                     |
| 256       | -0.9% | 3.7%         | 6.5%                     |
| 1024      | -0.7% | 3.4%         | 3.3%                     |
| 16384     | +0.6% | 0.67%        | 0.81%                    |

Bias stays under 2.5% at every precision, so the estimator is unbiased;
measured spread tracks the theoretical figure as expected.

## Caveats

- **Approximate.** Never use it for billing, quotas, or anything where an
  error is a correctness bug.
- **No removal.** There is no `discard()`. If a day's sketch looks inflated
  by a bot flood, you throw it away and rebuild - you cannot subtract.
- **No members.** It cannot tell you *which* items it saw.
- **Numeric types canonicalize.** `42`, `"42"`, and `b"42"` count as the same
  item, so an identifier arriving as an int from one code path and a str from
  another does not double-count.
- **Merge requires matching precision.**

## How it works

Each item is hashed to 64 bits. The top `p` bits select one of `2**p`
registers; the number of leading zeros in the remaining bits updates that
register to the maximum seen. The estimate is the harmonic mean of
`2**-register` scaled by the HLL constant.

Two corrections are applied:

- **Small range** (estimate below `2.5m`): linear counting,
  `m * ln(m / zeros)`, which is far more accurate than the harmonic mean when
  most registers are still empty.
- **Large range** (estimate above `2^32 / 30`): the 64-bit correction for
  register saturation.

Hashing is FNV-1a followed by the splitmix64 finalizer. The finalizer is not
decoration: FNV-1a's lowest bits are weakly distributed and the register index
is read from exactly those bits.

## Tests

    python -m pytest test_tiny_hll.py -v

## License

MIT

**Part of the [tiny-* ecosystem](https://github.com/hussain-alsaibai).**
