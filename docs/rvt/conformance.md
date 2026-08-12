# OpenEB Conformance

`tests/test_openeb_conformance.py` checks that evlib's EVT2 and EVT3 binary decoders produce the same event stream as the reference decoders in [OpenEB](https://github.com/prophesee-ai/openeb), Prophesee's own open-source SDK. It is a decode-correctness check, separate from the RVT preprocessing benchmark on the [RVT Pipeline](index.md) page: this harness only checks that the raw `(x, y, polarity, t)` tuples evlib reads out of a `.raw` file match what OpenEB reads out of the same file.

## What "byte-identical decode" means

Both decoders are reduced to the same canonical form before comparison, defined in `tests/conformance/canonical.py`:

1. Every event is a 4-tuple `(x, y, pol, t)` with `pol` in `{0, 1}` and `t` an integer microsecond timestamp.
2. The events are sorted into a total order by `(t, x, y, pol)`, so timestamp ties do not make the comparison order-dependent.
3. The sorted events are packed into a fixed-width little-endian byte stream (`x:u16, y:u16, pol:u8, t:i64`, no padding) and hashed with SHA-256.

"Byte-identical decode" means this SHA-256 digest matches between evlib's decode and the reference decoder's decode: every event, in the same canonical order, with the same coordinates, polarity and timestamp. The digest also records `n_events`, the sensor `geometry`, and the first and last 16 events (`head`/`tail`) in canonical order, so a mismatch can be localised without re-running both decoders.

`evlib`'s side of the comparison is `tests/conformance/evlib_stream.py:evlib_digest`, which calls `evlib.load_events(path, sort=False)` and maps evlib's `-1`/`+1` polarity to the canonical `0`/`1` encoding before hashing.

## Committed reference digests

The reference digests live under `tests/conformance/reference/`, one JSON file per sample (`80_balls.evt2.json`, `val_night_011.evt2.json`, `pedestrians.evt3.json`, `sample_data.aedat4.json`, `test-minimal.aedat4.json`). Each file holds the canonical digest (`n_events`, `geometry`, `stream_sha256`, `head`, `tail`) plus provenance: for EVT2/EVT3 samples, the OpenEB decoder name and the `lib/openeb` commit the digest was generated against; for the AEDAT4 samples, the `dv_processing` oracle used instead (OpenEB does not decode AEDAT4).

These files are committed so the conformance test can run anywhere without needing a local OpenEB checkout, a C++ compiler, or the reference decoders at test time. `tests/test_openeb_conformance.py` only reads the committed JSON and decodes the sample with evlib; it never builds or runs OpenEB.

The following block computes the same canonical digest against the tracked `80_balls.raw` EVT2 sample, to show the structure of what gets compared:

```python
from tests.conformance.evlib_stream import evlib_digest

digest = evlib_digest("data/prophesee/samples/evt2/80_balls.raw")
print(f"n_events: {digest['n_events']}")
print(f"geometry: {digest['geometry']}")
print(f"stream_sha256: {digest['stream_sha256'][:16]}...")
print(f"first event (x, y, pol, t): {digest['head'][0]}")
```

## Regenerating the digests

The digests are local-only to generate: `tests/conformance/generate_reference.py` compiles and runs the OpenEB standalone sample decoders (and, for AEDAT4, `dv_processing`) against a local `lib/openeb` checkout, so it needs that checkout and a C++ compiler, and it is never run in CI.

```bash
# Full diff against one sample: decode both sides, sort canonically, and report
# the first event index where evlib and the reference disagree.
.venv/bin/python tests/conformance/generate_reference.py --verify 80_balls.evt2

# Regenerate and overwrite every committed digest.
.venv/bin/python tests/conformance/generate_reference.py --all --update
```

`--verify KEY` decodes both sides in full and stops at the first mismatched event, which is the tool to reach for when `test_evlib_matches_openeb_reference` fails and the test's own `head`/`tail` diff is not enough to explain why. `--all --update` re-runs every sample in `tests/conformance/canonical.py:SAMPLES` and rewrites its reference JSON; `--sample KEY [--update]` does the same for one sample. Without `--update`, `generate_reference.py` refuses to overwrite a digest whose hash has changed, so a silent regression can't slip into the committed reference by accident.

## What runs in CI versus locally

`tests/test_openeb_conformance.py` is parametrised over every entry in `canonical.SAMPLES`, and each case skips itself with `pytest.skip` when its sample file is absent, rather than failing:

| Sample | Format | Path | Tracked in git? | Runs in CI? |
|---|---|---|---|---|
| `80_balls.evt2` | EVT2 | `data/prophesee/samples/evt2/80_balls.raw` | yes | yes |
| `val_night_011.evt2` | EVT2 | `tests/data/eTram/raw/val_2/val_night_011.raw` | yes | yes |
| `pedestrians.evt3` | EVT3 | `data/prophesee/samples/evt3/pedestrians.raw` | no (gitignored) | no, skips |
| `sample_data.aedat4` | AEDAT4 | `lib/dv-processing/tests/io/test_files/sample_data.aedat4` | no (gitignored) | no, skips |
| `test-minimal.aedat4` | AEDAT4 | `lib/dv-processing/tests/io/test_files/test-minimal.aedat4` | no (gitignored) | no, skips |

The two EVT2 samples are tracked, so the gate always runs in CI. The EVT3 sample is gitignored, so its case skips in CI and only runs on a machine that has the file locally; the same applies to the two AEDAT4 samples, which live under the gitignored `lib/dv-processing` checkout. This keeps the CI conformance gate real (it always exercises the tracked EVT2 path) without requiring every contributor, or every CI runner, to carry the larger EVT3 and AEDAT4 fixtures.
