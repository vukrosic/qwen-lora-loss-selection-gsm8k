# Reproduction notes

## Evidence-only verification

The packaged evidence requires only Python 3.12 standard-library modules:

```bash
python analyze_results.py
shasum -a 256 -c MANIFEST.sha256
```

`analyze_results.py` rejects changed protocol/data hashes, incomplete receipts,
wrong update counts, missing per-record evidence, adapter/selection mismatches,
test-before-selection ordering, nonfinite values, and failed informativeness
gates before regenerating `RESULT.json`.

## Full local rerun

The measured environment was Python 3.12.8, MLX 0.31.2, mlx-lm 0.31.3, and
NumPy 2.2.5 on an M4/16GB Mac. Supply the base model read-only from a local
path such as:

```text
<path-to-Qwen3-0.6B-3bit>
```

Install compatible versions from `requirements.txt`, inspect current memory
pressure and competing model processes, and run exactly one heavy process at a
time. The precise command array and working directory for every completed stage
are stored in its `runs/*/receipt.json`.

The required scientific order for each seed is:

1. train both arms for exactly 576 updates in the order fixed by `PROTOCOL.md`;
2. evaluate both on `valid` with the common token-weighted completion NLL;
3. run `select_by_validation.py` and preserve the resulting immutable
   `runs/selection-<seed>.json`;
4. only then evaluate both arms on `test` with that selection record supplied;
5. after all three pairs, run `analyze_results.py`.

Do not use training-log validation values for selection. Do not change data,
seeds, tie logic, endpoints, gates, record order, generation cap, or update
count after viewing final-test evidence.
