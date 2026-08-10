# GSM8K model-free data-lock proposal

**Owner:** natural-dataset-literature-audit  
**Date:** 2026-08-10 (Asia/Singapore)  
**Status:** `PROPOSAL / MODEL-FREE / NOT CONFIRMATORY`  
**Scope:** reconcile the original GSM8K official-split recommendation with Protocol N1's leakage-safe split requirements before any Qwen/MLX run.

This proposal does not load Qwen, MLX, or a tokenizer. It retrieves and audits only the small public GSM8K source, writes immutable candidate split manifests inside this project folder, and leaves exact Qwen chat-template lengths as a required model-slot preflight.

## Locked source and deterministic retrieval

The source is the OpenAI `grade-school-math` repository at the immutable commit:

```text
repository: https://github.com/openai/grade-school-math
commit:     3101c7d5072418e28b9008a6636bde82a006892c
dataset:    openai/grade-school-math, config main
raw base:   https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c
```

`lock_gsm8k.py` fetches by commit, never by a moving branch, and verifies every downloaded byte against the expected SHA-256 before using it. The pinned files are stored under `source/gsm8k-3101c7d/`:

| file | source path | bytes | SHA-256 |
|---|---|---:|---|
| `train.jsonl` | `grade_school_math/data/train.jsonl` | 4,166,206 | `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465` |
| `test.jsonl` | `grade_school_math/data/test.jsonl` | 749,738 | `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` |
| `README.md` | `README.md` | 6,112 | `27e1676d572f2289210170ff6dbb8c2ef7d9f6950c0f08c978ea8ee6b39dca29` |
| `LICENSE` | `LICENSE` | 1,062 | `86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b` |

The combined source-version manifest hash is:

```text
bc62143914da6723031303e6a6790f0ec7601226e617fe2b7a0fcdca82468d29
```

The retrieval, canonicalization, deduplication, and split implementation is [lock_gsm8k.py](./lock_gsm8k.py), SHA-256:

```text
29408107bd19cb2d13d526bf1ee0cc5c3b325572d96a9423024025377fcb7c4e
```

The generated machine-readable receipt is [LOCK-PROPOSAL.json](./lock/LOCK-PROPOSAL.json). Re-running the script after a source change must produce a new versioned lock; it must not silently overwrite a confirmatory run.

## Canonical record and group rules

The two pinned JSONL files contain 7,473 official `train` records and 1,319 official `test` records, 8,792 total. Each record is retained only if `question` and `answer` are non-empty. The lock preserves the original strings and records provenance:

```text
id = gsm8k/main/{source_split}/{source_line:06d}
task_family = gsm8k.main
```

For deduplication only, the script applies Unicode NFKC, converts CRLF/CR to LF, collapses all Unicode whitespace runs to one ASCII space, and trims. The original prompt and completion are never replaced by these canonical forms.

The exact duplicate key is SHA-256 of canonical `{prompt, completion}` JSON with sorted keys and stable separators. Near duplicates are connected components of token 5-shingle Jaccard at least 0.90. The audit found:

| check | result |
|---|---:|
| retained records | 8,792 |
| empty required records | 0 |
| exact duplicate groups | 0 |
| near-duplicate candidate pairs checked | 93,642 |
| near-duplicate pairs at Jaccard >= 0.90 | 0 |
| multi-record components | 0 |
| indivisible groups | 8,792 singleton groups |

Because no group crosses the original source boundary, preserving the official test partition is possible without splitting a dedup/near-duplicate component. This is an observation about this pinned revision only; it must be recomputed if the source changes.

The model-free length stratum is a deterministic proxy: rank records globally by canonical completion whitespace-token count, then assign four equal rank quartiles using source split and line as tie-breakers. The proxy is not a Qwen token count. For the generated candidate test partitions, the whitespace-token p90/p10 ratios are 3.87 (`n1-global`) and 3.87 (`official-test`), above N1's 1.5 minimum; the exact Qwen-token ratio remains a preflight gate.

## Two viable split options

### Option A — strict N1 global split (fallback if no amendment is accepted)

Apply N1's hash seed `20260810` and 70/15/15 allocation to the full 8,792-record union after dedup/group checks. Within the single declared task family and each length quartile, records are ordered by a stable SHA-256 key derived from `(seed, split, family, quartile, group_id)` and apportioned with deterministic largest-remainder counts. The resulting counts are exactly 6,154 train / 1,319 validation / 1,319 test.

The ordered manifest hashes are:

```text
train:      adefdbcbef75c32a507d08777e3bd633be597d526e56c6c24ab6874ee6474d85
validation: d95700fd70f5e1883f57a91072e6927a8258863fb90d82d6c958af5f42081b4a
test:       59b8fa1f69161c7878148e09b4d78dacb8c7a184b68f651d7658042be6ede5d4
combined:   e7558294e1ea5fee078aa5d86f924afd8e122bbbf85fafc209ab71a29fe2b59c
```

This is the only option that follows N1's 70/15/15 hash assignment literally. It intentionally makes the original public `test.jsonl` provenance mixed across all three partitions, so it must not be compared with a published GSM8K test score. It is safe against leakage *within this lock* because no exact or near-duplicate group crosses partitions, but it cannot remove pretraining contamination from a public 2021 corpus.

### Option B — official-test holdout with an explicit N1 amendment (recommended)

Keep all 1,319 records from the pinned official `test.jsonl` untouched as the test partition. Deterministically split only the 7,473 official `train.jsonl` records into 6,154 train and 1,319 validation using the same seed, stable hash ordering, task-family label, and global completion-length quartile. The global counts remain exactly 70/15/15 (6,154 / 1,319 / 1,319), while the test partition retains its upstream provenance.

The ordered manifest hashes are:

```text
train:      4a526f586bf549ef53c731207f70c2c66bc5c0ce14dc17c5a27d2d3a1ce86c5e
validation: b40ebbd0ee34fecc4d9b5bbf69f47ad0f1efd8b8f75e78b028358d44c39a6edc
test:       97ab4cb63c575225c17cf3d373755cf119725035f1de4dca04b7dcce89970d52
combined:   9db52ca0192e4a8894da4f9f571c7a30484ff5b24a946a7fc74e33cba20dfb6d
```

This requires one prospective N1 amendment, recorded before any model run:

1. Run exact/near-duplicate grouping over the full union first.
2. Permit the official test source partition to be a pre-locked provenance group only when no group crosses that boundary (true for this pinned revision).
3. Apply N1's deterministic hash/length stratification to the adjustable official-train pool, while reporting the official-test quartile counts rather than pretending they were hash-balanced.
4. Keep the N1 test embargo, selection-record sealing, metrics, and no-post-hoc-change rules unchanged.

Option B is preferable for the stated research question because it preserves the original audit's untouched official holdout and prevents direct SFT exposure to those 1,319 examples. It still does not make GSM8K uncontaminated: the official test is public and may have appeared in Qwen or other pretraining data before this experiment.

## Recommendation and contamination limitations

**Recommendation:** accept Option B only after the Sol leader records the amendment above; otherwise use Option A and label the result as a new N1 global split, never as an official GSM8K benchmark score.

Option B best reconciles the original recommendation (“do not train on the official test split”) with N1's exact global 70/15/15 sizes and group-safe rule. The zero cross-boundary duplicate result makes that reconciliation evidence-based rather than assumed. Option A remains a clean, literal N1 fallback if the protocol cannot accept a provenance-preserving exception.

For both options, the claim remains bounded:

* Public release means pretraining contamination cannot be ruled out. A high test score is not proof of clean arithmetic generalization.
* The source paper reports residual answer disagreement/ambiguity in a rechecked subset; preserve any malformed records and do not silently repair them.
* GSM8K's final-number parser is deterministic, but final-number exact match does not validate the rationale. Protocol N1's primary test endpoint remains macro-example completion NLL; a positive broad transfer claim still requires a locked structured endpoint.
* The exact Qwen chat template, supervised-token offsets, completion-token quartiles, and 256-token truncation rate are **pending**. The lock cannot become confirmatory until a model-free implementation preflight computes those fields against the hashed local tokenizer assets and verifies N1's <=1% truncation gate. No model or tokenizer was loaded here.

## Local artifacts and next gate

Generated inside this owned folder only:

* `source/gsm8k-3101c7d/{train,test}.jsonl`, `README.md`, and `LICENSE`;
* `lock/LOCK-PROPOSAL.json`;
* `lock/n1-global/{train,validation,test}.jsonl`;
* `lock/official-test/{train,validation,test}.jsonl`;
* `lock_gsm8k.py`.

The next owner should treat `LOCK-PROPOSAL.json` as immutable input, choose Option A or record the Option B amendment, then run the model-free chat-format/token-offset/truncation preflight. Only after that receipt is complete may the single Qwen slot be requested. No Qwen/model run, external publication, cloud use, spending, or write outside this project occurred in this continuation.
