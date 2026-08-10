# Serialized model process state

Current state: **RELEASED — frozen batch complete**  
Released: 2026-08-10T05:13:58+08:00  
Headroom at release: 75% system memory free; no competing Qwen/MLX process.

Exact command:

```text
/Users/vukrosic/miniconda3/bin/python3 run_capture.py --run-dir runs/test-20260822-example -- /Users/vukrosic/miniconda3/bin/python3 evaluate_condition.py --model /Users/vukrosic/.cache/models/Qwen3-0.6B-3bit --adapter runs/train-20260822-example/artifact --data data/v2 --data-lock DATASET-LOCK-v2.json --output runs/test-20260822-example/metrics --condition example --seed 20260822 --split test --selection runs/selection-20260822.json --batch-size 4 --max-seq-length 256 --max-generation-tokens 256
```

## Ledger

| Time | Event | Process | Evidence |
| --- | --- | --- | --- |
| 2026-08-10T01:53:08+08:00 | CLAIMED | seed 20260820 token train | 40% free; no competing model process |
| 2026-08-10T02:10:52+08:00 | RELEASED | seed 20260820 token train | exit 0; 576 updates; 1008.31 s launcher; 2.313 GB peak MLX; adapter `507fc59e…` |
| 2026-08-10T02:11:16+08:00 | CLAIMED | seed 20260820 example train | 54% free; no competing model process |
| 2026-08-10T02:28:23+08:00 | RELEASED | seed 20260820 example train | exit 0; 576 updates; 992.64 s launcher; 2.309 GB peak MLX; adapter `3ef04e21…` |
| 2026-08-10T02:28:34+08:00 | CLAIMED | seed 20260820 token validation | 62% free; no competing model process |
| 2026-08-10T02:30:02+08:00 | RELEASED | seed 20260820 token validation | exit 0; 60.71 s launcher; 1.756 GB peak MLX; selector NLL `0.94261534`; metrics `49e69726…` |
| 2026-08-10T02:33:00+08:00 | CLAIMED | seed 20260820 example validation | 72% free; no competing model process |
| 2026-08-10T02:33:36+08:00 | RELEASED | seed 20260820 example validation | exit 0; 59.31 s launcher; 1.756 GB peak MLX; selector NLL `0.94198879`; metrics `92f90f60…` |
| 2026-08-10T02:33:54+08:00 | SEALED | seed 20260820 selection | example selected by common validation NLL; selection `70a3269f…`; test not started |
| 2026-08-10T02:34:10+08:00 | CLAIMED | seed 20260820 token final test | 71% free; no competing model process; sealed selection present |
| 2026-08-10T02:46:42+08:00 | RELEASED | seed 20260820 token final test | exit 0; 715.89 s launcher; 1.618 GB peak MLX; macro NLL `0.91024583`; exact `0.03515625`; metrics `5db9373e…` |
| 2026-08-10T02:47:00+08:00 | CLAIMED | seed 20260820 example final test | 75% free; no competing model process; sealed selection `70a3269f…` |
| 2026-08-10T03:00:40+08:00 | RELEASED | seed 20260820 example final test | exit 0; 712.50 s launcher; 1.618 GB peak MLX; macro NLL `0.91197081`; exact `0.06640625`; metrics `0df464c4…` |
| 2026-08-10T03:00:51+08:00 | PAIR VALID | seed 20260820 | selected example loses primary by `0.00172498`, wins complementary exact by `0.03125`; marker rate tied; preserve as mixed per-seed evidence |
| 2026-08-10T03:01:00+08:00 | CLAIMED | seed 20260821 example training | 73% free; no competing model process |
| 2026-08-10T03:19:09+08:00 | RELEASED | seed 20260821 example training | exit 0; 576 updates; 1017.01 s launcher; 2.309 GB peak MLX; adapter `478a951c…` |
| 2026-08-10T03:19:16+08:00 | CLAIMED | seed 20260821 token training | 61% free; no competing model process |
| 2026-08-10T03:36:29+08:00 | RELEASED | seed 20260821 token training | exit 0; 576 updates; 998.86 s launcher; 2.309 GB peak MLX; adapter `fc686c4c…` |
| 2026-08-10T03:37:06+08:00 | CLAIMED | seed 20260821 example validation | 63% free; no competing model process |
| 2026-08-10T03:38:39+08:00 | RELEASED | seed 20260821 example validation | exit 0; 60.62 s launcher; selector NLL `0.94856075`; 1.756 GB peak MLX; metrics `b4c81131…` |
| 2026-08-10T03:38:46+08:00 | CLAIMED | seed 20260821 token validation | 68% free; no competing model process |
| 2026-08-10T03:40:16+08:00 | RELEASED | seed 20260821 token validation | exit 0; 61.06 s launcher; selector NLL `0.95206855`; 1.756 GB peak MLX; metrics `5f210560…` |
| 2026-08-10T03:40:27+08:00 | SEALED | seed 20260821 selection | example selected by common validation NLL; selection `a388ed03…`; test not started |
| 2026-08-10T03:41:00+08:00 | CLAIMED | seed 20260821 example final test | 68% free; no competing model process; sealed selection present |
| 2026-08-10T03:55:54+08:00 | RELEASED | seed 20260821 example final test | exit 0; 853.66 s launcher; 1.618 GB peak MLX; macro NLL `0.91640116`; exact `0.046875`; metrics `d884e57a…` |
| 2026-08-10T03:56:02+08:00 | CLAIMED | seed 20260821 token final test | 75% free; no competing model process; sealed selection `a388ed03…` |
| 2026-08-10T04:10:37+08:00 | RELEASED | seed 20260821 token final test | exit 0; 807.86 s launcher; 1.618 GB peak MLX; macro NLL `0.91276430`; exact `0.03515625`; metrics `2de9c1b0…` |
| 2026-08-10T04:10:45+08:00 | PAIR VALID | seed 20260821 | selected example loses primary macro NLL by `0.00363686`, wins exact by `0.01171875`, wins marker by `0.013671875`; preserve as mixed per-seed evidence |
| 2026-08-10T04:10:45+08:00 | HELD | model slot | separate initiative MLX tiny-trainer active; did not launch Qwen |
| 2026-08-10T04:11:00+08:00 | CLAIMED | seed 20260822 token training | competing MLX probe exited; 74% free; no competing model process |
| 2026-08-10T04:27:22+08:00 | RELEASED | seed 20260822 token training | exit 0; 576 updates; 905.53 s launcher; 2.309 GB peak MLX; adapter `7821e482…` |
| 2026-08-10T04:27:43+08:00 | CLAIMED | seed 20260822 example training | 64% free; no competing model process |
| 2026-08-10T04:43:21+08:00 | RELEASED | seed 20260822 example training | exit 0; 576 updates; 908.14 s launcher; 2.309 GB peak MLX; adapter `d8a92e76…` |
| 2026-08-10T04:44:00+08:00 | CLAIMED | seed 20260822 token validation | 63% free; no competing model process |
| 2026-08-10T04:45:29+08:00 | RELEASED | seed 20260822 token validation | exit 0; 55.21 s launcher; selector NLL `0.95117902`; 1.756 GB peak MLX; metrics `dbfd6f84…` |
| 2026-08-10T04:45:35+08:00 | CLAIMED | seed 20260822 example validation | 63% free; no competing model process |
| 2026-08-10T04:46:59+08:00 | RELEASED | seed 20260822 example validation | exit 0; 55.63 s launcher; selector NLL `0.95301524`; 1.756 GB peak MLX; metrics `23702231…` |
| 2026-08-10T04:47:09+08:00 | SEALED | seed 20260822 selection | token selected by common validation NLL; selection `8dd48989…`; test not started |
| 2026-08-10T04:47:09+08:00 | CLAIMED | seed 20260822 token final test | 69% free; no competing model process; sealed selection present |
| 2026-08-10T04:59:52+08:00 | RELEASED | seed 20260822 token final test | exit 0; 693.57 s launcher; 1.618 GB peak MLX; macro NLL `0.92626940`; exact `0.03515625`; metrics `e4e02af3…` |
| 2026-08-10T05:01:00+08:00 | CLAIMED | seed 20260822 example final test | 72% free; no competing model process; sealed selection `8dd48989…` |
| 2026-08-10T05:13:42+08:00 | RELEASED | seed 20260822 example final test | exit 0; 677.68 s launcher; 1.618 GB peak MLX; macro NLL `0.92423149`; exact `0.041015625`; metrics `27d1f9d4…` |
| 2026-08-10T05:13:58+08:00 | BATCH RELEASED | initiative model slot | all frozen heavy processes complete; 75% free; no competing model process; no further model use required |
