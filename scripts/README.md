# Scripts

Diagnostics and one-off analyses. These are working tools rather than a maintained CLI — several
carry hardcoded paths or video names, noted below. Run them from the repository root.

## `analyze_stratified_recall.py`

Asks whether dance-level detection depends on how long a dance is — specifically, on how many
sliding windows it spans. Reuses the evaluation pipeline up through `predicted_runs` / `gt_dances`,
then matches at a single threshold combination instead of the full sweep so per-dance
detected/not-detected status survives. Bins dances by window count and reports recall per bin.

Short dances spanning few windows are the ones clustering is most likely to lose, so a recall curve
that falls off at the low end points at post-processing rather than the network.

```bash
.venv/bin/python scripts/analyze_stratified_recall.py --ckpt_path ckpt/clean_split_v1/best.pth
```

## `dance_level_pass_rates.py`

The dance-level counterpart of `diagnostic.compute_pass_rates()`. The window-level version takes a
shortcut — top prediction against first ground truth — which makes its numbers incomparable to the
real matcher. This one uses the actual heuristic (confidence-sorted greedy, joint validity,
cost-minimised among valid candidates), so the position, temporal and angular pass rates it reports
can be compared directly against window-level ones.

Thresholds are fixed constants at the top of the file (`POS_T`, `IOU_T`, `ANG_T`), chosen to match
the window-level test.

```bash
.venv/bin/python scripts/dance_level_pass_rates.py --ckpt_path ckpt/clean_split_v1/best.pth
```

## `analyze_balance.py`

Waggle-run distribution across the train/val split, broken down by recording site. Useful for
checking that a split has not accidentally concentrated one lab on one side.

> **Hardcoded to horus.** Three absolute paths point at
> `/home/landgraf/workspaces/waggle_net/bb_wdd3`. Edit them, or run it there.

The site names differ from the filename prefixes used elsewhere in the codebase:

| Prefix | Codebase name | This script |
|---|---|---|
| digit | berlin | Berlin |
| `T` | nieh | San Diego |
| `C` | sharoni | Jerusalem |

## `propagate_gt.py`

Copies ground-truth overrides made on a 60 fps recording down to its 30 fps and 15 fps variants,
scaling frame numbers by the rate ratio. Saves correcting the same annotation three times.

> **Hardcoded to one recording** — the `C1_11_03_25…` sharoni clip, in `BASE` at the top of the
> file. Change it for anything else.

## `test_det.py`

Nine lines: runs the train/val split five times and prints the counts, to confirm it is
deterministic under a fixed seed. Not a test in the `pytest` sense — the actual test suite is in
`tests/` and `src/tests/`.
