"""
Quick standalone test comparing the old row-level lab capping
(cap_lab_samples) against the new dance-level, duration-stratified
capping (cap_lab_samples_by_dance).

This does not touch any real training run. It just loads the training
split, applies both capping functions with the same budget, and prints
some stats so we can see if the new one actually fixes the short/medium/
long dance imbalance before we wire it into main.py.

Usage:
    PYTHONPATH=. python scripts/test_dance_balance.py
"""
import pandas as pd

from src.utils.data_utils import load_config, cap_lab_samples, cap_lab_samples_by_dance
from src.utils.video_utils import train_val_split_videos, get_video_category

SEED = 42


def dance_table(df):
    """One row per dance instance, with its row count and duration."""
    df = df.copy()
    df['_category'] = df['video_name'].apply(get_video_category)
    df['_duration'] = df['waggle_end'] - df['waggle_start']
    dances = df.groupby(['_category', 'video_name', 'waggle_run_id']).agg(
        n_rows=('_duration', 'size'),
        duration=('_duration', 'max'),
    ).reset_index()
    return dances


def print_duration_breakdown(df, label):
    dances = dance_table(df)
    print(f"\n--- {label} ---")
    print(f"total rows: {len(df)}, total dances: {len(dances)}")
    for category, group in dances.groupby('_category'):
        short = (group['duration'] <= 9).sum()
        medium = ((group['duration'] > 9) & (group['duration'] <= 19)).sum()
        long_ = (group['duration'] > 19).sum()
        total = len(group)
        print(f"  lab={category}: dances={total}  "
              f"short<=9f={short} ({short/total:.0%})  "
              f"med10-19f={medium} ({medium/total:.0%})  "
              f"long>19f={long_} ({long_/total:.0%})  "
              f"rows={group['n_rows'].sum()}")


def assign_fixed_bins(raw_df, n_duration_bins=3):
    """Compute duration bins ONCE from the raw uncapped data per lab, so we
    can compare different capping methods against the same fixed yardstick
    instead of each method's own (different) subset shifting the bin
    boundaries."""
    dances = dance_table(raw_df)
    bin_map = {}
    for category, group in dances.groupby('_category'):
        bins = pd.qcut(group['duration'], n_duration_bins, duplicates='drop', labels=False)
        for (_, vname, rid), b in zip(group[['_category', 'video_name', 'waggle_run_id']].values, bins):
            bin_map[(vname, rid)] = b
    return bin_map


def print_fixed_bin_breakdown(df, bin_map, label):
    """Dance counts AND rows retained per duration bin, using bins fixed on
    the raw data. Dance count is what matters for Tim's "similar number of
    short/medium/long dances" goal; rows is just for reference (it's the
    thing being capped)."""
    dances = dance_table(df)
    dances['_bin'] = dances.apply(lambda r: bin_map.get((r['video_name'], r['waggle_run_id'])), axis=1)
    print(f"\n--- {label} (dances / rows per FIXED duration bin, same bins for all methods) ---")
    for category, group in dances.groupby('_category'):
        bin_counts = group.groupby('_bin').agg(n_dances=('n_rows', 'size'), n_rows=('n_rows', 'sum'))
        parts = [f"bin{int(b)}={row.n_dances}dances/{row.n_rows}rows"
                 for b, row in bin_counts.sort_index().iterrows()]
        print(f"  lab={category}: " + "  ".join(parts))


def main():
    config = load_config('./configs/config_balanced_300k.yaml')
    data = pd.read_csv(config['data']['annotations'])
    train_videos, _ = train_val_split_videos(
        data['video_name'].unique(),
        train_ratio=config['data']['train_ratio'],
        seed=SEED,
    )
    train_df = data[data['video_name'].isin(train_videos)].reset_index(drop=True)

    caps = config['data']['lab_caps']
    print(f"caps being applied: {caps}")

    print_duration_breakdown(train_df, "BEFORE any capping (raw train split)")

    old_capped = cap_lab_samples(train_df, caps, seed=SEED)
    print_duration_breakdown(old_capped, "AFTER cap_lab_samples (old, row-level random)")

    new_capped = cap_lab_samples_by_dance(train_df, caps, n_duration_bins=3, seed=SEED)
    print_duration_breakdown(new_capped, "AFTER cap_lab_samples_by_dance (new, dance-level stratified)")

    bin_map = assign_fixed_bins(train_df, n_duration_bins=3)
    print_fixed_bin_breakdown(train_df, bin_map, "BEFORE any capping")
    print_fixed_bin_breakdown(old_capped, bin_map, "OLD capping (cap_lab_samples)")
    print_fixed_bin_breakdown(new_capped, bin_map, "NEW capping (cap_lab_samples_by_dance)")

    # sanity check: did we actually stay under budget, and did we keep
    # every sampled dance fully intact (no partial dances)?
    print("\n--- sanity checks on the new function ---")
    for category, cap in caps.items():
        n_rows = len(new_capped[new_capped['video_name'].apply(get_video_category) == category])
        print(f"  lab={category}: cap={cap}, actual rows kept={n_rows}, "
              f"within budget={n_rows <= cap}")

    # intactness check: for every dance that appears at all in new_capped,
    # does it have the SAME row count as in the original uncapped data?
    orig_dances = dance_table(train_df).set_index(['video_name', 'waggle_run_id'])['n_rows']
    new_dances = dance_table(new_capped).set_index(['video_name', 'waggle_run_id'])['n_rows']
    mismatched = 0
    for key, n_rows in new_dances.items():
        if orig_dances.loc[key] != n_rows:
            mismatched += 1
    print(f"  dances partially dropped (should be 0): {mismatched} / {len(new_dances)}")


if __name__ == '__main__':
    main()
