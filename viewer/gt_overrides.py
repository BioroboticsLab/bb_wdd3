"""Ground-truth annotation overrides — sidecar JSON system.

Stores manual GT edits (add / modify / delete) separately from the main
annotations CSV, allowing non-destructive editing with easy revert.

Sidecar format:
{
    "added": {
        "video_name.mp4": [
            {"run_id": 99, "x": ..., "y": ..., "dir_x": ..., "dir_y": ...,
             "waggle_start": ..., "waggle_end": ...},
            ...
        ]
    },
    "modified": {
        "video_name.mp4": {
            "3": {"x": 200.0, "waggle_end": 90}   // run_id → partial updates
        }
    },
    "deleted": {
        "video_name.mp4": [1, 5]                    // list of deleted run_ids
    }
}
"""

import json
import os
from typing import Any


def load_overrides(path: str) -> dict:
    """Load sidecar file, returning empty structure if missing."""
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        # Ensure all sections exist
        data.setdefault('added', {})
        data.setdefault('modified', {})
        data.setdefault('deleted', {})
        return data
    return {'added': {}, 'modified': {}, 'deleted': {}}


def save_overrides(path: str, data: dict) -> None:
    """Atomically write sidecar JSON (write to .tmp then rename)."""
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def add_annotation(data: dict, video_name: str, fields: dict,
                   next_run_id: int) -> dict:
    """Add a new GT annotation.
    
    Args:
        data: sidecar data dict (mutated in place)
        video_name: which video this annotation belongs to
        fields: dict with x, y, dir_x, dir_y, waggle_start, waggle_end
        next_run_id: integer ID to assign
    
    Returns:
        The complete annotation dict (with run_id set).
    """
    run = {
        'run_id': next_run_id,
        'x': float(fields['x']),
        'y': float(fields['y']),
        'dir_x': float(fields['dir_x']),
        'dir_y': float(fields['dir_y']),
        'waggle_start': int(fields['waggle_start']),
        'waggle_end': int(fields['waggle_end']),
    }
    data['added'].setdefault(video_name, []).append(run)
    return run


def modify_annotation(data: dict, video_name: str, run_id: int,
                      updates: dict) -> None:
    """Modify an existing annotation.
    
    If the run was previously added (in the sidecar), update it in-place.
    Otherwise, store as a modification of a CSV-sourced run.
    """
    # Check if this run was added in the sidecar
    added_runs = data['added'].get(video_name, [])
    for run in added_runs:
        if run['run_id'] == run_id:
            # Update in-place
            for k, v in updates.items():
                run[k] = v
            return

    # It's a CSV-sourced run — store in 'modified'
    data['modified'].setdefault(video_name, {})
    rid_key = str(run_id)
    if rid_key not in data['modified'][video_name]:
        data['modified'][video_name][rid_key] = {}
    data['modified'][video_name][rid_key].update(updates)


def delete_annotation(data: dict, video_name: str, run_id: int) -> None:
    """Delete an annotation.
    
    If the run was previously added (in the sidecar), remove it.
    Otherwise, mark as deleted (CSV-sourced run).
    """
    # Check if this run was added in the sidecar
    added_runs = data['added'].get(video_name, [])
    for i, run in enumerate(added_runs):
        if run['run_id'] == run_id:
            added_runs.pop(i)
            # Also clean up any modifications
            mods = data['modified'].get(video_name, {})
            mods.pop(str(run_id), None)
            return

    # CSV-sourced run — mark as deleted
    data['deleted'].setdefault(video_name, [])
    if run_id not in data['deleted'][video_name]:
        data['deleted'][video_name].append(run_id)

    # Also remove any modifications for this run
    mods = data['modified'].get(video_name, {})
    mods.pop(str(run_id), None)


def merge_annotations(csv_runs: list[dict], data: dict,
                      video_name: str) -> list[dict]:
    """Merge CSV-sourced annotations with sidecar overrides.
    
    Args:
        csv_runs: list of run dicts from the CSV (for this video)
        data: sidecar override data
        video_name: video to merge for
    
    Returns:
        Merged list of run dicts.
    """
    deleted_ids = set(data['deleted'].get(video_name, []))
    modifications = data['modified'].get(video_name, {})
    added_runs = data['added'].get(video_name, [])

    result = []

    # Process CSV runs: skip deleted, apply modifications
    for run in csv_runs:
        if run['run_id'] in deleted_ids:
            continue
        
        run_copy = dict(run)  # shallow copy
        rid_key = str(run['run_id'])
        if rid_key in modifications:
            run_copy.update(modifications[rid_key])
        
        result.append(run_copy)

    # Append added runs
    for run in added_runs:
        result.append(dict(run))

    return result


def get_next_run_id(csv_runs: list[dict],
                    added_runs: list[dict]) -> int:
    """Compute the next available run_id for a video.
    
    Returns max(all existing run_ids) + 1, or 1 if none exist.
    """
    all_ids = [r['run_id'] for r in csv_runs] + [r['run_id'] for r in added_runs]
    return max(all_ids) + 1 if all_ids else 1
