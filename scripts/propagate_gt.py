#!/usr/bin/env python3
"""Propagate GT overrides from 60fps to 30fps and 15fps variants."""
import json, os

SIDECAR = os.path.join(os.path.dirname(__file__), "..", "data", "annotations", "gt_overrides.json")
SIDECAR = os.path.abspath(SIDECAR)
BASE = "C1_11_03_25_my_video-1-01.00.00.815-01.02.00.957_1224_1024"
VN_60 = BASE + ".mp4"
RATIOS = {BASE + "_ds30fps.mp4": 2.0, BASE + "_ds15fps.mp4": 4.0}

with open(SIDECAR) as f:
    data = json.load(f)

mods_60 = data["modified"].get(VN_60, {})
print(f"60fps: {len(mods_60)} mods (runs: {sorted(mods_60.keys(), key=int)})")

for target_vn, ratio in RATIOS.items():
    target_mods = {}
    for rid, fields in mods_60.items():
        scaled = dict(fields)
        if "waggle_start" in fields:
            scaled["waggle_start"] = round(fields["waggle_start"] / ratio)
        if "waggle_end" in fields:
            scaled["waggle_end"] = round(fields["waggle_end"] / ratio)
        target_mods[rid] = scaled
    data["modified"][target_vn] = target_mods
    suffix = "_ds30fps" if "ds30" in target_vn else "_ds15fps"
    print(f"  {suffix}: {len(target_mods)} mods propagated")

tmp = SIDECAR + ".tmp"
with open(tmp, "w") as f:
    json.dump(data, f, indent=2)
os.replace(tmp, SIDECAR)
print("Done - saved to", SIDECAR)
