# Bug Hunting Diary

## 2026-03-29: Downsampled FPS Variants Have Wrong Window Sizes

### Bug Description

The `_ds30fps` and `_ds15fps` video variants have annotation windows that are **shorter than 16 frames**, which means the model receives padded/repeated input instead of a proper 16-frame temporal window:

| Variant | Native FPS | Downsample Factor | Window Frames | Expected |
|---------|-----------|-------------------|---------------|----------|
| Native (60fps) | 60 | 1× | 16 ✅ | 16 |
| `_ds30fps` | 30 | 2× | **8** ❌ | 16 |
| `_ds15fps` | 15 | 4× | **4** ❌ | 16 |

**Impact**: 
- 18,067 rows (27% of training data) have 8-frame windows → repeat-padded to 16
- 18,067 rows (27% of training data) have 4-frame windows → repeat-padded to 16  
- The temporal labels (`waggle_start_in_window`, `waggle_end_in_window`) are also affected — they're scaled down proportionally, compressing the temporal signal
- The `_sample_frames()` function masks this by repeat-padding, but the model sees identical frames repeated which is NOT the same as seeing distinct frames from a wider temporal window

### Root Cause

In [prepare_training_data.py:89-106](file:///home/landgraf/workspaces/waggle_net/bb_wdd3/prepare_training_data.py#L89-L106), the `create_downsampled_annotations()` function divides **all** frame columns by the downsampling factor:

```python
def create_downsampled_annotations(df, src_video, dst_video, factor):
    # ...
    frame_cols = ["start_frame", "end_frame", "waggle_start", "waggle_end",
                  "waggle_start_in_window", "waggle_end_in_window"]
    for col in frame_cols:
        if col in dst_rows.columns:
            dst_rows[col] = (dst_rows[col] / factor).astype(int)
```

This is correct for converting frame indices to the new timebase (`frame_60fps / 2 = frame_30fps`), but it also shrinks the window size proportionally. A 16-frame window at 60fps becomes an 8-frame window at 30fps.

**The actual temporal window should remain 16 frames regardless of FPS.** The downsampled videos have fewer frames per second, so the same 16-frame window covers a longer real-time span at lower FPS — which is fine because the model should learn to recognize waggle dances at different temporal scales.

### What Should Happen

For a downsampled video, the annotation should:
1. **Keep `start_frame` and `end_frame` 16 frames apart** (expand the window in the downsampled timebase)
2. **Correctly place `waggle_start` / `waggle_end`** in the new timebase (divide by factor — this part is already correct)
3. **Recompute `waggle_start_in_window` and `waggle_end_in_window`** relative to the new wider window

### Where to Fix

#### Primary fix: [prepare_training_data.py](file:///home/landgraf/workspaces/waggle_net/bb_wdd3/prepare_training_data.py)

The `create_downsampled_annotations()` function (line 89) needs to be changed to:

1. Convert `waggle_start` and `waggle_end` to the new timebase (divide by factor) — **already correct**
2. **Recompute `start_frame` and `end_frame`** to create a proper 16-frame window centered on the waggle event in the new timebase, instead of just dividing by factor
3. **Recompute `waggle_start_in_window` and `waggle_end_in_window`** relative to the new 16-frame window
4. **Handle edge cases**: window might extend beyond video boundaries — need to clamp, and the waggle positions within the window need to match

#### Pseudocode for the fix:
```python
def create_downsampled_annotations(df, src_video, dst_video, factor, target_window=16):
    src_rows = df[df["video_name"] == src_video].copy()
    dst_rows = src_rows.copy()
    dst_rows["video_name"] = dst_video
    
    # Convert absolute frame positions to new timebase
    for col in ["waggle_start", "waggle_end"]:
        dst_rows[col] = (dst_rows[col] / factor).astype(int)
    
    # For each row, recompute the window to be target_window frames
    for idx in dst_rows.index:
        ws = dst_rows.loc[idx, "waggle_start"]
        we = dst_rows.loc[idx, "waggle_end"]
        
        # Center the window on the waggle midpoint
        mid = (ws + we) // 2
        new_start = mid - target_window // 2
        new_end = new_start + target_window
        
        # TODO: clamp to video bounds
        
        dst_rows.loc[idx, "start_frame"] = new_start
        dst_rows.loc[idx, "end_frame"] = new_end
        dst_rows.loc[idx, "waggle_start_in_window"] = ws - new_start
        dst_rows.loc[idx, "waggle_end_in_window"] = we - new_start
    
    return dst_rows
```

#### Secondary: Regenerate the clean CSV
After fixing `create_downsampled_annotations()`, re-run:
```bash
python prepare_training_data.py
```
This regenerates `data/annotations/fps_multires_clean.csv` with correct 16-frame windows.

#### No changes needed in:
- [dataset.py](file:///home/landgraf/workspaces/waggle_net/bb_wdd3/src/data/dataset.py) — the `_sample_frames()` repeat-padding will still work correctly, but with the fix it will see `16 >= 16` and use `linspace` (identity) instead of repeat-padding
- [main.py](file:///home/landgraf/workspaces/waggle_net/bb_wdd3/main.py) — no changes
- Viewer/inspector — will automatically show 16 frames once annotations are correct

### Verification Plan
1. After fix, check that all rows in `fps_multires_clean.csv` have `end_frame - start_frame == 16`
2. In the Pipeline Inspector, verify ds30fps/ds15fps variants show 16 distinct frames (no repeat-padding)
3. Verify that `waggle_start_in_window` and `waggle_end_in_window` are within [0, 16] for all rows
4. Check that the temporal labels (start_norm, end_norm) are reasonable for the new window positions
