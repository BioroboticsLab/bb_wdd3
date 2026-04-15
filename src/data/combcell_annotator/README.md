# Comb Cell Annotator

This tool lets you annotate comb cells in beehive videos to figure out how big a comb cell is in pixels. We need this because each camera setup has a different resolution, so the same physical cell looks different in size depending on the video source.

## Why comb cells?

Comb cells are a good reference because they always have roughly the same real-world size. Once we know how many pixels one cell takes up in a video, we can use that to estimate the size of anything else in that same video — like a bee's body.

## What it does

- Randomly picks a few videos from each data folder
- Extracts one or more frames from each video
- Shows each frame and lets you draw shapes over the comb cells
- At the end it gives you the average cell diameter in pixels per data source

## How to use it

Edit `config.yaml` to point to your data folders and set how many videos and frames you want to sample. Then run:

```bash
python annotator.py
```

A window will open for each frame. Draw your annotations, press D to move to the next frame, and Q if you want to stop early. When done, the results are printed to the terminal and saved as a CSV in the `output/` folder.

## Controls

| Key / Mouse | What it does |
|---|---|
| Left click + drag | Draw a circle over a cell |
| Click (hexagon mode) | Place one vertex — closes automatically after 6 |
| Right click + drag | Pan around when zoomed in |
| Arrow keys | Also pan around when zoomed in |
| + | Zoom in (centered on mouse) |
| - | Zoom out |
| B | Reset zoom back to normal |
| U | Undo last circle or last vertex |
| D | Done with this frame, go to next |
| Q | Quit early, still get the summary |

## Config options

```yaml
data_path: ../data          # where your video folders are
subfolders:
  - name: my_folder
    rotation: 90            # rotate frames if the camera was sideways
n_videos: 2                 # how many videos to pick per folder
n_frames: 3                 # how many frames to extract per video
shape: circle               # circle or hexagon
```

## Output

Results are saved to `output/annotations_<shape>_<timestamp>.csv`. Each row is one annotation with the pixel coordinates and the computed diameter. The terminal also prints a short summary with the average cell size per folder.
