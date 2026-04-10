# Waggle Dance Annotation Viewer

Browser-based tool for inspecting ground-truth waggle dance annotations overlaid on raw video frames.

## Quick Start

```bash
# From the bb_wdd3 directory
.venv/bin/python viewer/server.py
```

Then open **http://localhost:5050** in your browser.

## Remote Access

### Option A: Direct (faster, if port is reachable via VPN)
```bash
# On horus
.venv/bin/python viewer/server.py --host 0.0.0.0

# In browser
http://160.45.38.75:5050
```

### Option B: SSH Tunnel (always works)
```bash
# On your laptop
ssh -L 5050:localhost:5050 landgraf@horus

# On horus
.venv/bin/python viewer/server.py

# In browser
http://localhost:5050
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play / Pause |
| `←` / `→` | Previous / Next frame |
| `Shift+←` / `Shift+→` | Jump 10 frames |
| `A` | Toggle annotations |
| `R` | Random recording |
| `1-9` | Jump to waggle run N |

## Annotation States

- **Solid** — frame is within `[waggle_start, waggle_end]`
- **Ghost** (semi-transparent) — frame is within 1-second padding around waggle; shows where the bee will appear or last appeared
- **Hidden** — frame is outside the padding window
