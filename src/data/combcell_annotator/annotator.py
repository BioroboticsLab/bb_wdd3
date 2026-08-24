#!/usr/bin/env python3
"""
Comb Cell Annotator
Annotate comb cells in beehive video frames to estimate bee physical size in pixels.

Controls (circle mode):
  Left click + drag   : draw a circle over a comb cell

Controls (hexagon mode):
  Left click          : place a vertex (auto-closes after 6 vertices)

Shared controls:
  Right click + drag  : pan when zoomed in
  Arrow keys          : move around the image when zoomed in
  +                   : zoom in 10%, centered on mouse position
  -                   : zoom out 10%
  B                   : reset zoom back to original
  U                   : undo last point or last shape
  D                   : done with this frame, move to next
  Q                   : quit early and still get the summary
"""

import csv
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

# Honeybee comb cells are hexagons, so we always need exactly 6 vertices
HEX_VERTICES = 6

# How many pixels to move per arrow key press (in display space)
ARROW_STEP_PX = 40

# Arrow key codes vary by platform and OpenCV build.
# We collect all known codes so the tool works on macOS and Linux.
# waitKeyEx returns extended codes; waitKey sometimes returns different values.
ARROW_UP    = (63232, 65362, 2490368)
ARROW_DOWN  = (63233, 65364, 2621440)
ARROW_LEFT  = (63234, 65361, 2424832)
ARROW_RIGHT = (63235, 65363, 2555904)


# Config

def load_config(config_path: Path) -> dict:
    # Read the YAML config file and return it as a plain Python dictionary
    with open(config_path) as f:
        return yaml.safe_load(f)


# Video utilities

def get_video_files(folder: Path) -> list:
    # Search for all common video formats in the folder.
    # We include upper-case extensions too because some cameras save them that way.
    extensions = ("*.mp4", "*.avi", "*.mov", "*.MP4", "*.AVI", "*.MOV")
    videos = []
    for ext in extensions:
        videos.extend(folder.glob(ext))
    return videos


def parse_resolution(filename: str):
    """Extract (width, height) from a trailing '_W_H' tag, e.g. '001_2304_1760.mp4'.
    Returns None if the filename has no such tag."""
    m = re.search(r'_(\d+)_(\d+)(?:_ds\d+fps)?\.\w+$', filename)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def lab_source(filename: str) -> str:
    """Identify which lab a video came from, based on filename prefix.
    Mirrors get_video_category() in src/utils/video_utils.py (numeric prefix ->
    Berlin, 'T' prefix -> Nieh, 'C' prefix -> Sharoni/Jerusalem) but reimplemented
    here, with human-readable names, so this standalone tool doesn't have to
    import that module (it pulls in torch/torchvision/cv2 at module level)."""
    if filename.startswith('0'):
        return "berlin"
    elif filename.startswith('T'):
        return "nieh"
    elif filename.startswith('C'):
        return "sharoni"
    else:
        return "other"


def base_stem(filename: str) -> str:
    """Strip the resolution (and _dsXXfps) suffix to identify the source recording,
    so that '001_2304_1760.mp4' and '001_576_440.mp4' are recognized as the same
    underlying video at different pixel resolutions (same duration/fps/frame count --
    a plain re-encode, not a different recording)."""
    name = re.sub(r'\.\w+$', '', filename)
    name = re.sub(r'_ds\d+fps$', '', name)
    name = re.sub(r'_\d+_\d+$', '', name)
    return name


def group_by_source(videos: list) -> list:
    """Group same-source videos (different resolution encodes of one recording)
    and return one representative per source: the highest-resolution file, plus
    the full set of resolutions available for that source.

    We only ever need to annotate the highest-resolution copy -- the measured
    diameter can be projected exactly to every sibling resolution afterwards,
    since they're pixel-exact rescales of each other.

    Returns: list of (representative_path, representative_res, sibling_resolutions)
    """
    groups = defaultdict(list)
    for v in videos:
        groups[base_stem(v.name)].append((v, parse_resolution(v.name)))

    representatives = []
    for entries in groups.values():
        with_res = [(v, r) for v, r in entries if r is not None]
        if with_res:
            best_v, best_r = max(with_res, key=lambda t: t[1][0] * t[1][1])
            siblings = sorted({r for _, r in with_res}, key=lambda r: -r[0] * r[1])
        else:
            best_v, best_r = entries[0]
            siblings = []
        representatives.append((best_v, best_r, siblings))
    return representatives


def get_total_frames(video_path: Path) -> int:
    # Let OpenCV determine how many frames are in the video 
    # without reading any of them
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def extract_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))

    # First try seeking directly to the requested frame.
    # This works for most videos but can fail with certain codecs that
    # do not support random access (e.g. some H.265 / VFR recordings), like 
    # Nieh downsampled videos. 
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()

    if not ret:
        # if Seeking failed walk backwards in steps of 10 frames.
        # This handles cases where the reported frame count is slightly wrong.
        for fallback in range(frame_idx - 10, -1, -10):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fallback)
            ret, frame = cap.read()
            if ret:
                print(f"    [warn] sought frame {frame_idx}, read frame {fallback} instead")
                break

    if not ret:
        # Seeking does not work at all for this codec, particularl the downsampled
        # Nieh Lab videos. So if codec is not compatible with opencv, this is a fail safe
        #  and we read read frames sequentially.
        # We use grab() which decodes only keyframes, which is a bit or a lot faster
        # than calling read() on every frame.
        print(f"    [warn] seeking failed, reading sequentially to frame {frame_idx}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(frame_idx):
            if not cap.grab():
                break
        ret, frame = cap.read()

    if not ret:
        # Last resort if all fails just take the very first frame.
        # Re-open the capture because the previous attempts may have left it
        # in a broken state where even seeking back to 0 does not work.
        print(f"    [warn] could not reach frame {frame_idx}, using frame 0")
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()

    cap.release()
    if not ret:
        raise ValueError(f"Could not read any frame from {video_path}")
    return frame


def get_frame_indices(total_frames: int, n_frames: int) -> list:
    """Return n_frames evenly spaced frame indices, avoiding very start/end."""
    if n_frames == 1:
        # Just use the middle frame
        return [total_frames // 2]
    # Split the video into (n_frames + 1) equal sections.
    # We sample from the internal boundaries so we Skip the very first and last
    # frames, which often have bad lighting or are still starting up.
    step = total_frames // (n_frames + 1)
    return [step * (i + 1) for i in range(n_frames)]


def pick_frame_index(total_frames: int, exclude: set) -> int:
    """Pick a random frame not already tried, avoiding the very start/end (bad
    lighting / not-yet-stable footage). Used when the user skips a frame that
    has too few visible comb cells."""
    margin = total_frames // 10
    candidates = [i for i in range(margin, total_frames - margin) if i not in exclude]
    if not candidates:
        candidates = [i for i in range(total_frames) if i not in exclude]
    return random.choice(candidates) if candidates else 0


# Map rotation angles to the corresponding OpenCV constants
_ROTATION_MAP = {
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    -90: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

def rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    # Look up the OpenCV rotation code for this angle.
    # If the angle is not supported (e.g. 45°), return the frame unchanged.
    code = _ROTATION_MAP.get(degrees % 360)
    if code is None:
        return frame
    return cv2.rotate(frame, code)

# Zoom and pan shared by both annotator classes

class _ZoomPanMixin:
    """Zoom and pan state shared by both annotator modes."""

    def _init_zoom(self):
        # zoom = 1.0 means we show the full original frame with no magnification
        self.zoom = 1.0
        # view_x and view_y are the top-left corner of what we are currently
        # showing, measured in original frame pixels
        self.view_x = 0.0
        self.view_y = 0.0
        # Last mouse position in display coordinates, needed to center the zoom
        self.display_mouse = None
        # Pan drag state active while the user holds right-click and moves the mouse
        self.panning = False
        self.pan_start_display = None  # display position where the drag started
        self.pan_start_view    = None  # view_x/view_y at the moment the drag started

    def _display_to_orig(self, dx, dy):
        # Convert a point from display pixels to original frame pixels.
        # We undo the crop offset (view_x/y) and the scale (zoom).
        return self.view_x + dx / self.zoom, self.view_y + dy / self.zoom

    def _orig_to_display(self, ox, oy):
        # Convert a point from original frame pixels to display pixels.
        # This is the inverse of _display_to_orig.
        return (ox - self.view_x) * self.zoom, (oy - self.view_y) * self.zoom

    def _clamp_view(self):
        # Make sure the view window stays inside the original frame.
        # Without this, the user could pan into black empty space.
        if self.frame_orig is None:
            return
        fh, fw = self.frame_orig.shape[:2]
        crop_w = fw / self.zoom
        crop_h = fh / self.zoom
        self.view_x = max(0.0, min(self.view_x, fw - crop_w))
        self.view_y = max(0.0, min(self.view_y, fh - crop_h))

    def zoom_in(self):
        # Increase zoom by 10% while keeping the pixel under the mouse fixed.
        # We first find which original pixel is under the mouse, then after
        # updating zoom we shift the view so that same pixel stays in place.
        if self.display_mouse is None:
            return
        dx, dy = self.display_mouse
        ox, oy = self._display_to_orig(dx, dy)
        self.zoom = min(self.zoom * 1.1, 10.0)
        self.view_x = ox - dx / self.zoom
        self.view_y = oy - dy / self.zoom
        self._clamp_view()
        self.dirty = True

    def zoom_out(self):
        # Decrease zoom by 10%, same logic as zoom_in but in reverse.
        # We stop at 1.0 so the view never shrinks below the full original frame.
        if self.zoom <= 1.0:
            return
        if self.display_mouse is None:
            return
        dx, dy = self.display_mouse
        ox, oy = self._display_to_orig(dx, dy)
        self.zoom = max(self.zoom / 1.1, 1.0)
        self.view_x = ox - dx / self.zoom
        self.view_y = oy - dy / self.zoom
        self._clamp_view()
        self.dirty = True

    def zoom_reset(self):
        # Go back to the full original frame with no zoom or pan
        self.zoom = 1.0
        self.view_x = 0.0
        self.view_y = 0.0
        self.dirty = True

    def pan_step(self, dx_sign: int, dy_sign: int):
        # Move the view by ARROW_STEP_PX display pixels in the given direction.
        # We divide by zoom because one display pixel = 1/zoom original pixels.
        # Only does anything when actually zoomed in.
        if self.zoom == 1.0:
            return
        self.view_x += dx_sign * ARROW_STEP_PX / self.zoom
        self.view_y += dy_sign * ARROW_STEP_PX / self.zoom
        self._clamp_view()
        self.dirty = True

    def _handle_pan(self, event, x, y):
        """Handle right-click drag to pan the view. Returns True if the event was consumed."""
        if event == cv2.EVENT_RBUTTONDOWN:
            # Record start position so we can compute how far the mouse moved
            self.panning = True
            self.pan_start_display = (x, y)
            self.pan_start_view    = (self.view_x, self.view_y)
            return True
        elif event == cv2.EVENT_RBUTTONUP:
            self.panning = False
            return True
        elif event == cv2.EVENT_MOUSEMOVE and self.panning:
            # Shift the view by the mouse delta, converted to original frame pixels
            dx = x - self.pan_start_display[0]
            dy = y - self.pan_start_display[1]
            self.view_x = self.pan_start_view[0] - dx / self.zoom
            self.view_y = self.pan_start_view[1] - dy / self.zoom
            self._clamp_view()
            self.dirty = True
            return True
        return False

    def _base_frame(self, disp_w: int, disp_h: int) -> np.ndarray:
        """Return the display frame adjusted for the current zoom and pan, with no annotations."""
        fh, fw = self.frame_orig.shape[:2]
        if self.zoom == 1.0:
            # No zoom show the full original frame at original resolution.
            # If display size matches exactly we just copy, otherwise resize.
            if fw == disp_w and fh == disp_h:
                return self.frame_orig.copy()
            return cv2.resize(self.frame_orig, (disp_w, disp_h))
        # Zoomed in cut out the visible region from the original frame,
        # then scale that crop up to fill the display window.
        crop_w = int(fw / self.zoom)
        crop_h = int(fh / self.zoom)
        x1, y1 = int(self.view_x), int(self.view_y)
        x2 = min(x1 + crop_w, fw)
        y2 = min(y1 + crop_h, fh)
        return cv2.resize(self.frame_orig[y1:y2, x1:x2], (disp_w, disp_h))

# Circle annotator

class CircleAnnotator(_ZoomPanMixin):
    def __init__(self):
        # Finished circles stored as (center_x, center_y, radius) in original frame pixels
        self.circles      = []
        self.drawing      = False  # True while the user is holding left-click down
        self.center       = None   # click position where the current circle started
        self.current_mouse = None  # current mouse position, used for the live preview
        self.frame_orig   = None
        self.dirty        = True   # True means we need to redraw the display
        self._init_zoom()

    def reset(self, frame: np.ndarray):
        # Clear all annotations and load a new frame
        self.circles       = []
        self.drawing       = False
        self.center        = None
        self.current_mouse = None
        self.frame_orig    = frame.copy()
        self.dirty         = True
        self._init_zoom()

    def mouse_callback(self, event, x, y, flags, param):
        # x and y come in as display pixels convert to original frame pixels
        self.display_mouse = (x, y)
        ox, oy = self._display_to_orig(x, y)

        # Let the pan handler consume right-click events before we do anything else
        if self._handle_pan(event, x, y):
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.center  = (ox, oy)
            self.current_mouse = (ox, oy)
            self.dirty   = True

        elif event == cv2.EVENT_MOUSEMOVE:
            self.current_mouse = (ox, oy)
            if self.drawing:
                self.dirty = True  # only need to redraw while dragging

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.center:
                radius = int(np.hypot(ox - self.center[0], oy - self.center[1]))
                # Ignore accidental tiny clicks (less than 3 px radius)
                if radius > 3:
                    self.circles.append((self.center[0], self.center[1], radius))
            self.drawing = False
            self.center  = None
            self.dirty   = True

    def undo(self):
        if self.circles:
            self.circles.pop()
            self.dirty = True

    def count(self):
        return len(self.circles)

    def render(self, disp_w: int, disp_h: int) -> np.ndarray:
        frame = self._base_frame(disp_w, disp_h)

        # Draw all finished circles in green
        for cx, cy, r in self.circles:
            dcx, dcy = self._orig_to_display(cx, cy)
            dr = max(int(r * self.zoom), 1)  # scale the radius to match the current zoom
            cv2.circle(frame, (int(dcx), int(dcy)), dr, (0, 220, 0), 2)
            cv2.circle(frame, (int(dcx), int(dcy)), 3, (0, 220, 0), -1)  # small dot at center

        # Draw the circle currently being dragged in orange as a live preview
        if self.drawing and self.center and self.current_mouse:
            r_orig = np.hypot(
                self.current_mouse[0] - self.center[0],
                self.current_mouse[1] - self.center[1],
            )
            dcx, dcy = self._orig_to_display(*self.center)
            dr = max(int(r_orig * self.zoom), 1)
            cv2.circle(frame, (int(dcx), int(dcy)), dr, (0, 165, 255), 2)
            cv2.circle(frame, (int(dcx), int(dcy)), 3, (0, 165, 255), -1)

        self.dirty = False
        return frame

# Hexagon annotator
def _hexagon_diameter(vertices) -> float:
    """Average of the 3 long diagonals connecting opposite vertex pairs.

    A hexagon has 3 pairs of opposite vertices: 0 and 3, 1 and 4, 2 and 5.
    Averaging the lengths of these three diagonals gives us the cell diameter.
    """
    pts = np.array(vertices, dtype=float)
    return float(np.mean([np.linalg.norm(pts[i] - pts[i + 3]) for i in range(3)]))


class HexagonAnnotator(_ZoomPanMixin):
    def __init__(self):
        # Each finished hexagon is a list of 6 (x, y) tuples in original frame pixels
        self.hexagons    = []
        # Vertices placed so far for the hexagon currently being drawn
        self.current_pts = []
        self.current_mouse = None  # used to draw the preview line to the next vertex
        self.frame_orig  = None
        self.dirty       = True
        self._init_zoom()

    def reset(self, frame: np.ndarray):
        self.hexagons      = []
        self.current_pts   = []
        self.current_mouse = None
        self.frame_orig    = frame.copy()
        self.dirty         = True
        self._init_zoom()

    def mouse_callback(self, event, x, y, flags, param):
        self.display_mouse = (x, y)
        ox, oy = self._display_to_orig(x, y)

        if self._handle_pan(event, x, y):
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # Keep track of where the mouse is so we can draw the preview edge
            self.current_mouse = (ox, oy)
            if self.current_pts:  # only redraw if a hexagon is drawn
                self.dirty = True

        elif event == cv2.EVENT_LBUTTONDOWN:
            # Each click adds one vertex to the current hexagon
            self.current_pts.append((ox, oy))
            self.dirty = True
            # After the 6th vertex the hexagon is completed. Can be saved and start fresh. 
            if len(self.current_pts) == HEX_VERTICES:
                self.hexagons.append(list(self.current_pts))
                self.current_pts = []

    def undo(self):
        if self.current_pts:
            # Remove the last placed vertex if a hexagon is in progress
            self.current_pts.pop()
            self.dirty = True
        elif self.hexagons:
            # If no hexagon is in progress, remove the last completed one
            self.hexagons.pop()
            self.dirty = True

    def count(self):
        return len(self.hexagons)

    def render(self, disp_w: int, disp_h: int) -> np.ndarray:
        frame = self._base_frame(disp_w, disp_h)

        # Helper to convert a single point from original coords to display coords
        def to_d(pt):
            dx, dy = self._orig_to_display(*pt)
            return (int(dx), int(dy))

        # Draw all finished hexagons in green
        for hex_pts in self.hexagons:
            dpts = np.array([to_d(p) for p in hex_pts], dtype=np.int32)
            cv2.polylines(frame, [dpts], isClosed=True, color=(0, 220, 0), thickness=2)
            for dp in dpts:
                cv2.circle(frame, tuple(dp), 4, (0, 220, 0), -1)  # dot at each vertex

        # Draw the hexagon currently being placed in orange
        if self.current_pts:
            dpts = [to_d(p) for p in self.current_pts]

            # Connect the vertices placed so far with lines
            for i in range(len(dpts) - 1):
                cv2.line(frame, dpts[i], dpts[i + 1], (0, 165, 255), 2)

            # Draw a dot at each placed vertex
            for dp in dpts:
                cv2.circle(frame, dp, 4, (0, 165, 255), -1)

            # Draw a faint preview line from the last vertex to the current mouse position
            if self.current_mouse:
                cv2.line(frame, dpts[-1], to_d(self.current_mouse), (0, 165, 255), 1)

            # Show how many vertices have been placed, fixed in the top-left corner
            # so it does not overlap with the drawing area
            remaining = HEX_VERTICES - len(self.current_pts)
            cv2.putText(frame,
                        f"vertex {len(self.current_pts)}/{HEX_VERTICES}  ({remaining} left)",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)

        self.dirty = False
        return frame


# Annotation loop shared by both modes
def annotate_frame(frame: np.ndarray, window_name: str, label: str, shape: str = "circle"):
    """
    Show one video frame and let the user annotate it.

    shape: 'circle' or 'hexagon'
    Returns (annotations, quit_flag, skip_flag).
      circle  -> annotations = [(cx, cy, radius), ...]
      hexagon -> annotations = [[(x,y) x6], ...]
    skip_flag means the frame had too few visible cells -- caller should
    discard annotations and show a different frame from the same video instead.
    """
    # Pick the right annotator and instruction text for the chosen shape mode
    if shape == "hexagon":
        annotator   = HexagonAnnotator()
        shape_label = "click: place vertex (6 -> auto-close)"
    else:
        annotator   = CircleAnnotator()
        shape_label = "L-drag: draw circle"

    annotator.reset(frame)

    # Use the frames original resolution no downscaling
    fh, fw = frame.shape[:2]
    disp_w, disp_h = fw, fh

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w, disp_h)
    cv2.setMouseCallback(window_name, annotator.mouse_callback)

    bar_h     = 55
    info_base = (
        f"{shape_label}   |   R-drag/arrows: pan   |   "
        f"+/-: zoom   |   B: reset   |   U: undo   |   D: done   |   "
        f"S: skip frame (too few cells)   |   Q: quit"
    )

    while True:
        # Only redraw the window when something actually changed.
        # This avoids doing expensive image operations on every loop tick.
        if annotator.dirty:
            display = annotator.render(disp_w, disp_h)

            # Draw a semi-transparent black bar at the bottom for the instructions
            overlay = display.copy()
            cv2.rectangle(overlay, (0, disp_h - bar_h), (disp_w, disp_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

            # Top-left label: shape mode, subfolder/video/frame, and zoom level if zoomed
            zoom_str   = f"  [{annotator.zoom:.1f}x]" if annotator.zoom != 1.0 else ""
            shape_name = shape[0].upper() + shape[1:]
            count_word = "Hexagons" if shape == "hexagon" else "Circles"

            cv2.putText(display, f"[{shape_name}] {label}{zoom_str}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, f"{count_word}: {annotator.count()}   |   {info_base}",
                        (10, disp_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

            cv2.imshow(window_name, display)

        # waitKeyEx captures extended keys (arrow keys, function keys) reliably.
        # Regular waitKey can miss them on macOS, so this is a workaround
        key = cv2.waitKeyEx(20)
        if key == -1:
            continue  # nothing was pressed

        # Check arrow keys first their codes are large and platform-specific
        if key in ARROW_UP:
            annotator.pan_step(0, -1)
        elif key in ARROW_DOWN:
            annotator.pan_step(0, 1)
        elif key in ARROW_LEFT:
            annotator.pan_step(-1, 0)
        elif key in ARROW_RIGHT:
            annotator.pan_step(1, 0)
        else:
            # Regular ASCII keys mask to get just the character byte
            char = key & 0xFF
            if char in (ord('+'), ord('=')):   # '=' is '+' no shift on most keyboards I guess
                annotator.zoom_in()
            elif char == ord('-'):
                annotator.zoom_out()
            elif char in (ord('b'), ord('B')):
                annotator.zoom_reset()
            elif char in (ord('u'), ord('U')):
                annotator.undo()
            elif char in (ord('d'), ord('D')):
                break
            elif char in (ord('s'), ord('S')):
                # Too few cells visible here -- discard and let the caller
                # show a different frame from the same video instead.
                return [], False, True
            elif char in (ord('q'), ord('Q')):
                # Quit early still return what has been annotated so far
                result = annotator.hexagons if shape == "hexagon" else annotator.circles
                return result, True, False

    result = annotator.hexagons if shape == "hexagon" else annotator.circles
    return result, False, False


# Output summary and CSV

def compute_summary(all_annotations: dict, shape: str) -> dict:
    """
    For each subfolder, returns a dict keyed by resolution tier (width, height):
        { "avg_diameter_px", "n_source_videos", "measured" }

    Each annotated video is measured once, at its own (highest available)
    resolution, then that measurement is projected to every sibling resolution
    present for the same source recording using the exact width ratio -- since
    sibling files are pixel-exact rescales of one another (same duration/fps/
    frame count, verified separately). "measured" is True only for the tier(s)
    actually clicked on; every other tier's value is a derived projection.
    """
    summary = {}
    for subfolder, entries in all_annotations.items():
        per_target = defaultdict(list)   # target_res -> [projected diameters]
        measured_res = set()

        for _, _, annotations, res, sibling_res in entries:
            if shape == "hexagon":
                diam_list = [_hexagon_diameter(p) for p in annotations if len(p) == HEX_VERTICES]
            else:
                diam_list = [r * 2 for _, _, r in annotations]  # diameter = 2 * radius
            if not diam_list or res is None:
                continue

            avg_native = float(np.mean(diam_list))
            measured_res.add(res)
            per_target[res].append(avg_native)
            for sib in sibling_res:
                if sib == res:
                    continue
                ratio = sib[0] / res[0]
                per_target[sib].append(avg_native * ratio)

        res_summary = {}
        for target_res, vals in sorted(per_target.items(), key=lambda kv: -kv[0][0] * kv[0][1]):
            res_summary[target_res] = {
                "avg_diameter_px": float(np.mean(vals)),
                "n_source_videos": len(vals),
                "measured": target_res in measured_res,
            }
        summary[subfolder] = res_summary
    return summary


def save_annotations_csv(all_annotations: dict, shape: str, output_path: Path):
    # Write raw annotations to CSV. The columns differ between circle and hexagon modes.
    # Resolution columns record the file the annotation was actually made on --
    # projections to other tiers are computed separately, see save_projection_csv.
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # hexagon mode store
        if shape == "hexagon":
            # Store all 6 vertex coordinates and the computed diameter per row
            writer.writerow([
                "lab", "video", "res_w", "res_h", "frame_idx", "hex_idx",
                "x1","y1","x2","y2","x3","y3","x4","y4","x5","y5","x6","y6",
                "diameter_px",
            ])
            for subfolder, entries in all_annotations.items():
                for video, frame_idx, hexagons, res, _sibling_res in entries:
                    res_w, res_h = res if res else ("", "")
                    for i, pts in enumerate(hexagons):
                        flat = [c for p in pts for c in p]  # [(x,y),...] -> [x,y,x,y,...]
                        diam = _hexagon_diameter(pts)
                        writer.writerow(
                            [subfolder, Path(video).name, res_w, res_h, frame_idx, i] + flat + [f"{diam:.2f}"]
                        )
        else:
            writer.writerow([
                "lab", "video", "res_w", "res_h", "frame_idx",
                "circle_idx", "center_x", "center_y", "radius_px", "diameter_px",
            ])
            for subfolder, entries in all_annotations.items():
                for video, frame_idx, circles, res, _sibling_res in entries:
                    res_w, res_h = res if res else ("", "")
                    for i, (cx, cy, r) in enumerate(circles):
                        writer.writerow(
                            [subfolder, Path(video).name, res_w, res_h, frame_idx, i, cx, cy, r, r * 2]
                        )


# A worker bee's body is roughly as wide as a comb cell is across, and about
# twice as long -- see docs.txt for the caveats on this approximation.
BEE_WIDTH_RATIO = 1.0
BEE_LENGTH_RATIO = 2.0


def save_projection_csv(summary: dict, output_path: Path):
    """Write the per-resolution-tier projected summary (the actual deliverable
    for deciding model grid scales / postprocessing thresholds) to its own CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "lab", "res_w", "res_h", "avg_diameter_px",
            "bee_width_px", "bee_length_px", "n_source_videos", "measured",
        ])
        for subfolder, res_summary in summary.items():
            for (w, h), stats in res_summary.items():
                diam = stats["avg_diameter_px"]
                writer.writerow([
                    subfolder, w, h, f"{diam:.2f}",
                    f"{diam * BEE_WIDTH_RATIO:.2f}", f"{diam * BEE_LENGTH_RATIO:.2f}",
                    stats["n_source_videos"], stats["measured"],
                ])
def main():
    # Load config from the same folder as this script
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(config_path)

    # Resolve data_path relative to this script's location
    data_root  = (Path(__file__).parent / config["data_path"]).resolve()
    subfolders = config["subfolders"]
    n_videos   = config["n_videos"]
    n_frames   = config["n_frames"]
    shape      = config.get("shape", "circle").lower()
    save_csv   = config.get("save_csv", True)

    assert shape in ("circle", "hexagon"), \
        f"shape must be 'circle' or 'hexagon', got '{shape}'"

    # all_annotations maps lab name (berlin/nieh/sharoni) -> list of
    # (video_path, frame_idx, annotations, resolution, sibling_resolutions)
    all_annotations = defaultdict(list)
    window_name = "Comb Cell Annotator"

    print(f"Mode: {shape.upper()}")

    # Collect every video across all configured folders, then group into one
    # representative per source recording (highest-resolution copy -- see
    # group_by_source). We bucket by lab (parsed from the filename) rather than
    # by folder, since e.g. data/videos holds all three labs mixed together.
    representatives = []
    for subfolder_cfg in subfolders:
        subfolder = subfolder_cfg["name"] if isinstance(subfolder_cfg, dict) else subfolder_cfg
        folder_path = data_root / subfolder
        if not folder_path.exists():
            print(f"[Skip] Not found: {folder_path}")
            continue
        videos = get_video_files(folder_path)
        if not videos:
            print(f"[Skip] No videos in: {folder_path}")
            continue
        representatives.extend(group_by_source(videos))

    by_lab = defaultdict(list)
    for rep in representatives:
        video_path, _res, _sib = rep
        by_lab[lab_source(video_path.name)].append(rep)

    # Videos are already the preprocessed training clips, assumed correctly
    # oriented already -- rotate manually in the viewer (B/+/- keys) if not.
    rotation = 0

    quit_early = False
    for lab, lab_reps in sorted(by_lab.items()):
        sampled = random.sample(lab_reps, min(n_videos, len(lab_reps)))
        print(f"\n[{lab}] {len(sampled)} video(s) selected "
              f"(out of {len(lab_reps)} distinct source recordings)")

        for video_path, res, sibling_res in sampled:
            if res:
                print(f"  {video_path.name}: annotating at {res[0]}x{res[1]}"
                      + (f" (projects to {len(sibling_res) - 1} other tier(s))" if len(sibling_res) > 1 else ""))
            total = get_total_frames(video_path)
            if total == 0:
                print(f"  [Skip] Could not read {video_path.name}")
                continue

            indices = get_frame_indices(total, n_frames)
            print(f"  {video_path.name} ({total} frames) -> frames {indices}")

            for frame_idx in indices:
                tried_indices = set()
                while True:
                    tried_indices.add(frame_idx)
                    try:
                        frame = extract_frame(video_path, frame_idx)
                    except ValueError:
                        print(f"  [Skip] {video_path.name}: could not read frame {frame_idx}")
                        if "downsample" in video_path.name.lower():
                            # For some reason opencv cannot decode the downsampled versions of the
                            # Nieh Lab, but original works. If it doesnt work and original exist use
                            # original.
                            original_name = video_path.name.lower().replace("downsample", "original")
                            original_path = video_path.parent / original_name
                            if original_path.exists():
                                print(f"      Trying {original_path.name} instead.")
                                video_path = original_path
                                try:
                                    frame = extract_frame(video_path, frame_idx)
                                except ValueError:
                                    print(f"  [Skip] {original_path.name}: could not read frame {frame_idx} either")
                                    break
                            else:
                                print(f"      No original found. Skipping.")
                                break
                        else:
                            break
                    if rotation:
                        frame = rotate_frame(frame, rotation)
                    res_str = f" ({res[0]}x{res[1]})" if res else ""
                    label = f"{lab}  |  {video_path.name}{res_str}  |  frame {frame_idx}"
                    annotations, quit_early, skip_frame = annotate_frame(frame, window_name, label, shape)

                    if skip_frame and not quit_early:
                        new_idx = pick_frame_index(total, tried_indices)
                        print(f"    frame {frame_idx}: too few cells, trying frame {new_idx} instead")
                        frame_idx = new_idx
                        continue
                    break

                all_annotations[lab].append((str(video_path), frame_idx, annotations, res, sibling_res))
                print(f"    frame {frame_idx}: {len(annotations)} {shape}(s)")
                if quit_early:
                    break
            if quit_early:
                print("\n[Q] Quit early. Computing summary from annotated frames.")
                break
        if quit_early:
            break

    cv2.destroyAllWindows()

    # print results to term
    summary = compute_summary(all_annotations, shape)
    print("\n" + "=" * 60)
    print(f"Annotation Summary  (shape={shape})")
    print("=" * 60)
    for subfolder, res_summary in summary.items():
        if not res_summary:
            print(f"\n{subfolder}: nothing annotated")
            continue
        print(f"\n{subfolder}:")
        for (w, h), stats in res_summary.items():
            tag = "measured" if stats["measured"] else "projected"
            print(f"  {w:>5}x{h:<5} : {stats['avg_diameter_px']:6.1f} px "
                  f"({tag}, from {stats['n_source_videos']} source video(s))")

    if save_csv:
        # Fixed filenames, overwritten on every run -- only the latest
        # annotation session's results are kept, not a history of runs.
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        csv_path  = output_dir / f"annotations_{shape}.csv"
        save_annotations_csv(all_annotations, shape, csv_path)
        print(f"\nAnnotations saved to: {csv_path}")

        proj_path = output_dir / "projected_summary.csv"
        save_projection_csv(summary, proj_path)
        print(f"Per-resolution summary saved to: {proj_path}")


if __name__ == "__main__":
    main()
