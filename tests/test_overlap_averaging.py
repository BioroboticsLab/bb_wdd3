"""
Tests for cross-window prediction averaging.

The averaging approach works as follows:
  - Group windows by (video_name, crop_origin) to identify spatial overlap groups
  - Within each group, for each grid cell (i, j, k), collect detections across windows
  - The confidence denominator is the number of windows whose temporal range
    OVERLAPS the detection's temporal extent (not the total group size)
  - This makes the function work for both:
    - Annotation-based: small groups of 2-6 windows → similar to old behavior
    - Full-frame: all windows share (0,0) origin but only ~2 temporally overlap
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.dance_eval import average_overlapping_predictions


def _pred(conf, pos, direction=(1.0, 0.0), temporal=(10, 20),
          grid_cell=(0, 0, 0), window_start=0):
    """Helper to create a prediction dict."""
    return {
        'confidence': conf,
        'position': list(pos),
        'direction': list(direction),
        'temporal_offsets': list(temporal),
        'grid_cell': list(grid_cell),
        'window_start': window_start,
    }


class TestAverageOverlapping:
    """Test the cross-window grid-cell averaging function."""

    def test_consistent_detection_preserves_confidence(self):
        """If the same cell fires in ALL overlapping windows, confidence is
        averaged across the overlapping windows (preserves it if similar)."""
        # 3 overlapping windows, same crop origin, all detect at cell (2,2,0)
        per_window_preds = [
            [_pred(0.9, (100, 80), grid_cell=(2, 2, 0), window_start=0,
                   temporal=(2, 14))],
            [_pred(0.8, (102, 79), grid_cell=(2, 2, 0), window_start=4,
                   temporal=(6, 18))],
            [_pred(0.7, (101, 81), grid_cell=(2, 2, 0), window_start=8,
                   temporal=(10, 22))],
        ]
        crop_origins = [(50, 30), (50, 30), (50, 30)]
        video_names = ['vid_A', 'vid_A', 'vid_A']

        result = average_overlapping_predictions(
            per_window_preds, crop_origins, video_names
        )

        # Should produce exactly 1 averaged prediction
        all_preds = [p for preds in result for p in preds]
        assert len(all_preds) == 1

        avg = all_preds[0]
        # All 3 windows overlap the detection's extent [2, 22],
        # so denominator = 3 → mean(0.9, 0.8, 0.7) = 0.8
        assert abs(avg['confidence'] - 0.8) < 1e-6
        # Position should be averaged too
        assert abs(avg['position'][0] - 101.0) < 1e-6
        assert abs(avg['position'][1] - 80.0) < 1e-6

    def test_sporadic_detection_suppressed(self):
        """If a cell fires in only 1 of 3 overlapping windows and all windows
        temporally overlap, confidence is divided by 3."""
        per_window_preds = [
            [_pred(0.9, (100, 80), grid_cell=(2, 2, 0), window_start=0,
                   temporal=(2, 14))],
            [],  # no detection
            [],  # no detection
        ]
        crop_origins = [(50, 30), (50, 30), (50, 30)]
        video_names = ['vid_A', 'vid_A', 'vid_A']
        # Provide explicit window ranges so all 3 overlap the detection's [2, 14]
        window_ranges = [(0, 16), (4, 20), (8, 24)]

        result = average_overlapping_predictions(
            per_window_preds, crop_origins, video_names,
            window_ranges=window_ranges,
        )

        all_preds = [p for preds in result for p in preds]
        assert len(all_preds) == 1

        avg = all_preds[0]
        # All 3 windows overlap [2, 14], so denom = 3 → 0.9 / 3 = 0.3
        assert abs(avg['confidence'] - 0.3) < 1e-6

    def test_sporadic_detection_without_window_ranges(self):
        """Without explicit window_ranges, empty windows get None range and
        are conservatively counted as overlapping (matches old behavior)."""
        per_window_preds = [
            [_pred(0.9, (100, 80), grid_cell=(2, 2, 0), window_start=0,
                   temporal=(2, 14))],
            [],  # no detection → range inferred as None → counted as overlapping
            [],  # no detection → range inferred as None → counted as overlapping
        ]
        crop_origins = [(50, 30), (50, 30), (50, 30)]
        video_names = ['vid_A', 'vid_A', 'vid_A']

        result = average_overlapping_predictions(
            per_window_preds, crop_origins, video_names
        )

        all_preds = [p for preds in result for p in preds]
        assert len(all_preds) == 1

        avg = all_preds[0]
        # None-range windows counted as overlapping → denom = 3 → 0.9 / 3 = 0.3
        assert abs(avg['confidence'] - 0.3) < 1e-6

    def test_different_crop_origins_not_averaged(self):
        """Windows with different crop origins should NOT be averaged together,
        since they view different parts of the frame."""
        per_window_preds = [
            [_pred(0.9, (100, 80), grid_cell=(2, 2, 0), window_start=0)],
            [_pred(0.8, (200, 150), grid_cell=(2, 2, 0), window_start=0)],
        ]
        crop_origins = [(50, 30), (180, 120)]  # different origins
        video_names = ['vid_A', 'vid_A']

        result = average_overlapping_predictions(
            per_window_preds, crop_origins, video_names
        )

        all_preds = [p for preds in result for p in preds]
        # Should produce 2 independent predictions (no averaging across groups)
        assert len(all_preds) == 2
        confs = sorted([p['confidence'] for p in all_preds])
        assert abs(confs[0] - 0.8) < 1e-6  # each keeps its confidence / 1
        assert abs(confs[1] - 0.9) < 1e-6

    def test_different_videos_not_averaged(self):
        """Windows from different videos should not be averaged together."""
        per_window_preds = [
            [_pred(0.9, (100, 80), grid_cell=(2, 2, 0), window_start=0)],
            [_pred(0.8, (100, 80), grid_cell=(2, 2, 0), window_start=0)],
        ]
        crop_origins = [(50, 30), (50, 30)]
        video_names = ['vid_A', 'vid_B']  # different videos

        result = average_overlapping_predictions(
            per_window_preds, crop_origins, video_names
        )

        all_preds = [p for preds in result for p in preds]
        assert len(all_preds) == 2
        # Each stays at its own confidence / 1 window
        confs = sorted([p['confidence'] for p in all_preds])
        assert abs(confs[0] - 0.8) < 1e-6
        assert abs(confs[1] - 0.9) < 1e-6

    def test_different_grid_cells_stay_separate(self):
        """Detections in different grid cells within the same overlap group
        should be averaged independently."""
        per_window_preds = [
            [
                _pred(0.9, (100, 80), grid_cell=(2, 2, 0), window_start=0,
                      temporal=(2, 14)),
                _pred(0.6, (50, 40), grid_cell=(1, 1, 0), window_start=0,
                      temporal=(2, 14)),
            ],
            [
                _pred(0.8, (102, 79), grid_cell=(2, 2, 0), window_start=4,
                      temporal=(6, 18)),
                # cell (1,1,0) NOT detected in window 2
            ],
        ]
        crop_origins = [(50, 30), (50, 30)]
        video_names = ['vid_A', 'vid_A']

        result = average_overlapping_predictions(
            per_window_preds, crop_origins, video_names
        )

        all_preds = [p for preds in result for p in preds]
        assert len(all_preds) == 2

        # Sort by grid cell for deterministic checking
        by_cell = {tuple(p['grid_cell']): p for p in all_preds}

        # Cell (2,2,0): detected in both, both windows overlap → (0.9 + 0.8) / 2 = 0.85
        assert abs(by_cell[(2, 2, 0)]['confidence'] - 0.85) < 1e-6

        # Cell (1,1,0): detected in 1 window. The detection [2, 14] overlaps
        # both windows (window 1: [2,14], window 2: [6,18]), so denom = 2
        # → 0.6 / 2 = 0.3
        assert abs(by_cell[(1, 1, 0)]['confidence'] - 0.3) < 1e-6

    def test_direction_averaged_as_unit_vector(self):
        """Direction vectors should be averaged and re-normalized to unit length."""
        # Two windows detect similar directions
        per_window_preds = [
            [_pred(0.9, (100, 80), direction=(1.0, 0.0),
                   grid_cell=(0, 0, 0), window_start=0, temporal=(0, 16))],
            [_pred(0.8, (102, 79), direction=(0.0, 1.0),
                   grid_cell=(0, 0, 0), window_start=4, temporal=(4, 20))],
        ]
        crop_origins = [(50, 30), (50, 30)]
        video_names = ['vid_A', 'vid_A']

        result = average_overlapping_predictions(
            per_window_preds, crop_origins, video_names
        )

        all_preds = [p for preds in result for p in preds]
        assert len(all_preds) == 1

        d = all_preds[0]['direction']
        # Mean of (1,0) and (0,1) → (0.5, 0.5) → normalized to (√2/2, √2/2)
        expected = 1.0 / np.sqrt(2)
        assert abs(d[0] - expected) < 1e-4
        assert abs(d[1] - expected) < 1e-4

    def test_temporal_offsets_merged(self):
        """Temporal offsets should span the union of all detections
        (earliest start, latest end)."""
        per_window_preds = [
            [_pred(0.9, (100, 80), temporal=(5, 15),
                   grid_cell=(0, 0, 0), window_start=0)],
            [_pred(0.8, (102, 79), temporal=(9, 19),
                   grid_cell=(0, 0, 0), window_start=4)],
            [_pred(0.7, (101, 81), temporal=(13, 22),
                   grid_cell=(0, 0, 0), window_start=8)],
        ]
        crop_origins = [(50, 30), (50, 30), (50, 30)]
        video_names = ['vid_A', 'vid_A', 'vid_A']

        result = average_overlapping_predictions(
            per_window_preds, crop_origins, video_names
        )

        all_preds = [p for preds in result for p in preds]
        t = all_preds[0]['temporal_offsets']
        # Union: min(5,9,13)=5, max(15,19,22)=22
        assert t[0] == 5
        assert t[1] == 22

    def test_n_detections_and_n_overlapping_tracked(self):
        """The output should track how many windows detected and how many overlapped."""
        per_window_preds = [
            [_pred(0.9, (100, 80), grid_cell=(0, 0, 0), window_start=0,
                   temporal=(0, 16))],
            [_pred(0.8, (102, 79), grid_cell=(0, 0, 0), window_start=4,
                   temporal=(4, 20))],
            [],  # no detection
        ]
        crop_origins = [(50, 30), (50, 30), (50, 30)]
        video_names = ['vid_A', 'vid_A', 'vid_A']
        # Explicit ranges: window 3 covers [8, 24], overlaps [0, 20]
        window_ranges = [(0, 16), (4, 20), (8, 24)]

        result = average_overlapping_predictions(
            per_window_preds, crop_origins, video_names,
            window_ranges=window_ranges,
        )

        all_preds = [p for preds in result for p in preds]
        assert all_preds[0]['n_detections'] == 2
        assert all_preds[0]['n_overlapping'] == 3  # all 3 windows overlap [0, 20]
        assert all_preds[0]['n_windows'] == 3

    def test_empty_input(self):
        """No windows → no output."""
        result = average_overlapping_predictions([], [], [])
        assert result == []

    def test_single_window_passthrough(self):
        """A single window with no overlap should pass through unchanged."""
        per_window_preds = [
            [_pred(0.9, (100, 80), grid_cell=(2, 2, 0), window_start=0)],
        ]
        crop_origins = [(50, 30)]
        video_names = ['vid_A']

        result = average_overlapping_predictions(
            per_window_preds, crop_origins, video_names
        )

        all_preds = [p for preds in result for p in preds]
        assert len(all_preds) == 1
        # Confidence = 0.9 / 1 window = 0.9
        assert abs(all_preds[0]['confidence'] - 0.9) < 1e-6

    # ── Full-frame scenario tests ────────────────────────────────────────

    def test_full_frame_temporal_overlap_denominator(self):
        """Full-frame: all windows share (0,0) origin. Only temporally
        overlapping windows should affect the denominator, not the total
        number of windows in the video."""
        # 10 windows, all share (0, 0), stride=8, window_size=16
        # Detection at frames [32, 44] → overlaps windows starting at 24 and 32
        n_total = 10
        per_window_preds = [[] for _ in range(n_total)]
        # Place detection in window 4 (start=32)
        per_window_preds[4] = [
            _pred(0.9, (100, 80), grid_cell=(5, 5, 0), temporal=(32, 44))
        ]

        crop_origins = [(0, 0)] * n_total
        video_names = ['vid_A'] * n_total
        # Explicit stride-8, window-16 ranges
        window_ranges = [(i * 8, i * 8 + 16) for i in range(n_total)]

        result = average_overlapping_predictions(
            per_window_preds, crop_origins, video_names,
            window_ranges=window_ranges,
        )

        all_preds = [p for preds in result for p in preds]
        assert len(all_preds) == 1

        avg = all_preds[0]
        # Detection [32, 44] overlaps windows:
        #   w2=[16,32], w3=[24,40], w4=[32,48], w5=[40,56] → 4 windows overlap
        # (w2 touches at boundary: 16 <= 44 and 32 >= 32)
        # Confidence = 0.9 / 4 = 0.225
        assert avg['n_overlapping'] == 4
        assert abs(avg['confidence'] - 0.225) < 1e-6

    def test_full_frame_consistent_detection_across_overlapping_windows(self):
        """Full-frame: detection in 2 of 3 temporally overlapping windows
        yields confidence averaged over 3 (not over 10 total windows)."""
        n_total = 10
        per_window_preds = [[] for _ in range(n_total)]
        # Same cell fires in windows 3 and 4
        per_window_preds[3] = [
            _pred(0.8, (100, 80), grid_cell=(5, 5, 0), temporal=(28, 38))
        ]
        per_window_preds[4] = [
            _pred(0.9, (100, 80), grid_cell=(5, 5, 0), temporal=(34, 44))
        ]

        crop_origins = [(0, 0)] * n_total
        video_names = ['vid_A'] * n_total
        window_ranges = [(i * 8, i * 8 + 16) for i in range(n_total)]

        result = average_overlapping_predictions(
            per_window_preds, crop_origins, video_names,
            window_ranges=window_ranges,
        )

        all_preds = [p for preds in result for p in preds]
        assert len(all_preds) == 1

        avg = all_preds[0]
        # Detection union [28, 44] overlaps:
        #   w2=[16,32], w3=[24,40], w4=[32,48], w5=[40,56] → 4 windows
        assert avg['n_overlapping'] == 4
        # Confidence = (0.8 + 0.9) / 4 = 0.425
        assert abs(avg['confidence'] - 0.425) < 1e-6

    def test_full_frame_non_overlapping_windows_excluded(self):
        """Distant temporal windows should NOT inflate the denominator."""
        n_total = 10
        per_window_preds = [[] for _ in range(n_total)]
        # Detection only at window 0 (frames 0-16)
        per_window_preds[0] = [
            _pred(0.9, (100, 80), grid_cell=(5, 5, 0), temporal=(2, 14))
        ]

        crop_origins = [(0, 0)] * n_total
        video_names = ['vid_A'] * n_total
        window_ranges = [(i * 8, i * 8 + 16) for i in range(n_total)]

        result = average_overlapping_predictions(
            per_window_preds, crop_origins, video_names,
            window_ranges=window_ranges,
        )

        all_preds = [p for preds in result for p in preds]
        assert len(all_preds) == 1

        avg = all_preds[0]
        # Detection [2, 14] overlaps w0=[0,16] and w1=[8,24] → 2
        assert avg['n_overlapping'] == 2
        # NOT divided by 10! → 0.9 / 2 = 0.45
        assert abs(avg['confidence'] - 0.45) < 1e-6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
