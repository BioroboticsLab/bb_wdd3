"""
Tests for cross-window prediction averaging.

The averaging approach works as follows:
  - Group windows by (video_name, crop_origin) to identify temporal overlap groups
  - Within each group, for each grid cell (i, j, k), collect detections across windows
  - For cells with NO detection in a given window, count that as confidence=0
  - Average confidence (and other attributes) across all windows in the group
  - Output one averaged prediction per cell that had at least one detection

This suppresses sporadic false positives (detected in 1/5 windows → conf/5)
while preserving consistent detections (detected in 5/5 → conf stays the same).
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
        """If the same cell fires in ALL overlapping windows, confidence is averaged
        (which preserves it if confidences are similar)."""
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
        # Confidence = mean(0.9, 0.8, 0.7) = 0.8
        assert abs(avg['confidence'] - 0.8) < 1e-6
        # Position should be averaged too
        assert abs(avg['position'][0] - 101.0) < 1e-6
        assert abs(avg['position'][1] - 80.0) < 1e-6

    def test_sporadic_detection_suppressed(self):
        """If a cell fires in only 1 of 3 overlapping windows, its confidence
        should be divided by 3 (diluted by zeros from non-detecting windows)."""
        per_window_preds = [
            [_pred(0.9, (100, 80), grid_cell=(2, 2, 0), window_start=0,
                   temporal=(2, 14))],
            [],  # no detection
            [],  # no detection
        ]
        crop_origins = [(50, 30), (50, 30), (50, 30)]
        video_names = ['vid_A', 'vid_A', 'vid_A']

        result = average_overlapping_predictions(
            per_window_preds, crop_origins, video_names
        )

        all_preds = [p for preds in result for p in preds]
        assert len(all_preds) == 1

        avg = all_preds[0]
        # Confidence = (0.9 + 0 + 0) / 3 = 0.3
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
                _pred(0.9, (100, 80), grid_cell=(2, 2, 0), window_start=0),
                _pred(0.6, (50, 40), grid_cell=(1, 1, 0), window_start=0),
            ],
            [
                _pred(0.8, (102, 79), grid_cell=(2, 2, 0), window_start=4),
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

        # Cell (2,2,0): detected in both → (0.9 + 0.8) / 2 = 0.85
        assert abs(by_cell[(2, 2, 0)]['confidence'] - 0.85) < 1e-6

        # Cell (1,1,0): detected in 1 of 2 → 0.6 / 2 = 0.3
        assert abs(by_cell[(1, 1, 0)]['confidence'] - 0.3) < 1e-6

    def test_direction_averaged_as_unit_vector(self):
        """Direction vectors should be averaged and re-normalized to unit length."""
        # Two windows detect similar directions
        per_window_preds = [
            [_pred(0.9, (100, 80), direction=(1.0, 0.0),
                   grid_cell=(0, 0, 0), window_start=0)],
            [_pred(0.8, (102, 79), direction=(0.0, 1.0),
                   grid_cell=(0, 0, 0), window_start=4)],
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

    def test_n_detections_tracked(self):
        """The output should track how many windows contributed a detection."""
        per_window_preds = [
            [_pred(0.9, (100, 80), grid_cell=(0, 0, 0), window_start=0)],
            [_pred(0.8, (102, 79), grid_cell=(0, 0, 0), window_start=4)],
            [],  # no detection
        ]
        crop_origins = [(50, 30), (50, 30), (50, 30)]
        video_names = ['vid_A', 'vid_A', 'vid_A']

        result = average_overlapping_predictions(
            per_window_preds, crop_origins, video_names
        )

        all_preds = [p for preds in result for p in preds]
        assert all_preds[0]['n_detections'] == 2
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
