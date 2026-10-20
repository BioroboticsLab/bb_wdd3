"""Physical units for the dance-level pipeline.

Distances are millimetres and time is seconds. Both used to be carried in
units that depend on the recording -- fractions of frame width and frame
counts -- which meant one configured number meant different physical things
per lab, and a caller that forgot `video_fps` silently got a 15 fps answer.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.dance_eval import (
    FRAME_WIDTH_MM,
    frame_width_mm,
    norm_distance_for_mm,
    _fps_for,
)

# A comb cell is ~5 mm, so these follow from the measured cell diameters in
# src/data/combcell_annotator/output/projected_summary.csv:
#   berlin  2304 px wide, cell 25.80 px  ->  2304 / (25.80/5) = 446.5 mm
#   nieh     960 px wide, cell 37.20 px  ->   960 / (37.20/5) = 129.0 mm
#   sharoni 1224 px wide, cell 20.18 px  ->  1224 / (20.18/5) = 303.3 mm
EXPECTED = {'berlin': 446.5, 'nieh': 129.0, 'sharoni': 303.3}


class TestFrameWidthMM:
    def test_measured_values_match_the_comb_cell_measurements(self):
        for lab, mm in EXPECTED.items():
            assert FRAME_WIDTH_MM[lab] == pytest.approx(mm, abs=0.5)

    @pytest.mark.parametrize("video,lab", [
        ('001_2304_1760.mp4', 'berlin'),
        ('T3-D1-B15-V5-C_960_540.mp4', 'nieh'),
        ('C1_11_03_25_my_video_1224_1024.mp4', 'sharoni'),
        ('C_2026-04-28_11-38-05-seg1_1224_1024.mp4', 'sharoni'),
    ])
    def test_lookup_by_filename_prefix(self, video, lab):
        assert frame_width_mm(video) == pytest.approx(EXPECTED[lab], abs=0.5)

    def test_unknown_lab_has_no_calibration(self):
        assert frame_width_mm('zz_unmeasured_lab.mp4') is None


class TestMillimetreConversion:
    """Normalised distance is mm / frame width in mm, so it is free of both
    resolution and camera distance."""

    @pytest.mark.parametrize("lab,video", [
        ('berlin', '001_2304_1760.mp4'),
        ('nieh', 'T3_960_540.mp4'),
        ('sharoni', 'C1_1224_1024.mp4'),
    ])
    def test_round_trip(self, lab, video):
        assert norm_distance_for_mm(10.0, video) == pytest.approx(
            10.0 / EXPECTED[lab], rel=1e-3)

    def test_same_mm_is_a_different_fraction_per_lab(self):
        """The whole point: one physical distance, three frame fractions."""
        berlin = norm_distance_for_mm(10.0, '001_2304_1760.mp4')
        nieh = norm_distance_for_mm(10.0, 'T3_960_540.mp4')
        assert nieh > berlin * 3  # nieh is zoomed in ~3.5x relative to berlin

    def test_resolution_tier_does_not_matter(self):
        full = norm_distance_for_mm(10.0, 'C1_1224_1024.mp4')
        half = norm_distance_for_mm(10.0, 'C1_612_512.mp4')
        assert full == pytest.approx(half)

    def test_reproduces_the_old_bee_length_radius(self):
        """5 mm was exactly the old bee_size_multiplier=0.5 operating point,
        so the reparameterisation is exact and config changes are a rename."""
        old_bee_length_fraction = {'berlin': 0.0224, 'nieh': 0.0775, 'sharoni': 0.0330}
        for lab, video in [('berlin', '001.mp4'), ('nieh', 'T1.mp4'), ('sharoni', 'C1.mp4')]:
            assert norm_distance_for_mm(5.0, video) == pytest.approx(
                0.5 * old_bee_length_fraction[lab], rel=0.01)

    def test_unknown_lab_returns_none_rather_than_guessing(self):
        assert norm_distance_for_mm(10.0, 'zz_unknown.mp4') is None


class TestFpsIsMandatory:
    """Time is seconds; frames only exist at the boundary. A missing fps used
    to yield a silent 15 fps assumption -- six callers took it (#5)."""

    def test_known_video(self):
        assert _fps_for({'v.mp4': 60.0}, 'v.mp4') == 60.0

    def test_missing_video_raises(self):
        with pytest.raises(ValueError, match="v.mp4"):
            _fps_for({'other.mp4': 60.0}, 'v.mp4')

    def test_no_map_at_all_raises(self):
        with pytest.raises(ValueError, match="v.mp4"):
            _fps_for(None, 'v.mp4')


class TestPipelineEndToEnd:
    """The helpers are only useful if the two pipeline stages actually take
    the new units. Synthetic input, no dataset needed."""

    @staticmethod
    def _one_run(video):
        preds = [[{
            'confidence': 0.9, 'position': [100.0, 80.0], 'direction': [1.0, 0.0],
            'temporal_offsets': [float(t), float(t + 16)], 'grid_cell': [2, 2, 0],
            'window_start': t,
        }] for t in (0, 4, 8)]
        return preds, [(0, 0)] * 3, [video] * 3, [(1024, 1224)] * 3

    def test_cluster_radius_mm_runs_and_groups(self):
        from src.utils.dance_eval import cross_window_cluster_predictions
        preds, origins, names, res = self._one_run('C1_1224_1024.mp4')

        runs = cross_window_cluster_predictions(
            preds, origins, names, res,
            video_fps={'C1_1224_1024.mp4': 60.0},
            cluster_radius_mm=5.0,
            temporal_threshold_sec=0.3,
            confidence_threshold=0.1,
            min_samples=1,
            direction_threshold_deg=0,
        )
        assert runs['C1_1224_1024.mp4'], "three overlapping windows should yield a run"

    def test_a_tiny_radius_splits_what_a_large_one_merges(self):
        """The radius is doing real work, in physical units."""
        from src.utils.dance_eval import cross_window_cluster_predictions
        preds, origins, names, res = self._one_run('C1_1224_1024.mp4')
        # push the three detections ~30 mm apart (30/303.3 of frame width)
        for i, w in enumerate(preds):
            w[0]['position'] = [100.0 + i * 0.1 * 1224, 80.0]

        kw = dict(video_fps={'C1_1224_1024.mp4': 60.0}, temporal_threshold_sec=0.3,
                  confidence_threshold=0.1, min_samples=1, direction_threshold_deg=0)
        tight = cross_window_cluster_predictions(preds, origins, names, res,
                                                 cluster_radius_mm=2.0, **kw)
        loose = cross_window_cluster_predictions(preds, origins, names, res,
                                                 cluster_radius_mm=200.0, **kw)
        assert len(tight['C1_1224_1024.mp4']) > len(loose['C1_1224_1024.mp4'])

    def test_metrics_accept_mm_thresholds(self):
        from src.utils.dance_eval import compute_dance_level_metrics
        video = 'C1_1224_1024.mp4'
        runs = {video: [{'position': [0.5, 0.5], 'direction': [1.0, 0.0],
                         'temporal_offsets': [0.0, 16.0], 'confidence': 0.9,
                         'n_detections': 3}]}
        gt = {video: [{'position': [612.0, 512.0], 'direction': [1.0, 0.0],
                       'temporal_offsets': [0.0, 16.0], 'n_windows': 3}]}

        m = compute_dance_level_metrics(
            runs, gt, {video: (1024, 1224)},
            pos_thresholds_mm=[10.0, 20.0],
            iou_threshold_range=(0.3, 0.7),
            angular_thresholds=[10],
        )
        # a prediction sitting exactly on the GT matches at any sane threshold
        assert m['coverage']['detected_dances'] == 1
