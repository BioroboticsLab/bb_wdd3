"""
TDD tests for vectorized eval_utils_fast vs original eval_utils.

Tests ensure numerical equivalence between the original Python-loop
implementations and the new vectorized versions.
"""
import torch
import numpy as np
import pytest
import time

from src.utils.eval_utils import (
    yolo_to_img_space,
    yolo_to_img_space_gt,
)
from src.utils.eval_utils_fast import (
    yolo_to_img_space_vectorized,
    yolo_to_img_space_gt_vectorized,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_random_model_output(batch_size=4, grid_size=7, max_det=1, seed=42):
    """Create a random model output tensor (before sigmoid on channel 0)."""
    rng = torch.Generator().manual_seed(seed)
    # 7 channels: objectness(logit), norm_x, norm_y, dir_x, dir_y, t_start, t_end
    out = torch.randn(batch_size, grid_size, grid_size, max_det, 7, generator=rng)
    # Make ~30% of cells have high objectness so we get some detections
    # (logit > 0 means sigmoid > 0.5)
    mask = torch.rand(batch_size, grid_size, grid_size, max_det, generator=rng) < 0.3
    out[..., 0] = torch.where(mask, out[..., 0].abs() + 2.0, out[..., 0] - 3.0)
    return out


def _make_gt_output(batch_size=4, grid_size=7, max_det=1, seed=42, n_positives=3):
    """Create a GT tensor where channel 0 is binary (0 or 1), not logit."""
    rng = torch.Generator().manual_seed(seed)
    out = torch.zeros(batch_size, grid_size, grid_size, max_det, 7)
    # Sprinkle some GT detections
    for b in range(batch_size):
        for _ in range(n_positives):
            i = torch.randint(0, grid_size, (1,), generator=rng).item()
            j = torch.randint(0, grid_size, (1,), generator=rng).item()
            k = torch.randint(0, max_det, (1,), generator=rng).item()
            out[b, i, j, k, 0] = 1.0
            out[b, i, j, k, 1] = torch.rand(1, generator=rng).item()  # norm_x
            out[b, i, j, k, 2] = torch.rand(1, generator=rng).item()  # norm_y
            out[b, i, j, k, 3] = torch.randn(1, generator=rng).item()  # dir_x
            out[b, i, j, k, 4] = torch.randn(1, generator=rng).item()  # dir_y
            out[b, i, j, k, 5] = torch.rand(1, generator=rng).item() * 0.5  # t_start
            out[b, i, j, k, 6] = 0.5 + torch.rand(1, generator=rng).item() * 0.5  # t_end
    return out


def _make_metadata(batch_size=4, window_size=16, stride=8):
    """Create matching starts/ends arrays."""
    starts = np.array([i * stride for i in range(batch_size)])
    ends = starts + window_size
    return starts, ends


def _detections_match(det_list_orig, det_list_fast, atol=1e-3):
    """
    Check that two lists-of-lists of detection dicts are equivalent.
    Detections within each sample are sorted by confidence for comparison.
    Returns (match: bool, message: str).
    """
    if len(det_list_orig) != len(det_list_fast):
        return False, f"Length mismatch: {len(det_list_orig)} vs {len(det_list_fast)}"

    for sample_idx in range(len(det_list_orig)):
        orig = sorted(det_list_orig[sample_idx], key=lambda d: (-d['confidence'], d['position'][0], d['position'][1]))
        fast = sorted(det_list_fast[sample_idx], key=lambda d: (-d['confidence'], d['position'][0], d['position'][1]))

        if len(orig) != len(fast):
            return False, (
                f"Sample {sample_idx}: different detection count: "
                f"{len(orig)} (orig) vs {len(fast)} (fast)"
            )

        for det_idx, (o, f) in enumerate(zip(orig, fast)):
            # Compare confidence
            if abs(o['confidence'] - f['confidence']) > atol:
                return False, (
                    f"Sample {sample_idx}, det {det_idx}: confidence mismatch: "
                    f"{o['confidence']:.8f} vs {f['confidence']:.8f}"
                )
            # Compare position
            for dim in range(2):
                if abs(o['position'][dim] - f['position'][dim]) > atol:
                    return False, (
                        f"Sample {sample_idx}, det {det_idx}: position[{dim}] mismatch: "
                        f"{o['position'][dim]:.8f} vs {f['position'][dim]:.8f}"
                    )
            # Compare direction
            for dim in range(2):
                if abs(o['direction'][dim] - f['direction'][dim]) > atol:
                    return False, (
                        f"Sample {sample_idx}, det {det_idx}: direction[{dim}] mismatch: "
                        f"{o['direction'][dim]:.8f} vs {f['direction'][dim]:.8f}"
                    )
            # Compare grid_cell
            if o['grid_cell'] != f['grid_cell']:
                return False, (
                    f"Sample {sample_idx}, det {det_idx}: grid_cell mismatch: "
                    f"{o['grid_cell']} vs {f['grid_cell']}"
                )
            # Compare temporal_offsets
            for dim in range(2):
                if abs(o['temporal_offsets'][dim] - f['temporal_offsets'][dim]) > atol:
                    return False, (
                        f"Sample {sample_idx}, det {det_idx}: temporal_offsets[{dim}] mismatch: "
                        f"{o['temporal_offsets'][dim]} vs {f['temporal_offsets'][dim]}"
                    )

    return True, "All detections match"


# ---------------------------------------------------------------------------
# Tests: yolo_to_img_space (predictions)
# ---------------------------------------------------------------------------

class TestYoloToImgSpaceVectorized:
    """Test vectorized predictions decoding against original."""

    def test_basic_equivalence_small(self):
        """Small grid, few samples — basic sanity check."""
        output = _make_random_model_output(batch_size=4, grid_size=7, max_det=1)
        starts, ends = _make_metadata(batch_size=4)

        orig = yolo_to_img_space(output, starts, ends,
                                 confidence_threshold=0.3,
                                 original_size=(960, 540),
                                 max_dets=10)
        fast = yolo_to_img_space_vectorized(output, starts, ends,
                                            confidence_threshold=0.3,
                                            original_size=(960, 540),
                                            max_dets=10)

        match, msg = _detections_match(orig, fast)
        assert match, msg

    def test_equivalence_realistic_grid(self):
        """Realistic grid_size=28 as used in production."""
        output = _make_random_model_output(batch_size=8, grid_size=28, max_det=1, seed=123)
        starts, ends = _make_metadata(batch_size=8)

        orig = yolo_to_img_space(output, starts, ends,
                                 confidence_threshold=0.001,
                                 original_size=(224, 224),
                                 max_dets=5)
        fast = yolo_to_img_space_vectorized(output, starts, ends,
                                            confidence_threshold=0.001,
                                            original_size=(224, 224),
                                            max_dets=5)

        match, msg = _detections_match(orig, fast)
        assert match, msg

    def test_equivalence_multi_det_per_cell(self):
        """Multiple detections per cell (max_det > 1)."""
        output = _make_random_model_output(batch_size=4, grid_size=7, max_det=3, seed=99)
        starts, ends = _make_metadata(batch_size=4)

        orig = yolo_to_img_space(output, starts, ends,
                                 confidence_threshold=0.1,
                                 original_size=(960, 540),
                                 max_dets=20)
        fast = yolo_to_img_space_vectorized(output, starts, ends,
                                            confidence_threshold=0.1,
                                            original_size=(960, 540),
                                            max_dets=20)

        match, msg = _detections_match(orig, fast)
        assert match, msg

    def test_no_detections_above_threshold(self):
        """All confidences below threshold → empty lists."""
        output = torch.zeros(2, 7, 7, 1, 7) - 10.0  # sigmoid(-10) ≈ 0
        starts, ends = _make_metadata(batch_size=2)

        orig = yolo_to_img_space(output, starts, ends,
                                 confidence_threshold=0.5,
                                 original_size=(960, 540),
                                 max_dets=10)
        fast = yolo_to_img_space_vectorized(output, starts, ends,
                                            confidence_threshold=0.5,
                                            original_size=(960, 540),
                                            max_dets=10)

        assert all(len(s) == 0 for s in orig)
        assert all(len(s) == 0 for s in fast)

    def test_all_detections_above_threshold(self):
        """All cells have high confidence — should still match."""
        output = torch.ones(2, 4, 4, 1, 7) * 5.0  # sigmoid(5) ≈ 0.993
        starts, ends = _make_metadata(batch_size=2)

        orig = yolo_to_img_space(output, starts, ends,
                                 confidence_threshold=0.001,
                                 original_size=(960, 540),
                                 max_dets=100)
        fast = yolo_to_img_space_vectorized(output, starts, ends,
                                            confidence_threshold=0.001,
                                            original_size=(960, 540),
                                            max_dets=100)

        match, msg = _detections_match(orig, fast)
        assert match, msg

    def test_max_dets_truncation(self):
        """max_dets should truncate to top-N by confidence."""
        output = _make_random_model_output(batch_size=2, grid_size=10, max_det=1, seed=77)
        # Make most cells high-confidence
        output[..., 0] = output[..., 0].abs() + 3.0
        starts, ends = _make_metadata(batch_size=2)

        orig = yolo_to_img_space(output, starts, ends,
                                 confidence_threshold=0.001,
                                 original_size=(960, 540),
                                 max_dets=5)
        fast = yolo_to_img_space_vectorized(output, starts, ends,
                                            confidence_threshold=0.001,
                                            original_size=(960, 540),
                                            max_dets=5)

        # Both should have exactly max_dets per sample
        for i in range(2):
            assert len(orig[i]) == 5, f"orig sample {i}: {len(orig[i])}"
            assert len(fast[i]) == 5, f"fast sample {i}: {len(fast[i])}"

        match, msg = _detections_match(orig, fast)
        assert match, msg

    def test_clamping_respects_image_bounds(self):
        """Positions should be clamped to [0, original_size-1]."""
        output = torch.zeros(1, 3, 3, 1, 7)
        output[0, 0, 0, 0, 0] = 10.0  # high confidence
        output[0, 0, 0, 0, 1] = -5.0  # norm_x way out of bounds
        output[0, 0, 0, 0, 2] = -5.0  # norm_y way out of bounds
        starts, ends = _make_metadata(batch_size=1)

        orig = yolo_to_img_space(output, starts, ends,
                                 confidence_threshold=0.001,
                                 original_size=(100, 100),
                                 max_dets=10)
        fast = yolo_to_img_space_vectorized(output, starts, ends,
                                            confidence_threshold=0.001,
                                            original_size=(100, 100),
                                            max_dets=10)

        # Position should be clamped to 0
        assert orig[0][0]['position'][0] == 0.0
        assert orig[0][0]['position'][1] == 0.0

        match, msg = _detections_match(orig, fast)
        assert match, msg

    def test_different_original_sizes(self):
        """Test with various non-square original sizes."""
        output = _make_random_model_output(batch_size=3, grid_size=7, max_det=1, seed=55)
        starts, ends = _make_metadata(batch_size=3)

        for orig_size in [(640, 480), (1920, 1080), (224, 224), (100, 200)]:
            orig = yolo_to_img_space(output, starts, ends,
                                     confidence_threshold=0.1,
                                     original_size=orig_size,
                                     max_dets=10)
            fast = yolo_to_img_space_vectorized(output, starts, ends,
                                                confidence_threshold=0.1,
                                                original_size=orig_size,
                                                max_dets=10)

            match, msg = _detections_match(orig, fast)
            assert match, f"Failed for size {orig_size}: {msg}"


# ---------------------------------------------------------------------------
# Tests: yolo_to_img_space_gt (ground truth)
# ---------------------------------------------------------------------------

class TestYoloToImgSpaceGtVectorized:
    """Test vectorized GT decoding against original."""

    def test_basic_equivalence(self):
        """Basic GT equivalence check."""
        output = _make_gt_output(batch_size=4, grid_size=7, max_det=1, n_positives=3)
        starts, ends = _make_metadata(batch_size=4)

        orig = yolo_to_img_space_gt(output, starts, ends,
                                    original_size=(960, 540))
        fast = yolo_to_img_space_gt_vectorized(output, starts, ends,
                                               original_size=(960, 540))

        match, msg = _detections_match(orig, fast)
        assert match, msg

    def test_gt_has_correct_extra_fields(self):
        """GT detections should have is_ground_truth and window_idx fields."""
        output = _make_gt_output(batch_size=2, grid_size=4, max_det=1, n_positives=2)
        starts, ends = _make_metadata(batch_size=2)

        fast = yolo_to_img_space_gt_vectorized(output, starts, ends,
                                               original_size=(960, 540))

        for b, sample_dets in enumerate(fast):
            for det in sample_dets:
                assert det['is_ground_truth'] is True, "Missing is_ground_truth"
                assert det['window_idx'] == b, f"Wrong window_idx: {det['window_idx']} != {b}"

    def test_empty_gt(self):
        """No GT objects → all empty lists."""
        output = torch.zeros(3, 7, 7, 1, 7)  # all zeros = no detections
        starts, ends = _make_metadata(batch_size=3)

        orig = yolo_to_img_space_gt(output, starts, ends)
        fast = yolo_to_img_space_gt_vectorized(output, starts, ends)

        assert all(len(s) == 0 for s in orig)
        assert all(len(s) == 0 for s in fast)

    def test_gt_realistic_grid(self):
        """Realistic grid_size=28."""
        output = _make_gt_output(batch_size=8, grid_size=28, max_det=1,
                                 n_positives=2, seed=101)
        starts, ends = _make_metadata(batch_size=8)

        orig = yolo_to_img_space_gt(output, starts, ends,
                                    original_size=(224, 224))
        fast = yolo_to_img_space_gt_vectorized(output, starts, ends,
                                               original_size=(224, 224))

        match, msg = _detections_match(orig, fast)
        assert match, msg

    def test_gt_multi_det_per_cell(self):
        """GT with multiple detections per cell."""
        output = _make_gt_output(batch_size=4, grid_size=7, max_det=3,
                                 n_positives=5, seed=200)
        starts, ends = _make_metadata(batch_size=4)

        orig = yolo_to_img_space_gt(output, starts, ends,
                                    original_size=(960, 540))
        fast = yolo_to_img_space_gt_vectorized(output, starts, ends,
                                               original_size=(960, 540))

        match, msg = _detections_match(orig, fast)
        assert match, msg


# ---------------------------------------------------------------------------
# Performance benchmark (not a correctness test — skipped by default)
# ---------------------------------------------------------------------------

class TestPerformance:
    """Benchmark vectorized vs original. Run with: pytest -s -k 'performance'"""

    @pytest.mark.slow
    def test_speed_preds_large(self):
        """Benchmark predictions decoding on realistic-sized data."""
        batch_size = 100
        grid_size = 28
        max_det = 1

        output = _make_random_model_output(batch_size, grid_size, max_det, seed=42)
        starts, ends = _make_metadata(batch_size, stride=8)

        # Warm up
        yolo_to_img_space(output, starts, ends, confidence_threshold=0.001,
                          original_size=(224, 224), max_dets=5)
        yolo_to_img_space_vectorized(output, starts, ends, confidence_threshold=0.001,
                                     original_size=(224, 224), max_dets=5)

        # Time original
        t0 = time.time()
        for _ in range(3):
            orig = yolo_to_img_space(output, starts, ends,
                                     confidence_threshold=0.001,
                                     original_size=(224, 224), max_dets=5)
        t_orig = (time.time() - t0) / 3

        # Time vectorized
        t0 = time.time()
        for _ in range(3):
            fast = yolo_to_img_space_vectorized(output, starts, ends,
                                                confidence_threshold=0.001,
                                                original_size=(224, 224), max_dets=5)
        t_fast = (time.time() - t0) / 3

        speedup = t_orig / t_fast if t_fast > 0 else float('inf')

        print(f"\n{'='*60}")
        print(f"PREDICTIONS DECODING BENCHMARK")
        print(f"  Batch={batch_size}, Grid={grid_size}, MaxDet={max_det}")
        print(f"  Original:   {t_orig:.4f}s")
        print(f"  Vectorized: {t_fast:.4f}s")
        print(f"  Speedup:    {speedup:.1f}x")
        print(f"{'='*60}")

        # Verify equivalence on the last run
        match, msg = _detections_match(orig, fast)
        assert match, msg

    @pytest.mark.slow
    def test_speed_gt_large(self):
        """Benchmark GT decoding on realistic-sized data."""
        batch_size = 100
        grid_size = 28
        max_det = 1

        output = _make_gt_output(batch_size, grid_size, max_det, n_positives=2, seed=42)
        starts, ends = _make_metadata(batch_size, stride=8)

        # Warm up
        yolo_to_img_space_gt(output, starts, ends, original_size=(224, 224))
        yolo_to_img_space_gt_vectorized(output, starts, ends, original_size=(224, 224))

        # Time original
        t0 = time.time()
        for _ in range(3):
            orig = yolo_to_img_space_gt(output, starts, ends, original_size=(224, 224))
        t_orig = (time.time() - t0) / 3

        # Time vectorized
        t0 = time.time()
        for _ in range(3):
            fast = yolo_to_img_space_gt_vectorized(output, starts, ends, original_size=(224, 224))
        t_fast = (time.time() - t0) / 3

        speedup = t_orig / t_fast if t_fast > 0 else float('inf')

        print(f"\n{'='*60}")
        print(f"GT DECODING BENCHMARK")
        print(f"  Batch={batch_size}, Grid={grid_size}, MaxDet={max_det}")
        print(f"  Original:   {t_orig:.4f}s")
        print(f"  Vectorized: {t_fast:.4f}s")
        print(f"  Speedup:    {speedup:.1f}x")
        print(f"{'='*60}")

        # Verify equivalence on the last run
        match, msg = _detections_match(orig, fast)
        assert match, msg


# ---------------------------------------------------------------------------
# Helpers for detection metrics tests
# ---------------------------------------------------------------------------

def _make_synthetic_detections(n_samples=10, n_preds_per_sample=3,
                                n_gts_per_sample=2, seed=42):
    """Create synthetic preds and gts for metrics testing.

    Generates detections with known positions, directions, and temporal offsets
    so that some preds match GTs and some don't, creating a realistic mix of
    TP, FP, and FN.
    """
    rng = np.random.RandomState(seed)
    preds = []
    gts = []

    for s in range(n_samples):
        sample_preds = []
        sample_gts = []

        # Create GTs at known positions
        for g in range(n_gts_per_sample):
            x = 50 + g * 80 + rng.randn() * 5
            y = 50 + g * 60 + rng.randn() * 5
            angle = rng.uniform(0, 2 * np.pi)
            t_start = s * 16 + g * 4
            t_end = t_start + 8

            sample_gts.append({
                'confidence': 1.0,
                'position': [float(x), float(y)],
                'direction': [float(np.cos(angle)), float(np.sin(angle))],
                'grid_cell': [g, g, 0],
                'temporal_offsets': [int(t_start), int(t_end)],
                'is_ground_truth': True,
                'window_idx': s,
            })

        # Create predictions — some close to GTs (TPs), some far (FPs)
        for p in range(n_preds_per_sample):
            if p < n_gts_per_sample:
                # Close to GT — should be a TP for loose thresholds
                gt = sample_gts[p]
                x = gt['position'][0] + rng.randn() * 3
                y = gt['position'][1] + rng.randn() * 3
                dx = gt['direction'][0] + rng.randn() * 0.1
                dy = gt['direction'][1] + rng.randn() * 0.1
                t_start = gt['temporal_offsets'][0] + int(rng.randn() * 1)
                t_end = gt['temporal_offsets'][1] + int(rng.randn() * 1)
                conf = 0.8 + rng.rand() * 0.2
            else:
                # Far from any GT — should be a FP
                x = 400 + rng.randn() * 20
                y = 400 + rng.randn() * 20
                angle = rng.uniform(0, 2 * np.pi)
                dx = np.cos(angle)
                dy = np.sin(angle)
                t_start = s * 16 + 100  # far temporally
                t_end = t_start + 5
                conf = 0.3 + rng.rand() * 0.4

            # Normalize direction
            norm = np.sqrt(dx**2 + dy**2) + 1e-8
            dx, dy = dx / norm, dy / norm

            sample_preds.append({
                'confidence': float(conf),
                'position': [float(x), float(y)],
                'direction': [float(dx), float(dy)],
                'grid_cell': [p, p, 0],
                'temporal_offsets': [int(t_start), int(t_end)],
            })

        preds.append(sample_preds)
        gts.append(sample_gts)

    return preds, gts


def _metrics_dicts_match(orig, fast, atol=1e-6):
    """Compare two metrics dicts from calculate_detection_metrics.

    Compares scalar fields and matched_pairs_per_combo lengths.
    Returns (match, message).
    """
    scalar_keys = ['map', 'mean_precision', 'mean_recall', 'mean_f1', 'mean_best_conf']
    for key in scalar_keys:
        if abs(orig[key] - fast[key]) > atol:
            return False, f"{key}: {orig[key]:.8f} vs {fast[key]:.8f}"

    # Compare ap_per_combo
    if len(orig['ap_per_combo']) != len(fast['ap_per_combo']):
        return False, (f"ap_per_combo length: {len(orig['ap_per_combo'])} "
                       f"vs {len(fast['ap_per_combo'])}")
    for i, (a, b) in enumerate(zip(orig['ap_per_combo'], fast['ap_per_combo'])):
        if abs(a - b) > atol:
            return False, f"ap_per_combo[{i}]: {a:.8f} vs {b:.8f}"

    # Compare matched_pairs counts per combo
    if len(orig['matched_pairs_per_combo']) != len(fast['matched_pairs_per_combo']):
        return False, (f"matched_pairs_per_combo length: "
                       f"{len(orig['matched_pairs_per_combo'])} vs "
                       f"{len(fast['matched_pairs_per_combo'])}")
    for i, (op, fp) in enumerate(zip(orig['matched_pairs_per_combo'],
                                      fast['matched_pairs_per_combo'])):
        if len(op) != len(fp):
            return False, (f"matched_pairs_per_combo[{i}] length: "
                           f"{len(op)} vs {len(fp)}")

    return True, "All metrics match"


def _full_metrics_match(orig, fast, atol=1e-6):
    """Compare two full get_eval_metrics result dicts.

    Returns (match, message).
    """
    # Compare comprehensive sub-dict
    for key in ['map', 'mean_precision', 'mean_recall', 'mean_f1', 'mean_best_conf']:
        o_val = orig['comprehensive'][key]
        f_val = fast['comprehensive'][key]
        if abs(o_val - f_val) > atol:
            return False, f"comprehensive.{key}: {o_val:.8f} vs {f_val:.8f}"

    # Compare spatial
    for key in ['precision', 'recall', 'f1', 'mean_error']:
        o_val = orig['spatial'][key]
        f_val = fast['spatial'][key]
        if o_val == float('inf') and f_val == float('inf'):
            continue
        if abs(o_val - f_val) > atol:
            return False, f"spatial.{key}: {o_val:.8f} vs {f_val:.8f}"

    # Compare temporal
    for key in ['mean_iou', 'duration_accuracy', 'me_start', 'me_end']:
        o_val = orig['temporal'][key]
        f_val = fast['temporal'][key]
        if o_val == float('inf') and f_val == float('inf'):
            continue
        if abs(o_val - f_val) > atol:
            return False, f"temporal.{key}: {o_val:.8f} vs {f_val:.8f}"

    # Compare directional
    for key in ['accuracy', 'mean_error', 'mean_cosine_similarity']:
        o_val = orig['directional'][key]
        f_val = fast['directional'][key]
        if o_val == float('inf') and f_val == float('inf'):
            continue
        if abs(o_val - f_val) > atol:
            return False, f"directional.{key}: {o_val:.8f} vs {f_val:.8f}"

    # Compare counts
    if orig['counts'] != fast['counts']:
        return False, f"counts: {orig['counts']} vs {fast['counts']}"

    return True, "All metrics match"


# ---------------------------------------------------------------------------
# Tests: calculate_detection_metrics_parallel
# ---------------------------------------------------------------------------

from src.utils.eval_utils import calculate_detection_metrics, get_eval_metrics
from src.utils.eval_utils_fast import (
    calculate_detection_metrics_parallel,
    get_eval_metrics_fast,
)


class TestDetectionMetricsParallel:
    """Test parallel detection metrics against original serial version."""

    def test_greedy_equivalence(self):
        """Greedy matching should produce identical results."""
        preds, gts = _make_synthetic_detections(n_samples=10, seed=42)

        orig = calculate_detection_metrics(
            preds, gts,
            pos_thresholds=[10, 20, 30],
            iou_threshold_range=(0.25, 0.75),
            angular_thresholds=[15, 20],
            match_pairs='greedy',
        )
        fast = calculate_detection_metrics_parallel(
            preds, gts,
            pos_thresholds=[10, 20, 30],
            iou_threshold_range=(0.25, 0.75),
            angular_thresholds=[15, 20],
            match_pairs='greedy',
        )

        match, msg = _metrics_dicts_match(orig, fast)
        assert match, msg

    def test_hungarian_equivalence(self):
        """Hungarian matching should produce identical results."""
        preds, gts = _make_synthetic_detections(n_samples=10, seed=42)

        orig = calculate_detection_metrics(
            preds, gts,
            pos_thresholds=[10, 20],
            iou_threshold_range=(0.3, 0.5),
            angular_thresholds=[15],
            match_pairs='hungarian',
        )
        fast = calculate_detection_metrics_parallel(
            preds, gts,
            pos_thresholds=[10, 20],
            iou_threshold_range=(0.3, 0.5),
            angular_thresholds=[15],
            match_pairs='hungarian',
        )

        match, msg = _metrics_dicts_match(orig, fast)
        assert match, msg

    def test_single_threshold_combo(self):
        """Single combo: 1 pos × 1 iou × 1 angular = 1 combo total."""
        preds, gts = _make_synthetic_detections(n_samples=5, seed=77)

        orig = calculate_detection_metrics(
            preds, gts,
            pos_thresholds=[20],
            iou_threshold_range=(0.5, 0.5),
            angular_thresholds=[20],
            match_pairs='greedy',
        )
        fast = calculate_detection_metrics_parallel(
            preds, gts,
            pos_thresholds=[20],
            iou_threshold_range=(0.5, 0.5),
            angular_thresholds=[20],
            match_pairs='greedy',
        )

        match, msg = _metrics_dicts_match(orig, fast)
        assert match, msg

    def test_empty_predictions(self):
        """No predictions at all."""
        preds = [[] for _ in range(5)]
        _, gts = _make_synthetic_detections(n_samples=5, seed=42)

        orig = calculate_detection_metrics(
            preds, gts,
            pos_thresholds=[20],
            iou_threshold_range=(0.5, 0.5),
            angular_thresholds=[20],
            match_pairs='greedy',
        )
        fast = calculate_detection_metrics_parallel(
            preds, gts,
            pos_thresholds=[20],
            iou_threshold_range=(0.5, 0.5),
            angular_thresholds=[20],
            match_pairs='greedy',
        )

        match, msg = _metrics_dicts_match(orig, fast)
        assert match, msg

    def test_empty_ground_truths(self):
        """No GTs at all."""
        preds, _ = _make_synthetic_detections(n_samples=5, seed=42)
        gts = [[] for _ in range(5)]

        orig = calculate_detection_metrics(
            preds, gts,
            pos_thresholds=[20],
            iou_threshold_range=(0.5, 0.5),
            angular_thresholds=[20],
            match_pairs='greedy',
        )
        fast = calculate_detection_metrics_parallel(
            preds, gts,
            pos_thresholds=[20],
            iou_threshold_range=(0.5, 0.5),
            angular_thresholds=[20],
            match_pairs='greedy',
        )

        match, msg = _metrics_dicts_match(orig, fast)
        assert match, msg

    def test_production_thresholds(self):
        """Full production threshold grid: 5×11×3 = 165 combos."""
        preds, gts = _make_synthetic_detections(n_samples=15, n_preds_per_sample=4,
                                                 n_gts_per_sample=2, seed=123)

        orig = calculate_detection_metrics(
            preds, gts,
            pos_thresholds=[10, 15, 20, 25, 30],
            iou_threshold_range=(0.25, 0.75),
            angular_thresholds=[10, 15, 20],
            match_pairs='greedy',
        )
        fast = calculate_detection_metrics_parallel(
            preds, gts,
            pos_thresholds=[10, 15, 20, 25, 30],
            iou_threshold_range=(0.25, 0.75),
            angular_thresholds=[10, 15, 20],
            match_pairs='greedy',
        )

        match, msg = _metrics_dicts_match(orig, fast)
        assert match, msg
        # Verify correct number of combos
        assert len(orig['ap_per_combo']) == len(fast['ap_per_combo'])

    def test_production_thresholds_hungarian(self):
        """Full production threshold grid with hungarian matching."""
        preds, gts = _make_synthetic_detections(n_samples=8, n_preds_per_sample=3,
                                                 n_gts_per_sample=2, seed=200)

        orig = calculate_detection_metrics(
            preds, gts,
            pos_thresholds=[10, 15, 20, 25, 30],
            iou_threshold_range=(0.25, 0.75),
            angular_thresholds=[10, 15, 20],
            match_pairs='hungarian',
        )
        fast = calculate_detection_metrics_parallel(
            preds, gts,
            pos_thresholds=[10, 15, 20, 25, 30],
            iou_threshold_range=(0.25, 0.75),
            angular_thresholds=[10, 15, 20],
            match_pairs='hungarian',
        )

        match, msg = _metrics_dicts_match(orig, fast)
        assert match, msg


# ---------------------------------------------------------------------------
# Tests: get_eval_metrics_fast (full pipeline)
# ---------------------------------------------------------------------------

class TestGetEvalMetricsFast:
    """Test fast full eval metrics against original."""

    def test_greedy_full_pipeline(self):
        """Full pipeline equivalence with greedy matching."""
        preds, gts = _make_synthetic_detections(n_samples=10, seed=42)

        orig = get_eval_metrics(
            preds, gts,
            pos_thresholds=[10, 20],
            iou_threshold_range=(0.3, 0.5),
            angular_thresholds=[15, 20],
            match_pairs='greedy',
        )
        fast = get_eval_metrics_fast(
            preds, gts,
            pos_thresholds=[10, 20],
            iou_threshold_range=(0.3, 0.5),
            angular_thresholds=[15, 20],
            match_pairs='greedy',
        )

        match, msg = _full_metrics_match(orig, fast)
        assert match, msg

    def test_hungarian_full_pipeline(self):
        """Full pipeline equivalence with hungarian matching."""
        preds, gts = _make_synthetic_detections(n_samples=10, seed=42)

        orig = get_eval_metrics(
            preds, gts,
            pos_thresholds=[15, 25],
            iou_threshold_range=(0.3, 0.5),
            angular_thresholds=[20],
            match_pairs='hungarian',
        )
        fast = get_eval_metrics_fast(
            preds, gts,
            pos_thresholds=[15, 25],
            iou_threshold_range=(0.3, 0.5),
            angular_thresholds=[20],
            match_pairs='hungarian',
        )

        match, msg = _full_metrics_match(orig, fast)
        assert match, msg

    def test_config_passed_through(self):
        """Config section should reflect the thresholds used."""
        preds, gts = _make_synthetic_detections(n_samples=5, seed=42)

        result = get_eval_metrics_fast(
            preds, gts,
            pos_thresholds=[10, 20],
            iou_threshold_range=(0.3, 0.5),
            angular_thresholds=[15],
            match_pairs='greedy',
        )

        assert result['config']['pos_thresholds'] == [10, 20]
        assert result['config']['iou_threshold_range'] == (0.3, 0.5)
        assert result['config']['angular_thresholds'] == [15]


# ---------------------------------------------------------------------------
# Performance benchmark: detection metrics
# ---------------------------------------------------------------------------

class TestDetectionMetricsPerformance:
    """Benchmark parallel vs serial detection metrics."""

    @pytest.mark.slow
    def test_speed_detection_metrics(self):
        """Benchmark with production-sized data and threshold grid."""
        preds, gts = _make_synthetic_detections(
            n_samples=200, n_preds_per_sample=5,
            n_gts_per_sample=3, seed=42
        )

        pos_thresholds = [10, 15, 20, 25, 30]
        iou_range = (0.25, 0.75)
        angular_thresholds = [10, 15, 20]

        # Warm up
        calculate_detection_metrics(preds, gts, pos_thresholds, iou_range,
                                    angular_thresholds, match_pairs='greedy')
        calculate_detection_metrics_parallel(preds, gts, pos_thresholds, iou_range,
                                             angular_thresholds, match_pairs='greedy')

        # Time original (greedy)
        t0 = time.time()
        orig = calculate_detection_metrics(preds, gts, pos_thresholds, iou_range,
                                           angular_thresholds, match_pairs='greedy')
        t_orig = time.time() - t0

        # Time parallel (greedy)
        t0 = time.time()
        fast = calculate_detection_metrics_parallel(preds, gts, pos_thresholds,
                                                     iou_range, angular_thresholds,
                                                     match_pairs='greedy')
        t_fast = time.time() - t0

        n_combos = len(pos_thresholds) * len(np.arange(iou_range[0], iou_range[1] + 0.05, 0.05)) * len(angular_thresholds)
        speedup = t_orig / t_fast if t_fast > 0 else float('inf')
        total_preds = sum(len(p) for p in preds)

        print(f"\n{'='*60}")
        print(f"DETECTION METRICS BENCHMARK (greedy)")
        print(f"  Samples={len(preds)}, TotalPreds={total_preds}, Combos≈{int(n_combos)}")
        print(f"  Original:  {t_orig:.4f}s")
        print(f"  Parallel:  {t_fast:.4f}s")
        print(f"  Speedup:   {speedup:.1f}x")
        print(f"{'='*60}")

        match, msg = _metrics_dicts_match(orig, fast)
        assert match, msg

    @pytest.mark.slow
    def test_speed_detection_metrics_hungarian(self):
        """Benchmark with hungarian matching — shows larger speedup."""
        preds, gts = _make_synthetic_detections(
            n_samples=100, n_preds_per_sample=5,
            n_gts_per_sample=3, seed=42
        )

        pos_thresholds = [10, 15, 20, 25, 30]
        iou_range = (0.25, 0.75)
        angular_thresholds = [10, 15, 20]

        # Warm up
        calculate_detection_metrics(preds, gts, pos_thresholds, iou_range,
                                    angular_thresholds, match_pairs='hungarian')
        calculate_detection_metrics_parallel(preds, gts, pos_thresholds, iou_range,
                                             angular_thresholds, match_pairs='hungarian')

        # Time original (hungarian)
        t0 = time.time()
        orig = calculate_detection_metrics(preds, gts, pos_thresholds, iou_range,
                                           angular_thresholds, match_pairs='hungarian')
        t_orig = time.time() - t0

        # Time parallel (hungarian)
        t0 = time.time()
        fast = calculate_detection_metrics_parallel(preds, gts, pos_thresholds,
                                                     iou_range, angular_thresholds,
                                                     match_pairs='hungarian')
        t_fast = time.time() - t0

        n_combos = len(pos_thresholds) * len(np.arange(iou_range[0], iou_range[1] + 0.05, 0.05)) * len(angular_thresholds)
        speedup = t_orig / t_fast if t_fast > 0 else float('inf')
        total_preds = sum(len(p) for p in preds)

        print(f"\n{'='*60}")
        print(f"DETECTION METRICS BENCHMARK (hungarian)")
        print(f"  Samples={len(preds)}, TotalPreds={total_preds}, Combos≈{int(n_combos)}")
        print(f"  Original:  {t_orig:.4f}s")
        print(f"  Parallel:  {t_fast:.4f}s")
        print(f"  Speedup:   {speedup:.1f}x")
        print(f"{'='*60}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
