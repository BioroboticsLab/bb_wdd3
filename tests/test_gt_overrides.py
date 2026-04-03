"""Tests for GT annotation sidecar (overrides) system."""
import json
import os
import sys
import tempfile
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestGTOverrides:
    """Test the GT overrides sidecar file system."""

    def setup_method(self):
        """Create a temp dir for sidecar files."""
        self.tmpdir = tempfile.mkdtemp()
        self.sidecar_path = os.path.join(self.tmpdir, 'gt_overrides.json')

    def teardown_method(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_empty_sidecar(self):
        """Loading when no sidecar exists returns empty structure."""
        from viewer.gt_overrides import load_overrides
        data = load_overrides(self.sidecar_path)
        assert data == {'added': {}, 'modified': {}, 'deleted': {}}

    def test_save_and_load_roundtrip(self):
        """Save and load preserves data."""
        from viewer.gt_overrides import load_overrides, save_overrides
        data = load_overrides(self.sidecar_path)
        data['added']['test_video.mp4'] = [{
            'run_id': 99,
            'x': 100.0, 'y': 200.0,
            'dir_x': 0.5, 'dir_y': 0.866,
            'waggle_start': 100, 'waggle_end': 120,
        }]
        save_overrides(self.sidecar_path, data)
        assert os.path.exists(self.sidecar_path)

        reloaded = load_overrides(self.sidecar_path)
        assert len(reloaded['added']['test_video.mp4']) == 1
        assert reloaded['added']['test_video.mp4'][0]['run_id'] == 99

    def test_add_annotation(self):
        """Adding annotation creates entry in 'added' section."""
        from viewer.gt_overrides import load_overrides, add_annotation
        data = load_overrides(self.sidecar_path)
        
        run = add_annotation(data, 'video1.mp4', {
            'x': 150.0, 'y': 300.0,
            'dir_x': 1.0, 'dir_y': 0.0,
            'waggle_start': 50, 'waggle_end': 70,
        }, next_run_id=5)

        assert run['run_id'] == 5
        assert len(data['added']['video1.mp4']) == 1
        assert data['added']['video1.mp4'][0]['waggle_start'] == 50

    def test_add_multiple_annotations(self):
        """Adding multiple annotations to same video appends."""
        from viewer.gt_overrides import load_overrides, add_annotation
        data = load_overrides(self.sidecar_path)
        
        add_annotation(data, 'v.mp4', {
            'x': 10, 'y': 20, 'dir_x': 1, 'dir_y': 0,
            'waggle_start': 10, 'waggle_end': 20,
        }, next_run_id=1)
        add_annotation(data, 'v.mp4', {
            'x': 30, 'y': 40, 'dir_x': 0, 'dir_y': 1,
            'waggle_start': 30, 'waggle_end': 40,
        }, next_run_id=2)

        assert len(data['added']['v.mp4']) == 2

    def test_modify_existing_annotation(self):
        """Modifying a CSV annotation creates entry in 'modified' section."""
        from viewer.gt_overrides import load_overrides, modify_annotation
        data = load_overrides(self.sidecar_path)
        
        modify_annotation(data, 'video1.mp4', run_id=3, updates={
            'x': 200.0, 'waggle_end': 90,
        })

        assert '3' in data['modified'].get('video1.mp4', {})
        assert data['modified']['video1.mp4']['3']['x'] == 200.0

    def test_modify_added_annotation(self):
        """Modifying a previously added annotation updates in-place."""
        from viewer.gt_overrides import load_overrides, add_annotation, modify_annotation
        data = load_overrides(self.sidecar_path)
        
        add_annotation(data, 'v.mp4', {
            'x': 10, 'y': 20, 'dir_x': 1, 'dir_y': 0,
            'waggle_start': 10, 'waggle_end': 20,
        }, next_run_id=99)

        modify_annotation(data, 'v.mp4', run_id=99, updates={'x': 50.0})

        # Should update in-place in 'added', not create 'modified' entry
        assert data['added']['v.mp4'][0]['x'] == 50.0
        assert 'v.mp4' not in data.get('modified', {}) or '99' not in data['modified'].get('v.mp4', {})

    def test_delete_csv_annotation(self):
        """Deleting a CSV annotation adds to 'deleted' section."""
        from viewer.gt_overrides import load_overrides, delete_annotation
        data = load_overrides(self.sidecar_path)
        
        delete_annotation(data, 'video1.mp4', run_id=3)

        assert 3 in data['deleted'].get('video1.mp4', [])

    def test_delete_added_annotation(self):
        """Deleting a previously added annotation removes from 'added'."""
        from viewer.gt_overrides import load_overrides, add_annotation, delete_annotation
        data = load_overrides(self.sidecar_path)
        
        add_annotation(data, 'v.mp4', {
            'x': 10, 'y': 20, 'dir_x': 1, 'dir_y': 0,
            'waggle_start': 10, 'waggle_end': 20,
        }, next_run_id=99)

        delete_annotation(data, 'v.mp4', run_id=99)

        # Should be removed from 'added', not added to 'deleted'
        assert len(data['added'].get('v.mp4', [])) == 0
        assert 99 not in data['deleted'].get('v.mp4', [])

    def test_merge_with_csv_annotations(self):
        """Merge applies adds, modifies, and deletes to CSV-sourced runs."""
        from viewer.gt_overrides import load_overrides, add_annotation, modify_annotation, delete_annotation, merge_annotations

        csv_runs = [
            {'run_id': 1, 'video_name': 'v.mp4', 'x': 100, 'y': 200,
             'dir_x': 1.0, 'dir_y': 0.0, 'waggle_start': 10, 'waggle_end': 20},
            {'run_id': 2, 'video_name': 'v.mp4', 'x': 300, 'y': 400,
             'dir_x': 0.0, 'dir_y': 1.0, 'waggle_start': 50, 'waggle_end': 60},
        ]

        data = load_overrides(self.sidecar_path)
        
        # Delete run 1
        delete_annotation(data, 'v.mp4', run_id=1)
        
        # Modify run 2
        modify_annotation(data, 'v.mp4', run_id=2, updates={'x': 350.0})
        
        # Add run 99
        add_annotation(data, 'v.mp4', {
            'x': 500, 'y': 600, 'dir_x': 0.5, 'dir_y': 0.5,
            'waggle_start': 80, 'waggle_end': 95,
        }, next_run_id=99)

        merged = merge_annotations(csv_runs, data, 'v.mp4')

        # Run 1 should be deleted
        assert not any(r['run_id'] == 1 for r in merged)
        # Run 2 should be modified
        run2 = next(r for r in merged if r['run_id'] == 2)
        assert run2['x'] == 350.0
        # Run 99 should be added
        run99 = next(r for r in merged if r['run_id'] == 99)
        assert run99['waggle_start'] == 80

        # Total: 2 original - 1 deleted + 1 added = 2
        assert len(merged) == 2

    def test_next_run_id(self):
        """next_run_id is max(existing) + 1."""
        from viewer.gt_overrides import get_next_run_id
        
        csv_runs = [
            {'run_id': 1}, {'run_id': 5}, {'run_id': 3},
        ]
        assert get_next_run_id(csv_runs, []) == 6

    def test_next_run_id_with_added(self):
        """next_run_id considers both CSV and added runs."""
        from viewer.gt_overrides import get_next_run_id
        
        csv_runs = [{'run_id': 3}]
        added_runs = [{'run_id': 10}]
        assert get_next_run_id(csv_runs, added_runs) == 11

    def test_next_run_id_empty(self):
        """next_run_id with no existing runs returns 1."""
        from viewer.gt_overrides import get_next_run_id
        assert get_next_run_id([], []) == 1
