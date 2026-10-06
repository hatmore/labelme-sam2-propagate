"""
Unit tests for the auto-generation ledger.

The ledger is a ``.sam2_auto.json`` file stored *next to* the annotations.
It maps ``<stem> -> sha1(shapes)`` for every file the tool wrote, so a
file counts as "auto-generated" only while its shapes still hash to the
recorded value. Editing it in LabelMe turns it into a manual seed.
"""

import json
import tempfile
from pathlib import Path

import pytest

from labelme_sam2_propagate.ledger import (
    LEDGER_FILENAME,
    is_auto_generated,
    is_manual_seed,
    load_ledger,
    mark_as_auto,
    save_ledger,
)
from labelme_sam2_propagate.utils import shapes_hash

SHAPES = [{"label": "car", "points": [[10, 10], [20, 20], [10, 20]]}]
OTHER_SHAPES = [{"label": "bike", "points": [[30, 30], [40, 40], [30, 40]]}]


def write_json(path: Path, shapes, flags=None) -> None:
    path.write_text(json.dumps({"shapes": shapes, "flags": flags or {}}), encoding="utf-8")


class TestLedgerBasics:
    """Test ledger save/load operations."""

    def test_load_ledger_nonexistent(self):
        """Loading from a directory without a ledger returns an empty dict."""
        assert load_ledger("/nonexistent/path") == {}

    def test_save_and_load_ledger(self):
        """Round-trip save and load ledger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"frame_001": "abc123", "frame_002": "def456"}

            save_ledger(tmpdir, data)

            assert (Path(tmpdir) / LEDGER_FILENAME).exists()
            assert load_ledger(tmpdir) == data

    def test_save_ledger_creates_directory(self):
        """save_ledger creates the target directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "subdir"

            save_ledger(str(target), {"frame_001": "abc"})

            assert (target / LEDGER_FILENAME).exists()

    def test_save_ledger_is_atomic(self):
        """No .tmp file is left behind after saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_ledger(tmpdir, {"a": "b"})
            leftovers = [p.name for p in Path(tmpdir).iterdir()]
            assert leftovers == [LEDGER_FILENAME]

    def test_load_ledger_malformed(self):
        """A corrupt ledger file is treated as empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / LEDGER_FILENAME).write_text("{not json")
            assert load_ledger(tmpdir) == {}


class TestAutoGenerationTracking:
    """Test auto-generation detection logic."""

    def test_hash_match_is_auto(self):
        """Shapes matching the recorded hash are auto-generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "frame.json"
            write_json(json_path, SHAPES)

            ledger = {"frame": shapes_hash(SHAPES)}

            assert is_auto_generated(str(json_path), ledger) is True

    def test_hash_mismatch_is_not_auto(self):
        """Edited shapes no longer match the ledger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "frame.json"
            write_json(json_path, OTHER_SHAPES)

            ledger = {"frame": shapes_hash(SHAPES)}

            assert is_auto_generated(str(json_path), ledger) is False

    def test_legacy_flag_true(self):
        """Files from older versions carry flags.sam2_auto and are still detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "frame.json"
            write_json(json_path, [], flags={"sam2_auto": True})

            assert is_auto_generated(str(json_path), {}) is True

    def test_no_flag_no_ledger_is_manual(self):
        """Without a ledger entry or legacy flag the file is manual."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "frame.json"
            write_json(json_path, [])

            assert is_auto_generated(str(json_path), {}) is False

    def test_legacy_flag_false_is_manual(self):
        """flags.sam2_auto == false means manual."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "frame.json"
            write_json(json_path, [], flags={"sam2_auto": False})

            assert is_auto_generated(str(json_path), {}) is False

    def test_ledger_entry_wins_over_legacy_flag(self):
        """When a ledger entry exists the hash decides, not the flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "frame.json"
            write_json(json_path, OTHER_SHAPES, flags={"sam2_auto": True})

            ledger = {"frame": shapes_hash(SHAPES)}

            assert is_auto_generated(str(json_path), ledger) is False

    def test_nonexistent_file(self):
        """Non-existent file should return False."""
        assert is_auto_generated("/nonexistent/frame.json", {}) is False

    def test_malformed_json(self):
        """Malformed JSON should return False (fail-safe)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "frame.json"
            json_path.write_text("{invalid json")

            assert is_auto_generated(str(json_path), {}) is False

    def test_ledger_loaded_from_disk_when_omitted(self):
        """Omitting the ledger argument loads the ledger next to the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "frame.json"
            write_json(json_path, SHAPES)
            save_ledger(tmpdir, {"frame": shapes_hash(SHAPES)})

            assert is_auto_generated(str(json_path)) is True


class TestManualSeedDetection:
    """Test manual seed detection logic."""

    def test_modified_auto_result_is_seed(self):
        """An auto result edited by the user becomes a seed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "frame.json"
            write_json(json_path, SHAPES)

            ledger = {"frame": "old_hash"}

            assert is_manual_seed(str(json_path), ledger) is True

    def test_unchanged_auto_result_is_not_seed(self):
        """Unchanged auto result should not be a seed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "frame.json"
            write_json(json_path, SHAPES)

            ledger = {"frame": shapes_hash(SHAPES)}

            assert is_manual_seed(str(json_path), ledger) is False

    def test_untracked_file_with_shapes_is_seed(self):
        """A hand-made annotation unknown to the ledger is a seed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "frame.json"
            write_json(json_path, OTHER_SHAPES)

            assert is_manual_seed(str(json_path), {}) is True

    def test_empty_shapes_is_not_seed(self):
        """A JSON without shapes cannot seed propagation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "frame.json"
            write_json(json_path, [])

            assert is_manual_seed(str(json_path), {}) is False

    def test_missing_file_is_not_seed(self):
        assert is_manual_seed("/nonexistent/frame.json", {}) is False

    def test_malformed_json_is_not_seed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "frame.json"
            json_path.write_text("{invalid json")

            assert is_manual_seed(str(json_path), {}) is False


class TestMarkAsAuto:
    """Test recording auto-generated files in the ledger."""

    def test_mark_as_auto_records_hash(self):
        """mark_as_auto stores sha1(shapes) under the file stem."""
        ledger = {}

        mark_as_auto("/data/frame_001.json", SHAPES, ledger)

        assert ledger == {"frame_001": shapes_hash(SHAPES)}

    def test_mark_as_auto_overwrites_previous_entry(self):
        ledger = {"frame_001": "stale"}

        mark_as_auto("/data/frame_001.json", SHAPES, ledger)

        assert ledger["frame_001"] == shapes_hash(SHAPES)

    def test_mark_as_auto_keeps_other_entries(self):
        ledger = {"frame_000": "keep"}

        mark_as_auto("/data/frame_001.json", SHAPES, ledger)

        assert ledger["frame_000"] == "keep"
        assert "frame_001" in ledger

    def test_mark_then_detect_roundtrip(self):
        """A file written and marked by the tool is detected as auto."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "frame.json"
            write_json(json_path, SHAPES)

            ledger = {}
            mark_as_auto(str(json_path), SHAPES, ledger)

            assert is_auto_generated(str(json_path), ledger) is True
            assert is_manual_seed(str(json_path), ledger) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
