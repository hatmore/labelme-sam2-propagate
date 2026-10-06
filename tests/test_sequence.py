"""
Unit tests for sequence parsing and frame grouping.
"""

import tempfile
from pathlib import Path

import pytest

from labelme_sam2_propagate.sequence import (
    MISC_PREFIX,
    collect_sequences,
    compute_frame_intervals,
    find_frame_by_stem,
    parse_frame_name,
)


def parse_timestamp(stem: str) -> float:
    """Helper to extract timestamp as float from stem."""
    _, sec, nsec = parse_frame_name(stem)
    return sec + nsec / 1e9


class TestFrameNameParsing:
    """Test prefix / timestamp splitting."""

    def test_standard(self):
        assert parse_frame_name("N_camera_0_link_1788425812_781000000") == (
            "N_camera_0_link", 1788425812, 781000000)

    def test_extension_is_stripped(self):
        assert parse_frame_name("N_camera_0_link_1788425812_781000000.jpg") == (
            "N_camera_0_link", 1788425812, 781000000)

    def test_empty_prefix(self):
        assert parse_frame_name("_1788425812_781000000") == ("", 1788425812, 781000000)

    def test_seconds_only(self):
        assert parse_frame_name("frame_1788425812") == ("frame", 1788425812, 0)

    def test_numeric_prefix_component_is_not_mistaken_for_seconds(self):
        """'cam_2_<sec>_<nsec>' keeps '2' in the prefix."""
        assert parse_frame_name("cam_2_1788425812_781000000") == ("cam_2", 1788425812, 781000000)

    def test_no_timestamp(self):
        assert parse_frame_name("random_image") == (MISC_PREFIX, 0, 0)
        assert parse_frame_name("readme") == (MISC_PREFIX, 0, 0)


class TestTimestampParsing:
    """Test timestamp extraction from filenames."""

    def test_parse_timestamp_standard(self):
        """Parse standard timestamp format."""
        ts = parse_timestamp("N_camera_0_link_1788425812_781000000.jpg")
        assert ts == 1788425812.781

    def test_parse_timestamp_with_leading_underscore(self):
        """Parse timestamp with leading underscore."""
        ts = parse_timestamp("_1788425812_781000000.jpg")
        assert ts == 1788425812.781

    def test_parse_timestamp_no_nanoseconds(self):
        """Parse timestamp without nanoseconds."""
        ts = parse_timestamp("frame_1788425812.jpg")
        assert ts == 1788425812.0

    def test_parse_timestamp_invalid(self):
        """Invalid timestamp should return 0."""
        ts = parse_timestamp("no_timestamp_here.jpg")
        assert ts == 0.0

    def test_parse_timestamp_multiple_underscores(self):
        """Handle multiple underscores correctly."""
        ts = parse_timestamp("P_camera_0_link_1788424304_131000000.jpg")
        assert ts == 1788424304.131


class TestSequenceCollection:
    """Test frame sequence grouping by camera prefix."""

    def test_collect_sequences_single_camera(self):
        """Single camera should create one sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test files
            (tmpdir / "N_camera_0_link_1788425812_781000000.jpg").touch()
            (tmpdir / "N_camera_0_link_1788425813_281000000.jpg").touch()
            (tmpdir / "N_camera_0_link_1788425813_781000000.jpg").touch()

            sequences = collect_sequences(str(tmpdir))

            assert len(sequences) == 1
            assert "N_camera_0_link" in sequences
            assert len(sequences["N_camera_0_link"]) == 3

    def test_collect_sequences_multiple_cameras(self):
        """Multiple cameras should create separate sequences."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # N camera frames
            (tmpdir / "N_camera_0_link_1788425812_781000000.jpg").touch()
            (tmpdir / "N_camera_0_link_1788425813_281000000.jpg").touch()

            # P camera frames
            (tmpdir / "P_camera_0_link_1788425812_781000000.jpg").touch()
            (tmpdir / "P_camera_0_link_1788425813_281000000.jpg").touch()
            (tmpdir / "P_camera_0_link_1788425813_781000000.jpg").touch()

            sequences = collect_sequences(str(tmpdir))

            assert len(sequences) == 2
            assert "N_camera_0_link" in sequences
            assert "P_camera_0_link" in sequences
            assert len(sequences["N_camera_0_link"]) == 2
            assert len(sequences["P_camera_0_link"]) == 3

    def test_collect_sequences_sorted_by_timestamp(self):
        """Frames should be sorted by timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create files in random order
            (tmpdir / "N_camera_0_link_1788425813_781000000.jpg").touch()
            (tmpdir / "N_camera_0_link_1788425812_281000000.jpg").touch()
            (tmpdir / "N_camera_0_link_1788425812_781000000.jpg").touch()

            sequences = collect_sequences(str(tmpdir))
            frames = sequences["N_camera_0_link"]

            # Check sorted order
            timestamps = [parse_timestamp(stem) for stem, _ in frames]
            assert timestamps == sorted(timestamps)
            assert timestamps[0] == 1788425812.281
            assert timestamps[1] == 1788425812.781
            assert timestamps[2] == 1788425813.781

    def test_collect_sequences_ignores_non_images(self):
        """Should skip non-image files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create mix of files
            (tmpdir / "N_camera_0_link_1788425812_781000000.jpg").touch()
            (tmpdir / "N_camera_0_link_1788425812_781000000.json").touch()
            (tmpdir / "N_camera_0_link_1788425813_281000000.png").touch()
            (tmpdir / "readme.txt").touch()

            sequences = collect_sequences(str(tmpdir))

            assert len(sequences) == 1
            assert len(sequences["N_camera_0_link"]) == 2  # jpg + png

    def test_collect_sequences_groups_by_prefix(self):
        """Frames should be grouped by longest common prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Different prefix patterns
            (tmpdir / "camera_A_1788425812_781000000.jpg").touch()
            (tmpdir / "camera_A_1788425813_281000000.jpg").touch()
            (tmpdir / "camera_B_1788425812_781000000.jpg").touch()

            sequences = collect_sequences(str(tmpdir))

            assert len(sequences) == 2
            assert "camera_A" in sequences
            assert "camera_B" in sequences

    def test_collect_sequences_empty_directory(self):
        """Empty directory should return empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sequences = collect_sequences(str(tmpdir))
            assert sequences == {}

    def test_collect_sequences_no_timestamp_pattern(self):
        """Files without timestamp pattern should be grouped together."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # No timestamp in filename
            (tmpdir / "frame_001.jpg").touch()
            (tmpdir / "frame_002.jpg").touch()

            sequences = collect_sequences(str(tmpdir))

            # Trailing numbers are used as the sort key, prefix "frame" groups them
            assert list(sequences) == ["frame"]
            assert [stem for stem, _ in sequences["frame"]] == ["frame_001", "frame_002"]


class TestFrameLookup:
    FRAMES = [
        ("N_camera_0_link_1788425812_781000000", "/a.jpg"),
        ("N_camera_0_link_1788425813_281000000", "/b.jpg"),
        ("N_camera_0_link_1788425813_781000000", "/c.jpg"),
    ]

    def test_find_frame_by_stem_unique(self):
        assert find_frame_by_stem(self.FRAMES, "1788425812") == 0

    def test_find_frame_by_stem_no_match(self):
        with pytest.raises(ValueError, match="No frame matching"):
            find_frame_by_stem(self.FRAMES, "9999")

    def test_find_frame_by_stem_ambiguous(self):
        with pytest.raises(ValueError, match="Multiple frames"):
            find_frame_by_stem(self.FRAMES, "1788425813")

    def test_compute_frame_intervals(self):
        intervals = compute_frame_intervals(self.FRAMES)
        assert len(intervals) == 2
        assert intervals == pytest.approx([0.5, 0.5])

    def test_compute_frame_intervals_single_frame(self):
        assert compute_frame_intervals(self.FRAMES[:1]) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
