"""
Integration tests for end-to-end propagation workflow.

Note: These tests require torch and SAM2 dependencies.
Run with: pytest tests/test_integration.py --slow
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from labelme_sam2_propagate import run_propagation
from labelme_sam2_propagate.utils import load_labelme_json, make_labelme_shape

N_FRAMES = 5
BOX_SIZE = 100
BOX_STEP = 12  # pixels the box moves per frame
BOX_X0, BOX_Y0 = 100, 100


def frame_name(i: int) -> str:
    return f"frame_{1788425812 + i}_{i * 100000000:09d}"


def box_at(i: int):
    """Top-left corner of the tracked box in frame ``i``."""
    return BOX_X0 + i * BOX_STEP, BOX_Y0


@pytest.fixture
def sample_sequence():
    """Create a trackable image sequence with one seed annotation.

    Plain single-colour frames give SAM2 nothing to track, so each frame is
    a light, lightly textured background with a dark box that moves a few
    pixels per frame.
    """
    rng = np.random.default_rng(0)
    background = rng.integers(170, 230, size=(480, 640, 3), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for i in range(N_FRAMES):
            img = background.copy()
            x, y = box_at(i)
            img[y:y + BOX_SIZE, x:x + BOX_SIZE] = (30, 40, 120)
            Image.fromarray(img).save(tmpdir / f"{frame_name(i)}.jpg", quality=95)

        # Create seed annotation for first frame
        seed_json = tmpdir / f"{frame_name(0)}.json"
        x, y = box_at(0)
        shapes = [
            make_labelme_shape(
                "box",
                [[x, y], [x + BOX_SIZE, y], [x + BOX_SIZE, y + BOX_SIZE], [x, y + BOX_SIZE]],
            ),
        ]

        data = {
            "version": "5.10.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": f"{frame_name(0)}.jpg",
            "imageData": None,
            "imageHeight": 480,
            "imageWidth": 640,
        }

        seed_json.write_text(json.dumps(data, indent=2))

        yield str(tmpdir)


@pytest.mark.slow
class TestIntegration:
    """Integration tests requiring full SAM2 stack."""

    def test_basic_propagation(self, sample_sequence):
        """Test basic propagation from one seed frame."""
        written = run_propagation(
            data_dir=sample_sequence,
            preview=False,
            model_size="tiny",  # Use smallest model for speed
        )

        # Should generate 4 new annotations (5 frames - 1 seed)
        assert written == N_FRAMES - 1

        # Check all frames now have JSON files whose polygon follows the box
        for i in range(N_FRAMES):
            json_path = Path(sample_sequence) / f"{frame_name(i)}.json"
            assert json_path.exists()

            data = load_labelme_json(str(json_path))
            assert len(data["shapes"]) == 1
            shape = data["shapes"][0]
            assert shape["label"] == "box"
            assert shape["shape_type"] == "polygon"

            xs = [p[0] for p in shape["points"]]
            ys = [p[1] for p in shape["points"]]
            x, y = box_at(i)
            tol = 15
            assert abs(min(xs) - x) < tol and abs(max(xs) - (x + BOX_SIZE)) < tol
            assert abs(min(ys) - y) < tol and abs(max(ys) - (y + BOX_SIZE)) < tol

        # Ledger records every generated frame, not the seed
        ledger = json.loads((Path(sample_sequence) / ".sam2_auto.json").read_text())
        assert set(ledger) == {frame_name(i) for i in range(1, N_FRAMES)}

    def test_rerun_is_idempotent(self, sample_sequence):
        """Running twice regenerates auto results but writes nothing new for seeds."""
        first = run_propagation(data_dir=sample_sequence, model_size="tiny")
        second = run_propagation(data_dir=sample_sequence, model_size="tiny")

        assert first == N_FRAMES - 1
        assert second == N_FRAMES - 1  # auto results are regenerated, seed untouched
        seed = load_labelme_json(str(Path(sample_sequence) / f"{frame_name(0)}.json"))
        assert seed["shapes"][0]["points"][0] == list(box_at(0))

    def test_propagation_with_preview(self, sample_sequence):
        """Test propagation with preview generation."""
        written = run_propagation(
            data_dir=sample_sequence,
            preview=True,
            model_size="tiny",
        )

        assert written == N_FRAMES - 1

        # Check preview directory exists
        preview_dir = Path(sample_sequence) / "_preview"
        assert preview_dir.exists()

        # One preview per generated frame (the seed frame gets none)
        preview_files = sorted(p.name for p in preview_dir.glob("*.jpg"))
        assert preview_files == [f"{frame_name(i)}.jpg" for i in range(1, N_FRAMES)]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--slow"])
