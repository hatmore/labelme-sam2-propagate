"""
Unit tests for utils module (mask/polygon conversion, visualization).
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from labelme_sam2_propagate.utils import (
    load_labelme_json,
    make_labelme_shape,
    mask_to_polygon,
    polygon_to_mask,
    save_labelme_json,
    shapes_hash,
    visualize_annotation,
)


class TestPolygonMaskConversion:
    """Test bidirectional polygon <-> mask conversion."""

    def test_polygon_to_mask_basic(self):
        """Convert simple polygon to mask."""
        points = [[10, 10], [50, 10], [50, 50], [10, 50]]
        mask = polygon_to_mask(points, height=100, width=100)

        assert mask.shape == (100, 100)
        assert mask.dtype == bool
        assert mask[30, 30] == True  # Inside
        assert mask[5, 5] == False  # Outside

    def test_polygon_to_mask_triangle(self):
        """Convert triangle to mask."""
        points = [[50, 10], [90, 90], [10, 90]]
        mask = polygon_to_mask(points, height=100, width=100)

        assert mask[50, 50] == True
        assert mask[10, 10] == False

    def test_mask_to_polygon_rectangle(self):
        """Convert rectangular mask to polygon."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 30:70] = 1

        polygon = mask_to_polygon(mask, min_area=50, max_points=200)

        assert polygon is not None
        assert len(polygon) >= 4
        assert len(polygon) <= 200
        # Check approximate bounds
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        assert min(xs) >= 29 and max(xs) <= 71
        assert min(ys) >= 19 and max(ys) <= 81

    def test_mask_to_polygon_small_area_filtered(self):
        """Small masks below min_area should return None."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[45:55, 45:55] = 1  # 10×10 = 100 pixels

        polygon = mask_to_polygon(mask, min_area=200)
        assert polygon is None

    def test_mask_to_polygon_max_points_limit(self):
        """Complex shapes should respect max_points limit."""
        # Create star-like shape with many vertices
        mask = np.zeros((200, 200), dtype=np.uint8)
        center = (100, 100)
        for angle in range(0, 360, 10):
            rad = np.radians(angle)
            r = 50 + 20 * (angle % 20)
            x = int(center[0] + r * np.cos(rad))
            y = int(center[1] + r * np.sin(rad))
            mask[max(0, y-2):min(200, y+3), max(0, x-2):min(200, x+3)] = 1

        polygon = mask_to_polygon(mask, max_points=50)

        assert polygon is not None
        assert len(polygon) <= 50

    def test_roundtrip_conversion(self):
        """Polygon -> mask -> polygon should preserve approximate shape."""
        original = [[20, 20], [80, 20], [80, 80], [20, 80]]
        mask = polygon_to_mask(original, 100, 100)
        recovered = mask_to_polygon(mask, min_area=50, max_points=200)

        assert recovered is not None
        # Should still be roughly 4 corners (allowing for approximation)
        assert 4 <= len(recovered) <= 8


class TestLabelMeIO:
    """Test LabelMe JSON reading/writing."""

    def test_make_labelme_shape(self):
        """Create valid LabelMe shape dict."""
        shape = make_labelme_shape(
            label="car",
            points=[[10, 10], [20, 20], [30, 10]],
            group_id=1,
            description="test object"
        )

        assert shape["label"] == "car"
        assert shape["points"] == [[10, 10], [20, 20], [30, 10]]
        assert shape["group_id"] == 1
        assert shape["description"] == "test object"
        assert shape["shape_type"] == "polygon"
        assert shape["flags"] == {}
        assert shape["mask"] is None

    def test_save_and_load_labelme_json(self):
        """Round-trip save and load LabelMe JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.jpg"
            json_path = Path(tmpdir) / "test.json"

            # Create dummy image
            img = Image.new("RGB", (640, 480), color="red")
            img.save(img_path)

            shapes = [
                make_labelme_shape("car", [[10, 10], [50, 50], [10, 50]]),
                make_labelme_shape("person", [[100, 100], [150, 150], [100, 150]]),
            ]

            save_labelme_json(
                str(json_path),
                str(img_path),
                shapes,
                height=480,
                width=640,
                embed_image=False
            )

            assert json_path.exists()

            data = load_labelme_json(str(json_path))

            assert data["imageHeight"] == 480
            assert data["imageWidth"] == 640
            assert data["imagePath"] == "test.jpg"
            assert len(data["shapes"]) == 2
            assert data["shapes"][0]["label"] == "car"
            assert data["shapes"][1]["label"] == "person"

    def test_shapes_hash_consistency(self):
        """Same shapes should produce same hash."""
        shapes1 = [
            {"label": "car", "points": [[10, 10], [20, 20]]},
            {"label": "bike", "points": [[30, 30], [40, 40]]},
        ]
        shapes2 = [
            {"label": "car", "points": [[10, 10], [20, 20]]},
            {"label": "bike", "points": [[30, 30], [40, 40]]},
        ]

        hash1 = shapes_hash(shapes1)
        hash2 = shapes_hash(shapes2)

        assert hash1 == hash2
        assert len(hash1) == 40  # SHA1 hex digest

    def test_shapes_hash_difference(self):
        """Different shapes should produce different hashes."""
        shapes1 = [{"label": "car", "points": [[10, 10], [20, 20]]}]
        shapes2 = [{"label": "bike", "points": [[10, 10], [20, 20]]}]

        hash1 = shapes_hash(shapes1)
        hash2 = shapes_hash(shapes2)

        assert hash1 != hash2


class TestVisualization:
    """Test annotation visualization."""

    def test_visualize_annotation_creates_overlay(self):
        """Visualization should create output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "input.jpg"
            output_path = Path(tmpdir) / "output.jpg"

            # Create test image
            img = Image.new("RGB", (640, 480), color="white")
            img.save(img_path)

            shapes = [
                {"label": "car", "points": [[100, 100], [200, 100], [200, 200], [100, 200]]},
                {"label": "bike", "points": [[300, 300], [400, 300], [400, 400], [300, 400]]},
            ]

            visualize_annotation(
                str(img_path),
                shapes,
                str(output_path),
                alpha=0.5
            )

            assert output_path.exists()
            # Check output is valid image (close it so the temp dir can be removed on Windows)
            with Image.open(output_path) as result:
                assert result.size == (640, 480)

    def test_visualize_annotation_without_directory(self, tmp_path, monkeypatch):
        """A bare filename (no directory part) must not crash."""
        monkeypatch.chdir(tmp_path)
        img_path = tmp_path / "input.jpg"
        Image.new("RGB", (64, 48), color="white").save(img_path)

        shapes = [{"label": "car", "points": [[5, 5], [20, 5], [20, 20], [5, 20]]}]
        visualize_annotation(str(img_path), shapes, "output.jpg")

        assert (tmp_path / "output.jpg").exists()

    def test_visualize_annotation_with_missing_image(self):
        """Visualization should handle missing source image gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.jpg"

            shapes = [{"label": "car", "points": [[10, 10], [20, 20], [10, 20]]}]

            # Should not crash
            visualize_annotation(
                "/nonexistent/image.jpg",
                shapes,
                str(output_path)
            )

            # Output should not be created
            assert not output_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
