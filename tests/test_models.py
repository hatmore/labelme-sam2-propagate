"""
Unit tests for SAM2 model configuration and checkpoint management.

These tests never download anything: ``get_model_spec`` is pure, and
``get_model_config`` is exercised with the downloader patched out.
"""

import pytest

from labelme_sam2_propagate import models
from labelme_sam2_propagate.models import (
    CHECKPOINT_DIR,
    MODELS,
    get_checkpoint_urls,
    get_model_config,
    get_model_spec,
)


class TestModelSpec:
    """Test the download-free model description."""

    @pytest.mark.parametrize(
        "size, cfg_name, ckpt_name, repo",
        [
            ("tiny", "sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt", "sam2.1-hiera-tiny"),
            ("small", "sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt", "sam2.1-hiera-small"),
            ("base_plus", "sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt", "sam2.1-hiera-base-plus"),
            ("large", "sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt", "sam2.1-hiera-large"),
        ],
    )
    def test_spec_for_each_size(self, size, cfg_name, ckpt_name, repo):
        cfg_path, ckpt_path, url = get_model_spec(size)

        assert cfg_path.endswith(cfg_name)
        assert cfg_path.startswith("configs/sam2.1/")
        assert ckpt_path.endswith(ckpt_name)
        assert repo in url
        assert url.startswith("https://")

    def test_default_is_small(self):
        assert get_model_spec() == get_model_spec("small")

    def test_invalid_size(self):
        with pytest.raises(ValueError, match="Invalid model_size"):
            get_model_spec("invalid_model")

    def test_checkpoint_location(self):
        """Checkpoints live in the user cache directory."""
        _, ckpt_path, _ = get_model_spec("small")

        assert ckpt_path.startswith(CHECKPOINT_DIR)
        assert "sam2_ckpt" in ckpt_path

    def test_all_models_have_specs(self):
        for size in MODELS:
            cfg, ckpt, url = get_model_spec(size)
            assert cfg and ckpt and url


class TestCheckpointUrls:
    def test_mirror_order(self):
        """The China mirror is tried first, then Facebook CDN, then HuggingFace."""
        urls = get_checkpoint_urls("small")

        assert len(urls) == 3
        assert "hf-mirror.com" in urls[0]
        assert "dl.fbaipublicfiles.com" in urls[1]
        assert "huggingface.co" in urls[2]
        assert all(u.endswith("sam2.1_hiera_small.pt") for u in urls)


class TestModelConfig:
    """Test the downloading wrapper with the network patched out."""

    def test_returns_config_and_checkpoint(self, monkeypatch):
        calls = []

        def fake_download(model_size, **kwargs):
            calls.append(model_size)
            return "/cache/fake.pt"

        monkeypatch.setattr(models, "download_checkpoint", fake_download)

        cfg_path, ckpt_path = get_model_config("tiny")

        assert cfg_path.endswith("sam2.1_hiera_t.yaml")
        assert ckpt_path == "/cache/fake.pt"
        assert calls == ["tiny"]

    def test_invalid_size_does_not_download(self, monkeypatch):
        def fail_download(*args, **kwargs):
            raise AssertionError("download_checkpoint must not be called")

        monkeypatch.setattr(models, "download_checkpoint", fail_download)

        with pytest.raises(ValueError, match="Invalid model_size"):
            get_model_config("invalid_model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
