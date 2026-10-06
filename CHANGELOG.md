# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release with SAM 2.1 video propagation
- Multi-seed bidirectional propagation (forward + backward)
- Static object optimization (skip tracking for walls/floors)
- Auto-generated annotation tracking via a content-hash ledger (`.sam2_auto.json`
  next to the annotations); edited results are recognised as manual seeds.
  Legacy `flags: {"sam2_auto": true}` files are still detected.
- Preview generation with mask overlay
- China mirror support for model weights (Aliyun, Tencent Cloud, Tsinghua)
- Comprehensive test suite (unit + integration)
- CLI with progress bars and validation
- Support for multi-camera sequences (auto-groups by filename prefix)

### Performance
- ~1.3s/frame on RTX 3050 4GB (hiera_small model)
- ~3.5GB VRAM usage
- 10+ simultaneous objects

### Documentation
- README with quick start guide
- Demo GIF showing 1→24 frame propagation
- API documentation in docstrings
- PyPI publishing workflow

## [0.1.0] - TBD

First public release.

[Unreleased]: https://github.com/hatmore/labelme-sam2-propagate/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hatmore/labelme-sam2-propagate/releases/tag/v0.1.0
