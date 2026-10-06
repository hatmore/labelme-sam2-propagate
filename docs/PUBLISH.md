# Publishing to PyPI

## First-time Setup

1. **Create PyPI account**: https://pypi.org/account/register/
2. **Generate API token**: 
   - Go to https://pypi.org/manage/account/token/
   - Create token with scope "Entire account" (or project-specific after first upload)
   - Copy the token (starts with `pypi-...`)
3. **Add token to GitHub**:
   - Go to repository Settings → Secrets and variables → Actions
   - New repository secret: `PYPI_API_TOKEN` = your token

## Publishing Workflow

### Option 1: GitHub Release (Recommended)

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "0.1.0"  # Follow semver
   ```

2. **Commit and tag**:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.1.0"
   git tag v0.1.0
   git push origin main --tags
   ```

3. **Create GitHub Release**:
   - Go to repository → Releases → Draft a new release
   - Choose tag `v0.1.0`
   - Title: `v0.1.0`
   - Describe changes
   - Publish release

   GitHub Actions will automatically build and upload to PyPI.

### Option 2: Manual Upload

```bash
# Install build tools
pip install build twine

# Build distribution
python -m build

# Upload to PyPI
twine upload dist/*
```

For TestPyPI (dry run):
```bash
twine upload --repository testpypi dist/*
```

## Versioning

Follow [semantic versioning](https://semver.org/):
- `0.1.0` → First public release
- `0.1.1` → Bug fixes
- `0.2.0` → New features (backward compatible)
- `1.0.0` → Stable API

## Pre-release Checklist

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Version bumped in `pyproject.toml`
- [ ] CHANGELOG.md updated
- [ ] README.md accurate
- [ ] LICENSE included
- [ ] Requirements up to date

## After Publishing

Users can install with:
```bash
pip install labelme-sam2-propagate
```

Check package page: https://pypi.org/project/labelme-sam2-propagate/
