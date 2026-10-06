"""
SAM2 model management and checkpoint downloading.
"""

import os
import urllib.request
from typing import Dict, List, Tuple


# Model configurations: (config_path, checkpoint_name, huggingface_repo)
MODELS: Dict[str, Tuple[str, str, str]] = {
    "tiny": (
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt",
        "facebook/sam2.1-hiera-tiny"
    ),
    "small": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.pt",
        "facebook/sam2.1-hiera-small"
    ),
    "base_plus": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt",
        "facebook/sam2.1-hiera-base-plus"
    ),
    "large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt",
        "facebook/sam2.1-hiera-large"
    ),
}

CHECKPOINT_DIR = os.path.join(os.path.expanduser("~"), ".cache", "sam2_ckpt")


def get_checkpoint_urls(model_size: str) -> List[str]:
    """Get list of mirror URLs for checkpoint download.

    Args:
        model_size: Model size key ("tiny", "small", "base_plus", "large")

    Returns:
        List of URLs to try in order (mirrors first)
    """
    _, checkpoint_name, hf_repo = MODELS[model_size]
    return [
        # Chinese mirror (fastest in mainland China)
        f"https://hf-mirror.com/{hf_repo}/resolve/main/{checkpoint_name}",
        # Official Facebook CDN
        f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{checkpoint_name}",
        # HuggingFace official
        f"https://huggingface.co/{hf_repo}/resolve/main/{checkpoint_name}",
    ]


def download_checkpoint(
    model_size: str,
    max_retries: int = 4,
    timeout: int = 60
) -> str:
    """Download SAM2 checkpoint with resume support.

    Tries multiple mirror URLs in order. Supports partial download resumption.

    Args:
        model_size: Model size key ("tiny", "small", "base_plus", "large")
        max_retries: Number of retry attempts per URL
        timeout: Request timeout in seconds

    Returns:
        Path to downloaded checkpoint

    Raises:
        RuntimeError: If all download attempts fail
    """
    _, checkpoint_name, _ = MODELS[model_size]
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    final_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    if os.path.exists(final_path):
        return final_path

    temp_path = final_path + ".part"
    urls = get_checkpoint_urls(model_size)

    for url in urls:
        for attempt in range(max_retries):
            try:
                # Check existing partial download
                existing_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0

                # Build request with resume header
                req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
                if existing_size > 0:
                    req.add_header("Range", f"bytes={existing_size}-")

                print(f"[checkpoint] Downloading from {url}")
                if existing_size > 0:
                    print(f"[checkpoint] Resuming from {existing_size / 1e6:.1f} MB")

                with urllib.request.urlopen(req, timeout=timeout) as response:
                    # Handle resume vs fresh download
                    mode = "ab" if existing_size > 0 and response.status == 206 else "wb"
                    with open(temp_path, mode) as f:
                        # If server doesn't support resume, start over
                        if existing_size > 0 and response.status != 206:
                            f.seek(0)
                            f.truncate()
                            existing_size = 0

                        # Calculate total size
                        content_length = int(response.headers.get("Content-Length", 0))
                        total_size = content_length + (existing_size if response.status == 206 else 0)
                        downloaded = existing_size if response.status == 206 else 0

                        # Download with progress
                        chunk_size = 1 << 20  # 1MB chunks
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = downloaded / total_size * 100
                                print(
                                    f"\r[checkpoint] {downloaded/1e6:7.1f}/{total_size/1e6:.1f} MB "
                                    f"({progress:.1f}%)",
                                    end=""
                                )

                print()  # New line after progress
                os.replace(temp_path, final_path)
                return final_path

            except Exception as e:
                print(f"\n[checkpoint] Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    print(f"[checkpoint] All retries failed for {url}")

    # All URLs failed
    error_msg = (
        f"Failed to download {checkpoint_name} from all mirrors.\n"
        f"You can manually download it to: {CHECKPOINT_DIR}/{checkpoint_name}\n"
        f"Available sources:\n  " + "\n  ".join(urls)
    )
    raise RuntimeError(error_msg)


def get_model_config(model_size: str) -> Tuple[str, str]:
    """Get SAM2 config path and checkpoint path.

    Downloads checkpoint if not already cached.

    Args:
        model_size: Model size key ("tiny", "small", "base_plus", "large")

    Returns:
        Tuple of (config_path, checkpoint_path)
    """
    if model_size not in MODELS:
        raise ValueError(
            f"Unknown model size: {model_size}. "
            f"Available: {', '.join(MODELS.keys())}"
        )

    config_path, _, _ = MODELS[model_size]
    checkpoint_path = download_checkpoint(model_size)

    return config_path, checkpoint_path
