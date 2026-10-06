"""
Frame sequence management and sorting.
"""

import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple


# Frame filenames are "<prefix>_<seconds>_<nanoseconds>" (ROS-style stamps),
# but we also accept "<prefix>_<seconds>" and an empty prefix. Patterns are
# tried in order, strictest first.
_FRAME_PATTERNS = (
    re.compile(r"^(?P<prefix>.*)_(?P<sec>\d+)_(?P<nsec>\d+)$"),
    re.compile(r"^(?P<prefix>.*)_(?P<sec>\d+)$"),
)

# Key used for files whose name carries no timestamp
MISC_PREFIX = "_misc"

# Supported image extensions
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def parse_frame_name(stem: str) -> Tuple[str, int, int]:
    """Parse frame filename into prefix and timestamp.

    Args:
        stem: Filename without extension. A trailing image extension is
            tolerated and stripped.

    Returns:
        Tuple of (prefix, seconds, nanoseconds)
        Returns ("_misc", 0, 0) for filenames without a trailing timestamp

    Examples:
        "N_camera_0_link_1788425812_781000000" -> ("N_camera_0_link", 1788425812, 781000000)
        "P_camera_0_link_1788425813_281000000" -> ("P_camera_0_link", 1788425813, 281000000)
        "_1788425812_781000000"                -> ("", 1788425812, 781000000)
        "frame_1788425812"                     -> ("frame", 1788425812, 0)
        "random_image"                         -> ("_misc", 0, 0)
    """
    base, ext = os.path.splitext(stem)
    if ext.lower() in IMAGE_EXTENSIONS:
        stem = base

    for pattern in _FRAME_PATTERNS:
        match = pattern.match(stem)
        if match:
            nsec = match.groupdict().get("nsec")
            return (
                match.group("prefix"),
                int(match.group("sec")),
                int(nsec) if nsec is not None else 0,
            )
    return (MISC_PREFIX, 0, 0)


def collect_sequences(directory: str) -> Dict[str, List[Tuple[str, str]]]:
    """Collect and sort image sequences by camera prefix.

    Args:
        directory: Path to directory containing images

    Returns:
        Dict mapping prefix -> [(stem, image_path), ...] sorted by timestamp

    Examples:
        {
            "N_camera_0_link": [
                ("N_camera_0_link_1788425812_781000000", "/path/to/img1.jpg"),
                ("N_camera_0_link_1788425813_281000000", "/path/to/img2.jpg"),
            ],
            "P_camera_0_link": [
                ("P_camera_0_link_1788425812_781000000", "/path/to/img3.jpg"),
            ]
        }
    """
    groups = defaultdict(list)

    for filename in os.listdir(directory):
        stem, ext = os.path.splitext(filename)

        # Skip non-image files
        if ext.lower() not in IMAGE_EXTENSIONS:
            continue

        prefix, sec, nsec = parse_frame_name(stem)
        image_path = os.path.join(directory, filename)

        groups[prefix].append(((sec, nsec, stem), stem, image_path))

    # Sort each group by timestamp (stem as a stable tie-breaker)
    result = {}
    for prefix, items in groups.items():
        items.sort(key=lambda x: x[0])
        result[prefix] = [(stem, path) for _, stem, path in items]

    return result


def find_frame_by_stem(
    frames: List[Tuple[str, str]],
    search_term: str
) -> int:
    """Find frame index by partial stem match.

    Args:
        frames: List of (stem, path) tuples
        search_term: Substring to search for in stems

    Returns:
        Frame index (0-based)

    Raises:
        ValueError: If no match or multiple matches found
    """
    matches = [i for i, (stem, _) in enumerate(frames) if search_term in stem]

    if len(matches) == 0:
        raise ValueError(f"No frame matching '{search_term}' found")
    elif len(matches) > 1:
        raise ValueError(
            f"Multiple frames ({len(matches)}) match '{search_term}'. "
            f"Please be more specific."
        )

    return matches[0]


def get_frame_info(frames: List[Tuple[str, str]], index: int) -> Tuple[str, str]:
    """Get frame stem and path by index.

    Args:
        frames: List of (stem, path) tuples
        index: Frame index (0-based)

    Returns:
        Tuple of (stem, image_path)
    """
    return frames[index]


def compute_frame_intervals(frames: List[Tuple[str, str]]) -> List[float]:
    """Compute time intervals between consecutive frames.

    Args:
        frames: List of (stem, path) tuples with timestamp stems

    Returns:
        List of intervals in seconds (length = len(frames) - 1)
    """
    intervals = []

    for i in range(len(frames) - 1):
        stem1 = frames[i][0]
        stem2 = frames[i + 1][0]

        _, sec1, nsec1 = parse_frame_name(stem1)
        _, sec2, nsec2 = parse_frame_name(stem2)

        time1 = sec1 + nsec1 / 1e9
        time2 = sec2 + nsec2 / 1e9

        intervals.append(time2 - time1)

    return intervals
