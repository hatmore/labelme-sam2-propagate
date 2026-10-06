#!/usr/bin/env python3
"""
Generate demo GIFs from preview directory.
Wrapper for the module-based implementation.
"""

import sys
from pathlib import Path

# Add src to path for direct execution
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from labelme_sam2_propagate.create_demo_gifs import main

if __name__ == "__main__":
    main()
