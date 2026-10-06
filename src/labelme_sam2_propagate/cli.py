#!/usr/bin/env python3
"""
Command-line interface for labelme-sam2-propagate
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="SAM2 video propagation for LabelMe annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic propagation with preview
  labelme-sam2-propagate --dir /path/to/images --preview

  # With static object optimization
  labelme-sam2-propagate --dir /path/to/images --static wall,floor,ceiling

  # Use smaller model for low VRAM
  labelme-sam2-propagate --dir /path/to/images --model tiny

For more information, see: https://github.com/hatmore/labelme-sam2-propagate
        """
    )

    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing image sequence and seed annotations"
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Generate preview images with overlays in _preview/ subdirectory"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing annotations (creates .bak backups)"
    )

    parser.add_argument(
        "--static",
        default="",
        help="Comma-separated list of static labels to copy instead of track"
    )

    parser.add_argument(
        "--model",
        choices=["tiny", "small", "base_plus", "large"],
        default="small",
        help="SAM2 model size (default: small)"
    )

    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Only propagate forward, skip backward propagation"
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=200,
        help="Maximum polygon vertices (default: 200)"
    )

    parser.add_argument(
        "--checkpoint",
        help="Path to SAM2 checkpoint (auto-downloads if not specified)"
    )

    parser.add_argument(
        "--model-cfg",
        help="Path to SAM2 config file (uses default if not specified)"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )

    args = parser.parse_args()

    # Validate directory
    data_dir = Path(args.dir)
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    if not data_dir.is_dir():
        print(f"Error: Not a directory: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Import and run core logic
    # Note: Import here to avoid loading heavy dependencies during --help
    try:
        from .core import run_propagation
    except ImportError:
        # Fallback for direct script execution
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from labelme_sam2_propagate.core import run_propagation

    # Run propagation
    try:
        run_propagation(
            data_dir=str(data_dir),
            preview=args.preview,
            overwrite=args.overwrite,
            static_labels=args.static.split(",") if args.static else [],
            model_size=args.model,
            forward_only=args.forward_only,
            max_points=args.max_points,
            checkpoint=args.checkpoint,
            model_cfg=args.model_cfg,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
