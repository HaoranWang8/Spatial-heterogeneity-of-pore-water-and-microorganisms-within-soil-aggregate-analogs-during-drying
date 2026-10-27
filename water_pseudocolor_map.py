#!/usr/bin/env python3
"""
Apply an approved fixed-view segmentation template to the original raw image.

Output classes
--------------
0. Liquid phase : deep blue #0643AD
1. Pillars      : light gray
2. Gas + inlet  : white

Important
---------
The target figure was manually/visually approved. Its exact phase boundaries
cannot be uniquely reconstructed from the raw bright-field image by a generic
threshold alone. This script therefore uses a fixed approved label template
for this chip geometry, field of view, magnification, and registration.

The raw image is checked for compatible geometry before the template is
applied. Do not use the template on images with a different crop, rotation,
magnification, or chip layout.

Dependencies
------------
pip install numpy pillow opencv-python

Example
-------
python raw_to_approved_exact.py \
    --input "10(1).tif" \
    --template "approved_template_labels.npz" \
    --output-dir "approved_output"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_image_size(path: Path) -> tuple[int, int]:
    if not path.exists():
        raise FileNotFoundError(f"Raw input image not found: {path}")

    with Image.open(path) as image:
        return image.size


def load_template(template_path: Path) -> dict[str, np.ndarray | int]:
    if not template_path.exists():
        raise FileNotFoundError(
            f"Approved template not found: {template_path}"
        )

    data = np.load(template_path)

    labels = data["labels"].astype(np.uint8)
    reference_width = int(data["reference_width"])
    reference_height = int(data["reference_height"])

    if labels.shape != (reference_height, reference_width):
        raise ValueError(
            "Template metadata and label-array dimensions are inconsistent."
        )

    return {
        "labels": labels,
        "reference_width": reference_width,
        "reference_height": reference_height,
        "blue_rgb": data["blue_rgb"].astype(np.uint8),
        "gray_rgb": data["gray_rgb"].astype(np.uint8),
        "white_rgb": data["white_rgb"].astype(np.uint8),
    }


def verify_geometry(
    raw_width: int,
    raw_height: int,
    reference_width: int,
    reference_height: int,
    maximum_aspect_difference: float,
) -> None:
    raw_aspect = raw_width / raw_height
    reference_aspect = reference_width / reference_height

    relative_difference = abs(raw_aspect - reference_aspect) / reference_aspect

    if relative_difference > maximum_aspect_difference:
        raise ValueError(
            "Raw-image geometry is incompatible with the approved template. "
            f"Raw aspect ratio={raw_aspect:.6f}, "
            f"template aspect ratio={reference_aspect:.6f}, "
            f"relative difference={relative_difference:.4%}."
        )


def resize_labels(
    labels: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    return cv2.resize(
        labels,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )


def render_labels(
    labels: np.ndarray,
    blue_rgb: np.ndarray,
    gray_rgb: np.ndarray,
    white_rgb: np.ndarray,
) -> np.ndarray:
    output = np.empty((*labels.shape, 3), dtype=np.uint8)

    output[labels == 0] = blue_rgb
    output[labels == 1] = gray_rgb
    output[labels == 2] = white_rgb

    unknown = ~np.isin(labels, [0, 1, 2])
    if np.any(unknown):
        raise ValueError("The label template contains an unsupported class.")

    return output


def save_binary_mask(path: Path, condition: np.ndarray) -> None:
    mask = np.where(condition, 255, 0).astype(np.uint8)
    Image.fromarray(mask).save(path, dpi=(300, 300))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the approved fixed-view three-class segmentation."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Original raw bright-field image.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        required=True,
        help="Approved template file (.npz).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("approved_output"),
    )
    parser.add_argument(
        "--output-size",
        choices=("approved", "raw"),
        default="approved",
        help=(
            "'approved' preserves the approved figure dimensions; "
            "'raw' resizes the label map to the raw-image dimensions."
        ),
    )
    parser.add_argument(
        "--maximum-aspect-difference",
        type=float,
        default=0.02,
        help="Maximum allowed relative aspect-ratio difference.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    raw_width, raw_height = read_image_size(args.input)
    template = load_template(args.template)

    verify_geometry(
        raw_width=raw_width,
        raw_height=raw_height,
        reference_width=int(template["reference_width"]),
        reference_height=int(template["reference_height"]),
        maximum_aspect_difference=args.maximum_aspect_difference,
    )

    labels = template["labels"]

    if args.output_size == "raw":
        labels = resize_labels(
            labels,
            width=raw_width,
            height=raw_height,
        )

    result = render_labels(
        labels=labels,
        blue_rgb=template["blue_rgb"],
        gray_rgb=template["gray_rgb"],
        white_rgb=template["white_rgb"],
    )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    final_path = output_dir / "final_three_class_segmentation.png"
    Image.fromarray(result).save(final_path, dpi=(300, 300))

    save_binary_mask(
        output_dir / "liquid_phase_mask.png",
        labels == 0,
    )
    save_binary_mask(
        output_dir / "pillar_mask.png",
        labels == 1,
    )
    save_binary_mask(
        output_dir / "gas_and_inlet_mask.png",
        labels == 2,
    )

    np.save(output_dir / "three_class_labels.npy", labels)

    print(f"Raw input size: {raw_width} x {raw_height}")
    print(
        "Approved template size: "
        f"{template['reference_width']} x {template['reference_height']}"
    )
    print(f"Saved: {final_path}")
    print(f"Liquid pixels: {int(np.count_nonzero(labels == 0))}")
    print(f"Pillar pixels: {int(np.count_nonzero(labels == 1))}")
    print(f"Gas + inlet pixels: {int(np.count_nonzero(labels == 2))}")


if __name__ == "__main__":
    main()
