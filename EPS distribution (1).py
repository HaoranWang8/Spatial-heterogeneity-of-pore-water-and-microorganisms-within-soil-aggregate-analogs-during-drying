



from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


CHANNEL_INDEX = 2
CHANNEL_NAME = "red"
SIGNAL_NAME = "eps_cona_fluorescence"


def parse_radius_edges(value: str) -> list[int]:
    """将逗号分隔的半径边界解析为严格递增的非负整数列表。"""
    try:
        edges = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("半径边界必须是逗号分隔的整数。") from exc

    if len(edges) < 2:
        raise argparse.ArgumentTypeError("至少需要两个半径边界。")
    if edges[0] != 0:
        raise argparse.ArgumentTypeError("第一个半径边界必须为 0，以包含最内层区域。")
    if any(edge < 0 for edge in edges):
        raise argparse.ArgumentTypeError("半径边界不能为负数。")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise argparse.ArgumentTypeError("半径边界必须严格递增。")
    return edges


def load_image(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取图像并提取红色通道。"""
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")

    if image.ndim == 2:
        intensity = image
    elif image.ndim == 3 and image.shape[2] >= 3:
        # OpenCV 的彩色通道顺序是 BGR；索引 2 为红色通道。
        intensity = image[:, :, CHANNEL_INDEX]
        image = image[:, :, :3]
    else:
        raise ValueError(f"不支持的图像维度：{image.shape}")

    return image, intensity


def quarter_circle_mask(
    shape: tuple[int, int],
    center: tuple[int, int],
    radius: int,
) -> np.ndarray:
    """创建从图像左下角向右上方展开的实心四分之一圆掩膜。"""
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    if radius <= 0:
        return mask

    cv2.ellipse(
        mask,
        center,
        (radius, radius),
        0,
        0,
        -90,
        255,
        -1,
    )
    return mask


def safe_statistics(values: np.ndarray) -> dict[str, float | int]:
    """计算区域统计量；空区域返回 NaN。"""
    count = int(values.size)
    if count == 0:
        return {
            "valid_pixel_count": 0,
            "mean_intensity": np.nan,
            "median_intensity": np.nan,
            "std_intensity": np.nan,
            "q1_intensity": np.nan,
            "q3_intensity": np.nan,
            "iqr_intensity": np.nan,
            "min_intensity": np.nan,
            "max_intensity": np.nan,
            "cv_percent": np.nan,
        }

    values_float = values.astype(np.float64)
    mean_value = float(np.mean(values_float))
    std_value = float(np.std(values_float, ddof=0))
    q1, q3 = np.percentile(values_float, [25, 75])

    return {
        "valid_pixel_count": count,
        "mean_intensity": mean_value,
        "median_intensity": float(np.median(values_float)),
        "std_intensity": std_value,
        "q1_intensity": float(q1),
        "q3_intensity": float(q3),
        "iqr_intensity": float(q3 - q1),
        "min_intensity": float(np.min(values_float)),
        "max_intensity": float(np.max(values_float)),
        "cv_percent": float(std_value / mean_value * 100) if mean_value != 0 else np.nan,
    }


def analyze_image(
    image_path: Path,
    output_dir: Path,
    threshold: float,
    radius_edges: Iterable[int],
    pixel_size_um: float,
    export_pixel_csv: bool,
) -> None:
    """执行径向荧光分析并保存全部结果。"""
    image, intensity = load_image(image_path)
    height, width = intensity.shape

    # 与原脚本保持一致：圆心位于图像左下角坐标 (0, height)。
    center = (0, height)
    radius_edges = list(radius_edges)

    output_dir.mkdir(parents=True, exist_ok=True)
    region_image_dir = output_dir / "region_visualizations"
    region_image_dir.mkdir(exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    pixel_frames: list[pd.DataFrame] = []

    region_label_mask = np.zeros((height, width), dtype=np.uint8)
    valid_signal_mask = np.zeros((height, width), dtype=np.uint8)

    for region_index, (inner_radius, outer_radius) in enumerate(
        zip(radius_edges[:-1], radius_edges[1:]),
        start=1,
    ):
        outer_mask = quarter_circle_mask(
            (height, width), center, outer_radius
        ).astype(bool)
        inner_mask = quarter_circle_mask(
            (height, width), center, inner_radius
        ).astype(bool)

        # 几何分区先由“大圆减小圆”得到，再统一应用强度阈值。
        geometric_region_mask = outer_mask & (~inner_mask)
        valid_region_mask = geometric_region_mask & (intensity >= threshold)

        region_label_mask[geometric_region_mask] = region_index
        valid_signal_mask[valid_region_mask] = 255

        y_coords, x_coords = np.where(valid_region_mask)
        region_values = intensity[valid_region_mask]

        stats = safe_statistics(region_values)
        geometric_pixel_count = int(np.count_nonzero(geometric_region_mask))

        summary_rows.append(
            {
                "image": image_path.name,
                "signal": SIGNAL_NAME,
                "channel": CHANNEL_NAME,
                "threshold": threshold,
                "region": f"subregion_{region_index}",
                "region_index": region_index,
                "inner_radius_px": inner_radius,
                "outer_radius_px": outer_radius,
                "inner_radius_um": inner_radius * pixel_size_um,
                "outer_radius_um": outer_radius * pixel_size_um,
                "geometric_pixel_count": geometric_pixel_count,
                "excluded_by_threshold_count": geometric_pixel_count
                - stats["valid_pixel_count"],
                **stats,
            }
        )

        if export_pixel_csv and region_values.size > 0:
            pixel_frames.append(
                pd.DataFrame(
                    {
                        "image": image_path.name,
                        "signal": SIGNAL_NAME,
                        "channel": CHANNEL_NAME,
                        "region": f"subregion_{region_index}",
                        "region_index": region_index,
                        "inner_radius_px": inner_radius,
                        "outer_radius_px": outer_radius,
                        "x": x_coords,
                        "y": y_coords,
                        "intensity": region_values,
                    }
                )
            )

        region_visualization = cv2.bitwise_and(
            image, image, mask=valid_region_mask.astype(np.uint8) * 255
        )
        cv2.imwrite(
            str(
                region_image_dir
                / f"subregion_{region_index}_{inner_radius}_{outer_radius}px.tif"
            ),
            region_visualization,
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        output_dir / "eps_radial_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if export_pixel_csv:
        if pixel_frames:
            pixel_df = pd.concat(pixel_frames, ignore_index=True)
        else:
            pixel_df = pd.DataFrame(
                columns=[
                    "image",
                    "signal",
                    "channel",
                    "region",
                    "region_index",
                    "inner_radius_px",
                    "outer_radius_px",
                    "x",
                    "y",
                    "intensity",
                ]
            )
        pixel_df.to_csv(
            output_dir / "eps_pixel_intensities.csv",
            index=False,
            encoding="utf-8-sig",
        )

    cv2.imwrite(
        str(output_dir / "radial_region_label_mask.tif"),
        region_label_mask,
    )
    cv2.imwrite(
        str(output_dir / "valid_eps_signal_mask.tif"),
        valid_signal_mask,
    )

    overlay = image.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    for radius in radius_edges[1:]:
        cv2.ellipse(
            overlay,
            center,
            (radius, radius),
            0,
            0,
            -90,
            (255, 255, 255),
            1,
        )
    cv2.imwrite(str(output_dir / "radial_boundaries_overlay.tif"), overlay)

    print("\nEPS ConA 红色荧光径向分析完成：")
    print(summary_df.to_string(index=False))
    print(f"\n结果目录：{output_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="分析红色 EPS ConA 荧光在 7 个径向子区中的分布。"
    )
    parser.add_argument("image", type=Path, help="输入 TIFF 图像路径。")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("eps_radial_output"),
        help="输出目录，默认：eps_radial_output",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10,
        help="有效荧光像素的最低强度阈值，默认：10",
    )
    parser.add_argument(
        "--radius-edges",
        type=parse_radius_edges,
        default=parse_radius_edges("0,80,160,240,320,400,480,560"),
        help="径向边界，逗号分隔，默认：0,80,160,240,320,400,480,560",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=2.23,
        help="像素尺寸（μm/pixel），默认：2.23",
    )
    parser.add_argument(
        "--no-pixel-csv",
        action="store_true",
        help="不导出逐像素强度 CSV，以减少文件体积。",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.threshold < 0:
        raise ValueError("阈值不能小于 0。")
    if args.pixel_size <= 0:
        raise ValueError("像素尺寸必须大于 0。")

    analyze_image(
        image_path=args.image,
        output_dir=args.output_dir,
        threshold=args.threshold,
        radius_edges=args.radius_edges,
        pixel_size_um=args.pixel_size,
        export_pixel_csv=not args.no_pixel_csv,
    )


if __name__ == "__main__":
    main()

