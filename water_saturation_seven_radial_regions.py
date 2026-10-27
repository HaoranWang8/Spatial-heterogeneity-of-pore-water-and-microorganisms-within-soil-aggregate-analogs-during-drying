import os
import cv2
import numpy as np
import pandas as pd

# ==================== 1. Configuration parameters ====================
folder = r"C:\Users\30869\Desktop\2"  # Folder containing segmented images
save_xlsx = r"C:\Users\30869\Desktop\blue_white_ratio_7_radial_regions.xlsx"

# Boundaries of the seven radial regions:
# 0–80、80–160、160–240、240–320、
# 320–400、400–480、480–560 px
radius_edges = [0, 80, 160, 240, 320, 400, 480, 560]

# Use the lower-left corner of the image as the center of the quarter circle
start_angle = 0
end_angle = -90

# Color ranges in the segmented image (OpenCV HSV)
# Deep blue: liquid phase
blue_lower = np.array([100, 50, 50], dtype=np.uint8)
blue_upper = np.array([130, 255, 255], dtype=np.uint8)

# White: gas phase
white_lower = np.array([0, 0, 200], dtype=np.uint8)
white_upper = np.array([180, 30, 255], dtype=np.uint8)

# Image numbers to process
first_image = 1
last_image = 16

all_data = []


# ==================== 2. Helper functions ====================
def create_quarter_circle_mask(
    image_shape: tuple[int, int],
    center: tuple[int, int],
    radius: int,
) -> np.ndarray:
    """
    Create a quarter-circle mask centered at the specified point with radius `radius`.

    Parameters
    ----------
    image_shape : (height, width)
        Image dimensions.
    center : (x, y)
        Center of the quarter circle.
    radius : int
        Circle radius in pixels.

    Returns
    -------
    mask : np.ndarray
        Binary mask: 255 inside the region and 0 outside.
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    if radius <= 0:
        return mask

    cv2.ellipse(
        mask,
        center,
        (radius, radius),
        0,
        start_angle,
        end_angle,
        255,
        -1,
    )
    return mask


def create_radial_ring_mask(
    image_shape: tuple[int, int],
    center: tuple[int, int],
    inner_radius: int,
    outer_radius: int,
) -> np.ndarray:
    """
    Create a quarter-annulus mask spanning inner_radius–outer_radius.
    """
    outer_mask = create_quarter_circle_mask(
        image_shape=image_shape,
        center=center,
        radius=outer_radius,
    )

    if inner_radius == 0:
        return outer_mask

    inner_mask = create_quarter_circle_mask(
        image_shape=image_shape,
        center=center,
        radius=inner_radius,
    )

    return cv2.subtract(outer_mask, inner_mask)


def calculate_water_saturation(
    blue_mask: np.ndarray,
    white_mask: np.ndarray,
    region_mask: np.ndarray,
) -> dict:
    """
    Calculate two-dimensional pore-water saturation within the specified radial region.

    Calculation formula:
        blue pixels / (blue pixels + white pixels) × 100%

    where:
        blue = liquid phase
        white = gas phase
        gray pillars are excluded from both the numerator and denominator
    """
    blue_in_region = cv2.bitwise_and(
        blue_mask,
        blue_mask,
        mask=region_mask,
    )
    white_in_region = cv2.bitwise_and(
        white_mask,
        white_mask,
        mask=region_mask,
    )

    blue_pixels = int(np.count_nonzero(blue_in_region))
    white_pixels = int(np.count_nonzero(white_in_region))
    pore_pixels = blue_pixels + white_pixels

    if pore_pixels > 0:
        water_saturation = blue_pixels / pore_pixels * 100.0
    else:
        water_saturation = np.nan

    return {
        "water_saturation": water_saturation,
        "blue_pixels": blue_pixels,
        "white_pixels": white_pixels,
        "pore_pixels": pore_pixels,
    }


# Names of the seven radial regions
region_names = [
    f"Region_{radius_edges[index]}_{radius_edges[index + 1]}"
    for index in range(len(radius_edges) - 1)
]


# ==================== 3. Batch image processing ====================
print(f"Starting processing of {first_image}–{last_image}  images...")
print("Calculation formula: liquid-phase area / (liquid-phase area + gas-phase area) × 100%")
print(f"Radial boundaries:{radius_edges}")
print(f"Divided into {len(region_names)}  radial regions:")
for name in region_names:
    print(f"  {name}")

for image_number in range(first_image, last_image + 1):
    filename = f"{image_number}.tif"
    image_path = os.path.join(folder, filename)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        print(f"Unable to read {filename}; skipped.")
        continue

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Liquid- and gas-phase masks
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)

    height, width = image.shape[:2]

    # Use the lower-left corner as the center for radial partitioning
    center = (0, height)

    result_row = {"File": filename}

    for region_index, region_name in enumerate(region_names):
        inner_radius = radius_edges[region_index]
        outer_radius = radius_edges[region_index + 1]

        region_mask = create_radial_ring_mask(
            image_shape=(height, width),
            center=center,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
        )

        result = calculate_water_saturation(
            blue_mask=blue_mask,
            white_mask=white_mask,
            region_mask=region_mask,
        )

        result_row[f"{region_name}_water_saturation_percent"] = (
            result["water_saturation"]
        )
        result_row[f"{region_name}_blue_pixels"] = result["blue_pixels"]
        result_row[f"{region_name}_white_pixels"] = result["white_pixels"]
        result_row[f"{region_name}_pore_pixels"] = result["pore_pixels"]

    # Optional: calculate overall two-dimensional pore-water saturation across the full 0–560 px range
    total_mask = create_quarter_circle_mask(
        image_shape=(height, width),
        center=center,
        radius=radius_edges[-1],
    )

    total_result = calculate_water_saturation(
        blue_mask=blue_mask,
        white_mask=white_mask,
        region_mask=total_mask,
    )

    result_row["Total_0_560_water_saturation_percent"] = (
        total_result["water_saturation"]
    )
    result_row["Total_0_560_blue_pixels"] = total_result["blue_pixels"]
    result_row["Total_0_560_white_pixels"] = total_result["white_pixels"]
    result_row["Total_0_560_pore_pixels"] = total_result["pore_pixels"]

    all_data.append(result_row)

    total_value = total_result["water_saturation"]
    if np.isnan(total_value):
        total_text = "No valid blue or white pixels"
    else:
        total_text = f"{total_value:.2f}%"

    print(
        f"Completed: {filename}；"
        f"overall water saturation over 0–560 px: {total_text}；"
        f"liquid-phase pixels: {total_result['blue_pixels']}；"
        f"gas-phase pixels: {total_result['white_pixels']}"
    )


# ==================== 4. Save Excel ====================
if not all_data:
    raise RuntimeError(
        "No images were processed successfully. Please check the image path, filenames, and image format."
    )

df = pd.DataFrame(all_data)
df.to_excel(save_xlsx, index=False)

print(f"\nAll results have been saved to: {save_xlsx}")


# ==================== 5. Statistical summary for the seven radial regions ====================
print("\n" + "=" * 100)
print("Statistical summary of two-dimensional pore-water saturation for the seven radial regions")
print("=" * 100)

print(
    f"{'Region':<22}"
    f"{'Mean(%)':>12}"
    f"{'Standard deviation':>12}"
    f"{'Minimum(%)':>12}"
    f"{'Maximum(%)':>12}"
    f"{'Valid sample count':>12}"
)
print("-" * 100)

for region_name in region_names:
    column_name = f"{region_name}_water_saturation_percent"
    values = pd.to_numeric(df[column_name], errors="coerce").dropna()

    if values.empty:
        print(
            f"{region_name:<22}"
            f"{'NA':>12}"
            f"{'NA':>12}"
            f"{'NA':>12}"
            f"{'NA':>12}"
            f"{0:>12}"
        )
        continue

    print(
        f"{region_name:<22}"
        f"{values.mean():>11.2f}%"
        f"{values.std():>12.2f}"
        f"{values.min():>11.2f}%"
        f"{values.max():>11.2f}%"
        f"{len(values):>12}"
    )


# ==================== 6. Detailed results for each image ====================
print("\n" + "=" * 100)
print("Detailed results for the seven radial regions in each image")
print("=" * 100)

for _, row in df.iterrows():
    print(f"\n{row['File']}：")

    for region_name in region_names:
        saturation_column = (
            f"{region_name}_water_saturation_percent"
        )
        blue_column = f"{region_name}_blue_pixels"
        white_column = f"{region_name}_white_pixels"
        pore_column = f"{region_name}_pore_pixels"

        value = row[saturation_column]
        value_text = "NA" if pd.isna(value) else f"{value:.2f}%"

        print(
            f"  {region_name}: {value_text}; "
            f"liquid phase={int(row[blue_column])}, "
            f"gas phase={int(row[white_column])}, "
            f"valid pore pixels={int(row[pore_column])}"
        )


print("\n" + "=" * 100)
print("Analysis complete")
print(f"Number of successfully processed images: {len(df)}")
print(f"Radial boundaries:{radius_edges}")
print(f"Number of radial regions: {len(region_names)}")
print("Calculation formula: liquid-phase area / (liquid-phase area + gas-phase area) × 100%")
print("gray pillars are excluded from both the numerator and denominator")
print("=" * 100)
