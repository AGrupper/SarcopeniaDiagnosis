# test_segmentation_fix.py
"""
Test script to fix the muscle segmentation issue.
The problem was that the thresholding was too broad.
"""

import numpy as np
import SimpleITK as sitk
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology, measure


def segment_muscle_correctly(patient_id: str, l3_index: int = 189):
    """
    Properly segment muscle at L3 with correct HU thresholding.
    """
    # Load the preprocessed volume
    base_dir = Path(r"C:\CT_Project")
    volume_path = base_dir / "data_preproc" / f"{patient_id}_iso_norm.nii.gz"

    # Read image
    image = sitk.ReadImage(str(volume_path))
    array = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()

    # Get L3 slice
    slice_2d = array[l3_index, :, :]

    print(f"Slice stats: min={slice_2d.min():.3f}, max={slice_2d.max():.3f}, mean={slice_2d.mean():.3f}")

    # Your preprocessing normalized from [-150, 300] to [0, 1]
    # So muscle HU range [-29, 150] becomes:
    # Lower: (-29 - (-150)) / 450 = 121/450 = 0.269
    # Upper: (150 - (-150)) / 450 = 300/450 = 0.667

    muscle_min = 0.269
    muscle_max = 0.667

    # Create body mask first (to exclude background)
    body_mask = slice_2d > 0.05  # Just above air
    body_mask = ndimage.binary_fill_holes(body_mask)
    body_mask = morphology.remove_small_objects(body_mask, min_size=1000)

    # Keep only largest connected component (the body)
    labels = measure.label(body_mask)
    if labels.max() > 0:
        largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
        body_mask = labels == largest_label

    # Apply muscle thresholding WITHIN the body only
    muscle_mask = (slice_2d >= muscle_min) & (slice_2d <= muscle_max) & body_mask

    # Remove bone regions (they have higher intensity)
    # Bone would be > 150 HU, which is > 0.667 in normalized values
    bone_threshold = 0.667
    bone_mask = slice_2d > bone_threshold

    # Exclude bone from muscle
    muscle_mask = muscle_mask & ~bone_mask

    # Also exclude the spine region (central circular high-intensity area)
    h, w = slice_2d.shape
    center_y, center_x = h // 2, w // 2

    # Create a mask for the central spine region
    y_grid, x_grid = np.ogrid[:h, :w]
    spine_radius = 40  # Approximate radius of spine in pixels
    spine_mask = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2 <= spine_radius ** 2

    # Remove spine area if it's high intensity
    spine_area = slice_2d[spine_mask]
    if np.mean(spine_area) > 0.7:  # If spine area is bright
        muscle_mask = muscle_mask & ~spine_mask

    # Remove visceral organs (they're usually in the center-anterior region)
    # This is a rough approximation - organs are usually anterior to spine
    organ_region = np.zeros_like(muscle_mask)
    organ_region[center_y - 60:center_y + 20, center_x - 80:center_x + 80] = True

    # Check if this region has intermediate intensity (organs)
    organ_area = slice_2d[organ_region]
    organ_mean = np.mean(organ_area)
    if 0.3 < organ_mean < 0.6:  # Organs have intermediate intensity
        # Remove areas that are too smooth (organs are more homogeneous than muscle)
        local_std = ndimage.generic_filter(slice_2d, np.std, size=5)
        smooth_mask = local_std < 0.05
        muscle_mask = muscle_mask & ~(smooth_mask & organ_region)

    # Clean up the mask
    muscle_mask = morphology.remove_small_objects(muscle_mask.astype(bool), min_size=50)
    muscle_mask = morphology.remove_small_holes(muscle_mask, area_threshold=50)

    # Calculate metrics
    pixel_area_cm2 = (spacing[0] * spacing[1]) / 100
    muscle_area = np.sum(muscle_mask) * pixel_area_cm2

    if np.sum(muscle_mask) > 0:
        muscle_values = slice_2d[muscle_mask]
        # Convert back to HU
        hu_values = muscle_values * 450 - 150
        mean_hu = np.mean(hu_values)
        std_hu = np.std(hu_values)
    else:
        mean_hu = 0
        std_hu = 0

    print(f"\nResults:")
    print(f"  Muscle area: {muscle_area:.1f} cm²")
    print(f"  Mean HU: {mean_hu:.1f}")
    print(f"  Std HU: {std_hu:.1f}")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(slice_2d, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original L3 Slice')
    axes[0].axis('off')

    # Muscle mask
    axes[1].imshow(muscle_mask, cmap='hot')
    axes[1].set_title(f'Muscle Mask\nArea: {muscle_area:.1f} cm²')
    axes[1].axis('off')

    # Overlay
    slice_8bit = (slice_2d * 255).astype(np.uint8)
    rgb_image = cv2.cvtColor(slice_8bit, cv2.COLOR_GRAY2RGB)
    muscle_overlay = np.zeros_like(rgb_image)
    muscle_overlay[:, :, 0] = muscle_mask * 255  # Red
    muscle_overlay[:, :, 1] = muscle_mask * 100  # Some green

    overlay = cv2.addWeighted(rgb_image, 0.7, muscle_overlay, 0.3, 0)

    # Add contours
    contours, _ = cv2.findContours(muscle_mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    axes[2].imshow(overlay)
    axes[2].set_title('Segmentation Overlay')
    axes[2].axis('off')

    plt.suptitle(f'Muscle Segmentation for {patient_id} at L3 (slice {l3_index})')
    plt.tight_layout()
    plt.show()

    return muscle_mask, muscle_area, mean_hu


if __name__ == "__main__":
    import sys

    patient_id = sys.argv[1] if len(sys.argv) > 1 else "patient01"
    l3_index = int(sys.argv[2]) if len(sys.argv) > 2 else 189

    mask, area, hu = segment_muscle_correctly(patient_id, l3_index)

    print(f"\nExpected values for healthy adult at L3:")
    print(f"  Muscle area: 100-200 cm² (varies by sex and build)")
    print(f"  Mean HU: 30-50 HU (healthy muscle)")