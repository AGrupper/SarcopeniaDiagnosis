# muscle_segmentation.py
"""
Advanced muscle segmentation that properly distinguishes muscle from organs.
Uses anatomical location and shape constraints.
"""

import numpy as np
import SimpleITK as sitk
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology, measure, filters
from scipy.ndimage import distance_transform_edt


def segment_muscle_anatomically(patient_id: str, l3_index: int = 189, visualize: bool = True):
    """
    Segment muscle using anatomical constraints specific to L3 level.

    At L3, we expect:
    1. Psoas muscles: bilateral, posterior-lateral, round/oval
    2. Paraspinal muscles: along the spine posteriorly
    3. Abdominal wall muscles: thin layer around the periphery
    4. NOT organs (liver, kidneys, intestines) which are central
    """

    # Load volume
    base_dir = Path(r"C:\CT_Project")
    volume_path = base_dir / "data_preproc" / f"{patient_id}_iso_norm.nii.gz"

    image = sitk.ReadImage(str(volume_path))
    array = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()

    # Get L3 slice
    slice_2d = array[l3_index, :, :]
    h, w = slice_2d.shape
    center_y, center_x = h // 2, w // 2

    print(f"Processing {patient_id} slice {l3_index}")
    print(f"Slice shape: {slice_2d.shape}")
    print(f"Value range: [{slice_2d.min():.3f}, {slice_2d.max():.3f}]")

    # Step 1: Create body mask
    body_mask = slice_2d > 0.1
    body_mask = ndimage.binary_fill_holes(body_mask)
    body_mask = morphology.remove_small_objects(body_mask, min_size=5000)

    # Keep largest component
    labels = measure.label(body_mask)
    if labels.max() > 0:
        largest = np.argmax(np.bincount(labels.flat)[1:]) + 1
        body_mask = labels == largest

    # Step 2: Find the spine (high intensity central structure)
    # The spine should be roughly central and have high intensity
    spine_region = np.zeros_like(slice_2d, dtype=bool)
    spine_search_radius = 50

    # Look for high intensity in central region
    for y in range(center_y - spine_search_radius, center_y + spine_search_radius):
        for x in range(center_x - spine_search_radius, center_x + spine_search_radius):
            if 0 <= y < h and 0 <= x < w:
                if slice_2d[y, x] > 0.75:  # Bone intensity
                    spine_region[y, x] = True

    # Dilate spine region slightly
    spine_region = morphology.binary_dilation(spine_region, morphology.disk(5))

    # Step 3: Identify muscle intensity range
    # Muscle: -29 to 150 HU → 0.269 to 0.667 normalized
    muscle_min = 0.269
    muscle_max = 0.667

    # Initial muscle candidate
    muscle_candidate = (slice_2d >= muscle_min) & (slice_2d <= muscle_max) & body_mask

    # Step 4: Remove organs using anatomical location
    # Organs are typically in the anterior-central region

    # Create distance map from body boundary
    body_dist = distance_transform_edt(body_mask)
    max_dist = body_dist.max()

    # Organs are usually more than 30-40 pixels from the boundary
    deep_tissue = body_dist > 40

    # Create anterior mask (front half of body)
    anterior_mask = np.zeros_like(slice_2d, dtype=bool)
    anterior_mask[:center_y, :] = True

    # Organs are in the deep anterior region
    organ_region = deep_tissue & anterior_mask & ~spine_region

    # Remove organ region from muscle candidates
    muscle_mask = muscle_candidate & ~organ_region

    # Step 5: Identify specific muscle groups

    # 5a. Psoas muscles (bilateral, posterior to organs)
    psoas_mask = np.zeros_like(slice_2d, dtype=bool)

    # Left psoas (roughly)
    psoas_left_x = center_x - 60
    psoas_left_y = center_y + 20
    for y in range(max(0, psoas_left_y - 30), min(h, psoas_left_y + 30)):
        for x in range(max(0, psoas_left_x - 25), min(w, psoas_left_x + 25)):
            dist_to_center = np.sqrt((x - psoas_left_x) ** 2 + (y - psoas_left_y) ** 2)
            if dist_to_center < 25:  # Circular region
                if muscle_min <= slice_2d[y, x] <= muscle_max:
                    psoas_mask[y, x] = True

    # Right psoas (mirror)
    psoas_right_x = center_x + 60
    psoas_right_y = center_y + 20
    for y in range(max(0, psoas_right_y - 30), min(h, psoas_right_y + 30)):
        for x in range(max(0, psoas_right_x - 25), min(w, psoas_right_x + 25)):
            dist_to_center = np.sqrt((x - psoas_right_x) ** 2 + (y - psoas_right_y) ** 2)
            if dist_to_center < 25:
                if muscle_min <= slice_2d[y, x] <= muscle_max:
                    psoas_mask[y, x] = True

    # 5b. Paraspinal muscles (posterior, along spine)
    paraspinal_mask = np.zeros_like(slice_2d, dtype=bool)

    # These are posterior to spine
    posterior_spine_region = np.zeros_like(slice_2d, dtype=bool)
    posterior_spine_region[center_y:, center_x - 80:center_x + 80] = True

    paraspinal_candidate = muscle_candidate & posterior_spine_region & ~spine_region
    paraspinal_mask = paraspinal_candidate

    # 5c. Abdominal wall muscles (periphery)
    # These are within 30 pixels of the body boundary
    peripheral_region = (body_dist > 0) & (body_dist < 30)
    abdominal_wall = muscle_candidate & peripheral_region

    # Combine all muscle regions
    final_muscle_mask = psoas_mask | paraspinal_mask | abdominal_wall

    # Clean up
    final_muscle_mask = morphology.remove_small_objects(final_muscle_mask, min_size=100)
    final_muscle_mask = morphology.remove_small_holes(final_muscle_mask, area_threshold=50)

    # Calculate metrics
    pixel_area_cm2 = (spacing[0] * spacing[1]) / 100
    muscle_area = np.sum(final_muscle_mask) * pixel_area_cm2

    if np.sum(final_muscle_mask) > 0:
        muscle_values = slice_2d[final_muscle_mask]
        hu_values = muscle_values * 450 - 150
        mean_hu = np.mean(hu_values)
        std_hu = np.std(hu_values)

        # Calculate areas for each muscle group
        psoas_area = np.sum(psoas_mask) * pixel_area_cm2
        paraspinal_area = np.sum(paraspinal_mask) * pixel_area_cm2
        abdominal_area = np.sum(abdominal_wall) * pixel_area_cm2
    else:
        mean_hu = 0
        std_hu = 0
        psoas_area = 0
        paraspinal_area = 0
        abdominal_area = 0

    print(f"\nResults:")
    print(f"  Total muscle area: {muscle_area:.1f} cm²")
    print(f"    - Psoas: {psoas_area:.1f} cm²")
    print(f"    - Paraspinal: {paraspinal_area:.1f} cm²")
    print(f"    - Abdominal wall: {abdominal_area:.1f} cm²")
    print(f"  Mean HU: {mean_hu:.1f}")
    print(f"  Std HU: {std_hu:.1f}")

    if visualize:
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original
        axes[0, 0].imshow(slice_2d, cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title('Original L3 Slice')
        axes[0, 0].axis('off')

        # Body and organ regions
        axes[0, 1].imshow(slice_2d, cmap='gray', vmin=0, vmax=1)
        axes[0, 1].contour(organ_region, colors='yellow', linewidths=2)
        axes[0, 1].contour(spine_region, colors='cyan', linewidths=2)
        axes[0, 1].set_title('Organ (yellow) & Spine (cyan) Regions')
        axes[0, 1].axis('off')

        # Individual muscle groups
        muscle_groups = np.zeros((h, w, 3))
        muscle_groups[:, :, 0] = psoas_mask.astype(float)  # Red for psoas
        muscle_groups[:, :, 1] = paraspinal_mask.astype(float)  # Green for paraspinal
        muscle_groups[:, :, 2] = abdominal_wall.astype(float)  # Blue for abdominal

        axes[0, 2].imshow(muscle_groups)
        axes[0, 2].set_title('Muscle Groups\nR:Psoas G:Paraspinal B:Abdominal')
        axes[0, 2].axis('off')

        # Final muscle mask
        axes[1, 0].imshow(final_muscle_mask, cmap='hot')
        axes[1, 0].set_title(f'Final Muscle Mask\nArea: {muscle_area:.1f} cm²')
        axes[1, 0].axis('off')

        # Overlay on original
        slice_8bit = (slice_2d * 255).astype(np.uint8)
        rgb_image = cv2.cvtColor(slice_8bit, cv2.COLOR_GRAY2RGB)

        muscle_overlay = np.zeros_like(rgb_image)
        muscle_overlay[:, :, 0] = final_muscle_mask * 255
        muscle_overlay[:, :, 1] = final_muscle_mask * 100

        overlay = cv2.addWeighted(rgb_image, 0.7, muscle_overlay, 0.3, 0)

        # Add contours
        contours, _ = cv2.findContours(final_muscle_mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Segmentation Overlay')
        axes[1, 1].axis('off')

        # Comparison with initial threshold
        axes[1, 2].imshow(slice_2d, cmap='gray', vmin=0, vmax=1)
        axes[1, 2].contour(muscle_candidate, colors='red', linewidths=1, alpha=0.5)
        axes[1, 2].contour(final_muscle_mask, colors='green', linewidths=2)
        axes[1, 2].set_title('Initial (red) vs Final (green)')
        axes[1, 2].axis('off')

        plt.suptitle(f'Advanced Muscle Segmentation for {patient_id} at L3 (slice {l3_index})')
        plt.tight_layout()
        plt.show()

    return final_muscle_mask, muscle_area, mean_hu


if __name__ == "__main__":
    import sys

    patient_id = sys.argv[1] if len(sys.argv) > 1 else "patient01"
    l3_index = int(sys.argv[2]) if len(sys.argv) > 2 else 189

    mask, area, hu = segment_muscle_anatomically(patient_id, l3_index)

    print(f"\n" + "=" * 50)
    print("Expected ranges for healthy adults at L3:")
    print(f"  Total muscle area:")
    print(f"    - Males: 150-180 cm²")
    print(f"    - Females: 110-140 cm²")
    print(f"  Mean HU: 30-50 HU (healthy muscle)")
    print(f"  Lower values indicate fatty infiltration")
    print("=" * 50)