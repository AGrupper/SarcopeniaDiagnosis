# muscle_segmentation.py
"""
Final muscle segmentation module with proper saving functionality.
"""

import numpy as np
import SimpleITK as sitk
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology, measure, filters
from scipy.ndimage import distance_transform_edt
import json


def segment_muscle_anatomically(patient_id: str, l3_index: int = None,
                                base_dir: Path = Path(r"C:\CT_Project"),
                                visualize: bool = True, save_results: bool = True):
    """
    Segment muscle using anatomical constraints specific to L3 level.
    """

    # If L3 index not provided, try to read from manual override file
    if l3_index is None:
        manual_csv = base_dir / "meta" / "l3_slices_manual.csv"
        if manual_csv.exists():
            import csv
            with open(manual_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['patient_id'] == patient_id:
                        l3_index = int(row['slice_index'])
                        break

        if l3_index is None:
            raise ValueError(f"No L3 index found for {patient_id}. Please run debug_l3_detection.py first.")

    # Load volume
    volume_path = base_dir / "data_preproc" / f"{patient_id}_iso_norm.nii.gz"

    if not volume_path.exists():
        raise FileNotFoundError(f"Preprocessed volume not found: {volume_path}")

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

    # Step 2: Find the spine
    spine_region = np.zeros_like(slice_2d, dtype=bool)
    spine_search_radius = 50

    for y in range(center_y - spine_search_radius, center_y + spine_search_radius):
        for x in range(center_x - spine_search_radius, center_x + spine_search_radius):
            if 0 <= y < h and 0 <= x < w:
                if slice_2d[y, x] > 0.75:  # Bone intensity
                    spine_region[y, x] = True

    spine_region = morphology.binary_dilation(spine_region, morphology.disk(5))

    # Step 3: Identify muscle intensity range
    muscle_min = 0.269  # -29 HU normalized
    muscle_max = 0.667  # 150 HU normalized

    # Initial muscle candidate
    muscle_candidate = (slice_2d >= muscle_min) & (slice_2d <= muscle_max) & body_mask

    # Step 4: Remove organs using anatomical location
    body_dist = distance_transform_edt(body_mask)
    deep_tissue = body_dist > 40

    anterior_mask = np.zeros_like(slice_2d, dtype=bool)
    anterior_mask[:center_y, :] = True

    organ_region = deep_tissue & anterior_mask & ~spine_region
    muscle_mask = muscle_candidate & ~organ_region

    # Step 5: Identify specific muscle groups

    # 5a. Psoas muscles
    psoas_mask = np.zeros_like(slice_2d, dtype=bool)

    # Left psoas
    psoas_left_x = center_x - 60
    psoas_left_y = center_y + 20
    for y in range(max(0, psoas_left_y - 30), min(h, psoas_left_y + 30)):
        for x in range(max(0, psoas_left_x - 25), min(w, psoas_left_x + 25)):
            dist_to_center = np.sqrt((x - psoas_left_x) ** 2 + (y - psoas_left_y) ** 2)
            if dist_to_center < 25:
                if muscle_min <= slice_2d[y, x] <= muscle_max:
                    psoas_mask[y, x] = True

    # Right psoas
    psoas_right_x = center_x + 60
    psoas_right_y = center_y + 20
    for y in range(max(0, psoas_right_y - 30), min(h, psoas_right_y + 30)):
        for x in range(max(0, psoas_right_x - 25), min(w, psoas_right_x + 25)):
            dist_to_center = np.sqrt((x - psoas_right_x) ** 2 + (y - psoas_right_y) ** 2)
            if dist_to_center < 25:
                if muscle_min <= slice_2d[y, x] <= muscle_max:
                    psoas_mask[y, x] = True

    # 5b. Paraspinal muscles
    paraspinal_mask = np.zeros_like(slice_2d, dtype=bool)
    posterior_spine_region = np.zeros_like(slice_2d, dtype=bool)
    posterior_spine_region[center_y:, center_x - 80:center_x + 80] = True

    paraspinal_candidate = muscle_candidate & posterior_spine_region & ~spine_region
    paraspinal_mask = paraspinal_candidate

    # 5c. Abdominal wall muscles
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
        mean_hu = float(np.mean(hu_values))
        std_hu = float(np.std(hu_values))
        low_atten_percent = float((np.sum(hu_values < 30) / len(hu_values)) * 100)

        psoas_area = np.sum(psoas_mask) * pixel_area_cm2
        paraspinal_area = np.sum(paraspinal_mask) * pixel_area_cm2
        abdominal_area = np.sum(abdominal_wall) * pixel_area_cm2
    else:
        mean_hu = 0
        std_hu = 0
        low_atten_percent = 0
        psoas_area = 0
        paraspinal_area = 0
        abdominal_area = 0

    print(f"\nResults:")
    print(f"  Total muscle area: {muscle_area:.1f} cm²")
    print(f"    - Psoas: {psoas_area:.1f} cm²")
    print(f"    - Paraspinal: {paraspinal_area:.1f} cm²")
    print(f"    - Abdominal wall: {abdominal_area:.1f} cm²")
    print(f"  Mean HU: {mean_hu:.1f}")
    print(f"  Low attenuation %: {low_atten_percent:.1f}%")

    # Save results if requested
    if save_results:
        output_dir = base_dir / "results" / patient_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_dict = {
            'patient_id': patient_id,
            'l3_slice_index': int(l3_index),
            'total_muscle_area_cm2': float(muscle_area),
            'psoas_area_cm2': float(psoas_area),
            'paraspinal_area_cm2': float(paraspinal_area),
            'abdominal_wall_area_cm2': float(abdominal_area),
            'mean_muscle_hu': float(mean_hu),
            'std_muscle_hu': float(std_hu),
            'low_attenuation_percentage': float(low_atten_percent),
            'volume_shape': [int(x) for x in array.shape],
            'spacing_mm': [float(x) for x in spacing]
        }

        with open(output_dir / "muscle_metrics.json", 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        print(f"  Saved metrics to: {output_dir / 'muscle_metrics.json'}")

        # Save muscle mask
        mask_3d = np.zeros((1, final_muscle_mask.shape[0], final_muscle_mask.shape[1]), dtype=np.uint8)
        mask_3d[0, :, :] = final_muscle_mask

        mask_img = sitk.GetImageFromArray(mask_3d)
        mask_img.SetSpacing((spacing[0], spacing[1], spacing[2]))
        mask_img.SetOrigin(image.GetOrigin())
        sitk.WriteImage(mask_img, str(output_dir / "muscle_mask_l3.nii.gz"))

        # Save segmentation overlay
        slice_8bit = (slice_2d * 255).astype(np.uint8)
        rgb_image = cv2.cvtColor(slice_8bit, cv2.COLOR_GRAY2RGB)

        muscle_overlay = np.zeros_like(rgb_image)
        muscle_overlay[:, :, 0] = final_muscle_mask * 255
        muscle_overlay[:, :, 1] = final_muscle_mask * 100

        overlay = cv2.addWeighted(rgb_image, 0.7, muscle_overlay, 0.3, 0)

        contours, _ = cv2.findContours(final_muscle_mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        cv2.imwrite(str(output_dir / "segmentation_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

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
        muscle_groups[:, :, 0] = psoas_mask.astype(float)
        muscle_groups[:, :, 1] = paraspinal_mask.astype(float)
        muscle_groups[:, :, 2] = abdominal_wall.astype(float)

        axes[0, 2].imshow(muscle_groups)
        axes[0, 2].set_title('Muscle Groups\nR:Psoas G:Paraspinal B:Abdominal')
        axes[0, 2].axis('off')

        # Final muscle mask
        axes[1, 0].imshow(final_muscle_mask, cmap='hot')
        axes[1, 0].set_title(f'Final Muscle Mask\nArea: {muscle_area:.1f} cm²')
        axes[1, 0].axis('off')

        # Overlay on original
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Segmentation Overlay')
        axes[1, 1].axis('off')

        # Comparison with initial threshold
        axes[1, 2].imshow(slice_2d, cmap='gray', vmin=0, vmax=1)
        axes[1, 2].contour(muscle_candidate, colors='red', linewidths=1, alpha=0.5)
        axes[1, 2].contour(final_muscle_mask, colors='green', linewidths=2)
        axes[1, 2].set_title('Initial (red) vs Final (green)')
        axes[1, 2].axis('off')

        plt.suptitle(f'Muscle Segmentation for {patient_id} at L3 (slice {l3_index})')
        plt.tight_layout()

        if save_results:
            plt.savefig(output_dir / "segmentation_visualization.png", dpi=150, bbox_inches='tight')

        plt.show()

    return final_muscle_mask, muscle_area, mean_hu


if __name__ == "__main__":
    import sys

    patient_id = sys.argv[1] if len(sys.argv) > 1 else "patient01"
    l3_index = int(sys.argv[2]) if len(sys.argv) > 2 else None

    mask, area, hu = segment_muscle_anatomically(patient_id, l3_index)

    print(f"\n" + "=" * 50)
    print("Expected ranges for healthy adults at L3:")
    print(f"  Total muscle area:")
    print(f"    - Males: 150-180 cm²")
    print(f"    - Females: 110-140 cm²")
    print(f"  Mean HU: 30-50 HU (healthy muscle)")
    print("=" * 50)