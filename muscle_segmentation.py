# muscle_segmentation.py
"""
Skeletal muscle segmentation at L3 vertebral level.
Implements both traditional image processing and deep learning approaches.
"""
import csv
import numpy as np
import SimpleITK as sitk
import cv2
from pathlib import Path
from typing import Tuple, Dict, Optional
from scipy import ndimage
from skimage import morphology, measure
import json


class MuscleSegmentor:
    """
    Segments skeletal muscle in L3 CT slices using HU thresholding,
    morphological operations, and anatomical constraints.
    """

    # HU ranges for different tissues (before normalization)
    MUSCLE_HU_RANGE = (-29, 150)  # Skeletal muscle
    FAT_HU_RANGE = (-190, -30)  # Adipose tissue
    BONE_HU_RANGE = (150, 3000)  # Bone

    def __init__(self, preprocessed_volume_path: Path):
        """
        Initialize segmentor with preprocessed CT volume.

        Args:
            preprocessed_volume_path: Path to preprocessed NIfTI file
        """
        self.volume_path = preprocessed_volume_path
        self.image = sitk.ReadImage(str(preprocessed_volume_path))
        self.array = sitk.GetArrayFromImage(self.image)
        self.spacing = self.image.GetSpacing()

        # Convert HU ranges to normalized values (based on your preprocessing)
        # You normalized from [-150, 300] to [0, 1]
        self.muscle_norm_range = self.hu_to_normalized(self.MUSCLE_HU_RANGE)
        self.fat_norm_range = self.hu_to_normalized(self.FAT_HU_RANGE)
        self.bone_norm_range = self.hu_to_normalized(self.BONE_HU_RANGE)

    def hu_to_normalized(self, hu_range: Tuple[float, float]) -> Tuple[float, float]:
        """
        Convert HU range to normalized range based on preprocessing parameters.
        Your preprocessing: normalized from [-150, 300] to [0, 1]
        """
        hu_min_preprocess = -150
        hu_max_preprocess = 300

        norm_min = (max(hu_range[0], hu_min_preprocess) - hu_min_preprocess) / (hu_max_preprocess - hu_min_preprocess)
        norm_max = (min(hu_range[1], hu_max_preprocess) - hu_min_preprocess) / (hu_max_preprocess - hu_min_preprocess)

        return (max(0, norm_min), min(1, norm_max))

    def segment_body_mask(self, slice_2d: np.ndarray) -> np.ndarray:
        """
        Create a mask of the body outline, excluding air and the CT table.

        Args:
            slice_2d: 2D CT slice

        Returns:
            Binary mask of body region
        """
        # Threshold to separate body from air/background
        body_threshold = 0.1  # Slightly above air
        body_mask = slice_2d > body_threshold

        # Fill holes
        body_mask = ndimage.binary_fill_holes(body_mask)

        # Remove small objects (noise)
        body_mask = morphology.remove_small_objects(body_mask, min_size=1000)

        # Keep largest connected component (the body)
        labels = measure.label(body_mask)
        if labels.max() > 0:
            largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
            body_mask = labels == largest_label

        # Morphological closing to smooth boundaries
        kernel = morphology.disk(3)
        body_mask = morphology.binary_closing(body_mask, kernel)

        return body_mask.astype(np.uint8)

    def segment_muscle_threshold(self, slice_2d: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
        """
        Segment muscle using HU thresholding.

        Args:
            slice_2d: 2D CT slice (normalized)
            body_mask: Binary mask of body region

        Returns:
            Binary mask of muscle tissue
        """
        # Apply HU threshold for muscle
        muscle_mask = (slice_2d >= self.muscle_norm_range[0]) & \
                      (slice_2d <= self.muscle_norm_range[1])

        # Only keep muscle within body
        muscle_mask = muscle_mask & body_mask

        # Remove small objects (noise)
        muscle_mask = morphology.remove_small_objects(muscle_mask, min_size=50)

        return muscle_mask.astype(np.uint8)

    def identify_muscle_groups(self, muscle_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Identify and separate different muscle groups at L3 level.

        Expected muscle groups at L3:
        - Psoas major (left and right)
        - Quadratus lumborum
        - Erector spinae
        - Rectus abdominis
        - External/Internal obliques
        - Transversus abdominis

        Args:
            muscle_mask: Binary mask of all muscle tissue

        Returns:
            Dictionary of muscle group masks
        """
        muscle_groups = {}

        # Get image dimensions
        h, w = muscle_mask.shape
        center_y, center_x = h // 2, w // 2

        # Label connected components
        labeled_muscles = measure.label(muscle_mask)
        regions = measure.regionprops(labeled_muscles)

        # Classify muscle groups based on location
        for region in regions:
            centroid_y, centroid_x = region.centroid

            # Create mask for this region
            region_mask = (labeled_muscles == region.label)

            # Determine muscle group based on anatomical position
            # Relative to spine center
            if abs(centroid_x - center_x) < w * 0.15:  # Near center
                if centroid_y < center_y:  # Anterior
                    muscle_groups.setdefault('rectus_abdominis', np.zeros_like(muscle_mask))
                    muscle_groups['rectus_abdominis'] |= region_mask
                else:  # Posterior
                    muscle_groups.setdefault('erector_spinae', np.zeros_like(muscle_mask))
                    muscle_groups['erector_spinae'] |= region_mask

            elif centroid_x < center_x - w * 0.15:  # Left side
                if abs(centroid_y - center_y) < h * 0.2:  # Lateral
                    if region.area > 500:  # Large muscle
                        muscle_groups.setdefault('psoas_left', np.zeros_like(muscle_mask))
                        muscle_groups['psoas_left'] |= region_mask
                    else:
                        muscle_groups.setdefault('quadratus_lumborum_left', np.zeros_like(muscle_mask))
                        muscle_groups['quadratus_lumborum_left'] |= region_mask
                else:  # Obliques
                    muscle_groups.setdefault('obliques_left', np.zeros_like(muscle_mask))
                    muscle_groups['obliques_left'] |= region_mask

            else:  # Right side
                if abs(centroid_y - center_y) < h * 0.2:  # Lateral
                    if region.area > 500:  # Large muscle
                        muscle_groups.setdefault('psoas_right', np.zeros_like(muscle_mask))
                        muscle_groups['psoas_right'] |= region_mask
                    else:
                        muscle_groups.setdefault('quadratus_lumborum_right', np.zeros_like(muscle_mask))
                        muscle_groups['quadratus_lumborum_right'] |= region_mask
                else:  # Obliques
                    muscle_groups.setdefault('obliques_right', np.zeros_like(muscle_mask))
                    muscle_groups['obliques_right'] |= region_mask

        return muscle_groups

    def calculate_muscle_metrics(self, slice_2d: np.ndarray, muscle_mask: np.ndarray,
                                 muscle_groups: Optional[Dict[str, np.ndarray]] = None) -> Dict:
        """
        Calculate quantitative metrics for muscle tissue.

        Args:
            slice_2d: Original CT slice (normalized)
            muscle_mask: Binary mask of muscle tissue
            muscle_groups: Optional dictionary of individual muscle group masks

        Returns:
            Dictionary of muscle metrics
        """
        metrics = {}

        # Pixel spacing in mm
        pixel_area_mm2 = self.spacing[0] * self.spacing[1]
        pixel_area_cm2 = pixel_area_mm2 / 100  # Convert to cm²

        # Total muscle area
        total_muscle_pixels = np.sum(muscle_mask)
        metrics['total_muscle_area_cm2'] = total_muscle_pixels * pixel_area_cm2

        # Mean muscle attenuation (radiodensity)
        if total_muscle_pixels > 0:
            muscle_values = slice_2d[muscle_mask > 0]

            # Convert back to HU for clinical interpretation
            hu_values = muscle_values * 450 - 150  # Reverse normalization

            metrics['mean_muscle_hu'] = float(np.mean(hu_values))
            metrics['std_muscle_hu'] = float(np.std(hu_values))
            metrics['median_muscle_hu'] = float(np.median(hu_values))

            # Muscle radiation attenuation categories
            # Low attenuation: < 30 HU (fatty infiltration)
            low_atten_mask = hu_values < 30
            metrics['low_attenuation_area_cm2'] = np.sum(low_atten_mask) * pixel_area_cm2
            metrics['low_attenuation_percentage'] = (np.sum(low_atten_mask) / len(hu_values)) * 100
        else:
            metrics['mean_muscle_hu'] = 0
            metrics['std_muscle_hu'] = 0
            metrics['median_muscle_hu'] = 0
            metrics['low_attenuation_area_cm2'] = 0
            metrics['low_attenuation_percentage'] = 0

        # Individual muscle group areas
        if muscle_groups:
            for group_name, group_mask in muscle_groups.items():
                group_pixels = np.sum(group_mask)
                metrics[f'{group_name}_area_cm2'] = group_pixels * pixel_area_cm2

        # Shape metrics
        if total_muscle_pixels > 0:
            # Find contours for shape analysis
            contours, _ = cv2.findContours(muscle_mask.astype(np.uint8),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Combine all contours for overall muscle shape
                all_points = np.vstack(contours)

                # Fit ellipse to get major/minor axes
                if len(all_points) >= 5:
                    ellipse = cv2.fitEllipse(all_points)
                    metrics['muscle_major_axis'] = max(ellipse[1])
                    metrics['muscle_minor_axis'] = min(ellipse[1])
                    metrics['muscle_eccentricity'] = metrics['muscle_minor_axis'] / metrics['muscle_major_axis']

        return metrics

    def segment_l3_slice(self, l3_index: int) -> Tuple[np.ndarray, Dict]:
        """
        Complete segmentation pipeline for L3 slice.

        Args:
            l3_index: Z-index of L3 slice

        Returns:
            Tuple of (muscle_mask, metrics_dict)
        """
        # Extract L3 slice
        slice_2d = self.array[l3_index, :, :]

        # Segment body outline
        body_mask = self.segment_body_mask(slice_2d)

        # Segment muscle tissue
        muscle_mask = self.segment_muscle_threshold(slice_2d, body_mask)

        # Identify muscle groups
        muscle_groups = self.identify_muscle_groups(muscle_mask)

        # Calculate metrics
        metrics = self.calculate_muscle_metrics(slice_2d, muscle_mask, muscle_groups)

        # Add slice information
        metrics['l3_slice_index'] = l3_index
        metrics['volume_shape'] = self.array.shape

        return muscle_mask, metrics

    def visualize_segmentation(self, l3_index: int, muscle_mask: np.ndarray,
                               save_path: Optional[Path] = None) -> np.ndarray:
        """
        Create visualization of segmentation results.

        Args:
            l3_index: Z-index of L3 slice
            muscle_mask: Binary mask of muscle tissue
            save_path: Optional path to save visualization

        Returns:
            RGB visualization array
        """
        # Get original slice
        slice_2d = self.array[l3_index, :, :]

        # Convert to 8-bit for visualization
        slice_8bit = (slice_2d * 255).astype(np.uint8)

        # Create RGB image
        rgb_image = cv2.cvtColor(slice_8bit, cv2.COLOR_GRAY2RGB)

        # Overlay muscle segmentation in red
        muscle_overlay = np.zeros_like(rgb_image)
        muscle_overlay[:, :, 0] = muscle_mask * 255  # Red channel
        muscle_overlay[:, :, 1] = muscle_mask * 100  # Some green for visibility

        # Blend with original
        alpha = 0.4
        visualization = cv2.addWeighted(rgb_image, 1 - alpha, muscle_overlay, alpha, 0)

        # Add contours for clarity
        contours, _ = cv2.findContours(muscle_mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(visualization, contours, -1, (0, 255, 0), 2)

        if save_path:
            cv2.imwrite(str(save_path), visualization)

        return visualization


def process_patient(patient_id: str, l3_index: int,
                    base_dir: Path = Path(r"C:\CT_Project")) -> Dict:
    """
    Process a patient's CT scan for muscle segmentation at L3.

    Args:
        patient_id: Patient identifier
        l3_index: Pre-selected L3 slice index
        base_dir: Project base directory

    Returns:
        Dictionary with segmentation results and metrics
    """
    # Paths
    volume_path = base_dir / "data_preproc" / f"{patient_id}_iso_norm.nii.gz"
    output_dir = base_dir / "results" / patient_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize segmentor
    segmentor = MuscleSegmentor(volume_path)

    # Perform segmentation
    muscle_mask, metrics = segmentor.segment_l3_slice(l3_index)

    # Save results
    # 1. Save muscle mask as NIfTI
    mask_img = sitk.GetImageFromArray(muscle_mask[np.newaxis, :, :])
    mask_img.CopyInformation(segmentor.image)
    sitk.WriteImage(mask_img, str(output_dir / f"muscle_mask_l3.nii.gz"))

    # 2. Save visualization
    vis = segmentor.visualize_segmentation(l3_index, muscle_mask,
                                           save_path=output_dir / "segmentation_vis.png")

    # 3. Save metrics to JSON
    with open(output_dir / "muscle_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # 4. Save metrics to CSV for easy viewing
    import csv
    with open(output_dir / "muscle_metrics.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print(f"Processed {patient_id}:")
    print(f"  - Total muscle area: {metrics['total_muscle_area_cm2']:.2f} cm²")
    print(f"  - Mean muscle HU: {metrics['mean_muscle_hu']:.1f}")
    print(f"  - Low attenuation %: {metrics['low_attenuation_percentage']:.1f}%")

    return {
        'patient_id': patient_id,
        'l3_index': l3_index,
        'muscle_mask': muscle_mask,
        'metrics': metrics,
        'visualization': vis,
        'output_dir': output_dir
    }


if __name__ == "__main__":
    import sys

    # Example usage
    if len(sys.argv) > 1:
        patient_id = sys.argv[1]
    else:
        patient_id = "patient01"

    # Load L3 index from your CSV
    csv_path = Path(r"C:\CT_Project\meta\l3_slices.csv")
    l3_indices = {}

    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                l3_indices[row['patient_id']] = int(row['slice_index'])

    if patient_id in l3_indices:
        l3_idx = l3_indices[patient_id]
        results = process_patient(patient_id, l3_idx)
        print(f"\nResults saved to: {results['output_dir']}")
    else:
        print(f"No L3 index found for {patient_id}. Please run pick_l3.py first.")