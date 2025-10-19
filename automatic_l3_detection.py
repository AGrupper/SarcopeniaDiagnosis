# automatic_l3_detection.py
"""
Automatic L3 vertebra detection in CT scans using image processing
and pattern recognition techniques.
"""

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.signal import find_peaks
import cv2
from pathlib import Path
from typing import Tuple, Optional


class L3Detector:
    """
    Detects L3 vertebra slice in CT scans using multiple approaches:
    1. Spine detection and vertebra counting
    2. Body composition analysis
    3. Anatomical landmark detection
    """

    def __init__(self, volume_path: Path):
        """
        Initialize detector with CT volume.

        Args:
            volume_path: Path to preprocessed NIfTI file
        """
        self.volume_path = volume_path
        self.image = sitk.ReadImage(str(volume_path))
        self.array = sitk.GetArrayFromImage(self.image)
        self.spacing = self.image.GetSpacing()

    def detect_spine_centerline(self) -> np.ndarray:
        """
        Detect spine centerline through the volume.

        Returns:
            Array of (x, y) coordinates for each z-slice
        """
        centerline = []

        for z in range(self.array.shape[0]):
            slice_2d = self.array[z, :, :]

            # Threshold for bone (high HU values in normalized data)
            bone_mask = slice_2d > 0.7  # Adjust based on your normalization

            # Find connected components
            labeled, num_features = ndimage.label(bone_mask)

            if num_features > 0:
                # Find the component closest to center (likely spine)
                center_y, center_x = slice_2d.shape[0] // 2, slice_2d.shape[1] // 2
                min_dist = float('inf')
                best_component = None

                for i in range(1, num_features + 1):
                    component_mask = labeled == i
                    y_coords, x_coords = np.where(component_mask)

                    if len(y_coords) > 0:
                        centroid_y = np.mean(y_coords)
                        centroid_x = np.mean(x_coords)
                        dist = np.sqrt((centroid_y - center_y) ** 2 + (centroid_x - center_x) ** 2)

                        if dist < min_dist:
                            min_dist = dist
                            best_component = (centroid_x, centroid_y)

                if best_component:
                    centerline.append(best_component)
                else:
                    centerline.append((center_x, center_y))
            else:
                centerline.append((slice_2d.shape[1] // 2, slice_2d.shape[0] // 2))

        return np.array(centerline)

    def detect_vertebrae_boundaries(self, centerline: np.ndarray) -> list:
        """
        Detect intervertebral disc spaces along the spine.

        Args:
            centerline: Spine centerline coordinates

        Returns:
            List of z-indices where vertebrae boundaries are detected
        """
        # Extract intensity profile along spine
        intensity_profile = []

        for z in range(self.array.shape[0]):
            x, y = int(centerline[z][0]), int(centerline[z][1])

            # Get average intensity in a small region around spine center
            y_min = max(0, y - 5)
            y_max = min(self.array.shape[1], y + 6)
            x_min = max(0, x - 5)
            x_max = min(self.array.shape[2], x + 6)

            roi = self.array[z, y_min:y_max, x_min:x_max]
            intensity_profile.append(np.mean(roi))

        intensity_profile = np.array(intensity_profile)

        # Smooth the profile
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(intensity_profile, sigma=2)

        # Find valleys (intervertebral discs have lower intensity)
        inverted = -smoothed
        peaks, properties = find_peaks(inverted, distance=15, prominence=0.05)

        return peaks.tolist()

    def detect_iliac_crest(self) -> Optional[int]:
        """
        Detect the iliac crest level as anatomical landmark.
        L3 is typically at or slightly above the iliac crest.

        Returns:
            Z-index of iliac crest or None if not found
        """
        # Look for characteristic pelvic bone shape
        max_area = 0
        iliac_z = None

        # Search in lower half of volume
        start_z = self.array.shape[0] // 2

        for z in range(start_z, self.array.shape[0]):
            slice_2d = self.array[z, :, :]

            # Threshold for bone
            bone_mask = slice_2d > 0.7

            # Morphological operations to connect bones
            kernel = np.ones((5, 5), np.uint8)
            closed = cv2.morphologyEx(bone_mask.astype(np.uint8),
                                      cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Look for large, bilateral bone structures
                total_area = sum(cv2.contourArea(c) for c in contours)

                if total_area > max_area:
                    max_area = total_area
                    iliac_z = z

        return iliac_z

    def estimate_l3_from_body_composition(self) -> int:
        """
        Estimate L3 position based on body composition changes.
        L3 level typically shows maximum muscle area.

        Returns:
            Estimated z-index for L3
        """
        muscle_areas = []

        # Define muscle HU range in normalized values
        muscle_min = 0.2  # Corresponds to ~-30 HU after normalization
        muscle_max = 0.6  # Corresponds to ~150 HU after normalization

        for z in range(self.array.shape[0]):
            slice_2d = self.array[z, :, :]

            # Segment muscle tissue
            muscle_mask = (slice_2d >= muscle_min) & (slice_2d <= muscle_max)

            # Calculate area
            area = np.sum(muscle_mask)
            muscle_areas.append(area)

        # Smooth the curve
        from scipy.ndimage import gaussian_filter1d
        smoothed_areas = gaussian_filter1d(muscle_areas, sigma=5)

        # Find the peak in the middle third of the scan
        start = len(smoothed_areas) // 3
        end = 2 * len(smoothed_areas) // 3

        mid_section = smoothed_areas[start:end]
        if len(mid_section) > 0:
            max_idx = np.argmax(mid_section)
            return start + max_idx

        return len(smoothed_areas) // 2

    def detect_l3_multi_method(self, confidence_threshold: float = 0.7) -> Tuple[int, float]:
        """
        Detect L3 using multiple methods and combine results.

        Args:
            confidence_threshold: Minimum confidence for detection

        Returns:
            Tuple of (l3_index, confidence_score)
        """
        results = []
        weights = []

        # Method 1: Body composition analysis (most reliable for L3)
        l3_body = self.estimate_l3_from_body_composition()
        results.append(l3_body)
        weights.append(0.4)

        # Method 2: Spine and vertebrae detection
        try:
            centerline = self.detect_spine_centerline()
            boundaries = self.detect_vertebrae_boundaries(centerline)

            if boundaries:
                # L3 is typically in the middle third
                target_idx = len(boundaries) // 2
                if target_idx < len(boundaries):
                    l3_spine = boundaries[target_idx]
                    results.append(l3_spine)
                    weights.append(0.3)
        except Exception as e:
            print(f"Spine detection failed: {e}")

        # Method 3: Iliac crest landmark
        iliac_z = self.detect_iliac_crest()
        if iliac_z:
            # L3 is typically 20-40mm above iliac crest
            mm_above = 30  # Average distance
            slices_above = int(mm_above / self.spacing[2])
            l3_iliac = max(0, iliac_z - slices_above)
            results.append(l3_iliac)
            weights.append(0.3)

        if not results:
            # Fallback to middle of scan
            return self.array.shape[0] // 2, 0.0

        # Weighted average of methods
        weights = np.array(weights[:len(results)])
        weights = weights / weights.sum()

        weighted_l3 = int(np.average(results, weights=weights))

        # Calculate confidence based on agreement between methods
        if len(results) > 1:
            std_dev = np.std(results)
            max_std = self.array.shape[0] * 0.1  # 10% of volume height
            confidence = max(0, 1 - (std_dev / max_std))
        else:
            confidence = weights[0]

        return weighted_l3, confidence

    def validate_l3_selection(self, z_index: int) -> bool:
        """
        Validate that the selected slice is anatomically consistent with L3.

        Args:
            z_index: Proposed L3 slice index

        Returns:
            True if selection appears valid
        """
        slice_2d = self.array[z_index, :, :]

        # Check for expected anatomical structures
        # 1. Vertebral body should be present
        center_y, center_x = slice_2d.shape[0] // 2, slice_2d.shape[1] // 2
        roi_size = 30
        vertebra_roi = slice_2d[center_y - roi_size:center_y + roi_size,
        center_x - roi_size:center_x + roi_size]

        bone_pixels = np.sum(vertebra_roi > 0.7)
        expected_bone = roi_size * roi_size * 0.1  # At least 10% bone

        if bone_pixels < expected_bone:
            return False

        # 2. Should have substantial muscle area
        muscle_mask = (slice_2d >= 0.2) & (slice_2d <= 0.6)
        muscle_area = np.sum(muscle_mask)
        min_muscle_area = slice_2d.size * 0.05  # At least 5% muscle

        if muscle_area < min_muscle_area:
            return False

        return True


def automatic_l3_detection(patient_id: str, data_dir: Path = Path(r"C:\CT_Project")) -> Tuple[int, float]:
    """
    Main function to automatically detect L3 slice for a patient.

    Args:
        patient_id: Patient identifier
        data_dir: Base directory for CT project

    Returns:
        Tuple of (l3_slice_index, confidence_score)
    """
    # Path to preprocessed volume
    volume_path = data_dir / "data_preproc" / f"{patient_id}_iso_norm.nii.gz"

    if not volume_path.exists():
        raise FileNotFoundError(f"Preprocessed volume not found: {volume_path}")

    # Initialize detector
    detector = L3Detector(volume_path)

    # Detect L3
    l3_index, confidence = detector.detect_l3_multi_method()

    # Validate result
    if not detector.validate_l3_selection(l3_index):
        print(f"Warning: L3 detection for {patient_id} failed validation")
        confidence *= 0.5

    print(f"Detected L3 for {patient_id}: slice {l3_index} (confidence: {confidence:.2f})")

    # Optionally save to CSV
    import csv
    csv_path = data_dir / "meta" / "l3_slices_auto.csv"
    csv_path.parent.mkdir(exist_ok=True)

    # Read existing data
    existing = {}
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row['patient_id']] = {
                    'slice_index': row['slice_index'],
                    'confidence': row.get('confidence', '1.0')
                }

    # Update with new detection
    existing[patient_id] = {
        'slice_index': str(l3_index),
        'confidence': f"{confidence:.3f}"
    }

    # Write back
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['patient_id', 'slice_index', 'confidence'])
        writer.writeheader()
        for pid, data in existing.items():
            writer.writerow({
                'patient_id': pid,
                'slice_index': data['slice_index'],
                'confidence': data['confidence']
            })

    return l3_index, confidence


if __name__ == "__main__":
    # Test with your patients
    for patient_id in ["patient01", "patient02"]:
        try:
            l3_idx, conf = automatic_l3_detection(patient_id)
            print(f"{patient_id}: L3 at slice {l3_idx} (confidence: {conf:.2%})")
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")