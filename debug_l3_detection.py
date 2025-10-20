# debug_l3_detection.py
"""
Debug script to understand why L3 detection confidence is 0%.
Run this to see what's happening with your CT data.
"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path


def debug_ct_volume(patient_id: str, base_dir: Path = Path(r"C:\CT_Project")):
    """Debug CT volume to understand its characteristics."""

    volume_path = base_dir / "data_preproc" / f"{patient_id}_iso_norm.nii.gz"

    if not volume_path.exists():
        print(f"Volume not found: {volume_path}")
        return

    # Load volume
    image = sitk.ReadImage(str(volume_path))
    array = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()

    print(f"\n{'=' * 60}")
    print(f"DEBUG INFO FOR {patient_id}")
    print(f"{'=' * 60}")

    # Basic info
    print(f"\nVolume shape: {array.shape} (Z, Y, X)")
    print(f"Spacing: {spacing} mm")
    print(f"Total slices: {array.shape[0]}")

    # Value statistics
    print(f"\nValue statistics (normalized):")
    print(f"  Min: {array.min():.3f}")
    print(f"  Max: {array.max():.3f}")
    print(f"  Mean: {array.mean():.3f}")
    print(f"  Std: {array.std():.3f}")

    # Check middle slice
    mid_z = array.shape[0] // 2
    mid_slice = array[mid_z, :, :]

    print(f"\nMiddle slice (z={mid_z}) statistics:")
    print(f"  Min: {mid_slice.min():.3f}")
    print(f"  Max: {mid_slice.max():.3f}")
    print(f"  Mean: {mid_slice.mean():.3f}")

    # Check for muscle-like values
    # Muscle should be in range 0.26 to 0.67 after normalization from [-150, 300]
    muscle_min = (-29 + 150) / 450  # About 0.27
    muscle_max = (150 + 150) / 450  # About 0.67

    print(f"\nChecking for muscle-like intensities ({muscle_min:.2f} to {muscle_max:.2f}):")

    # Sample every 10th slice
    muscle_areas = []
    for z in range(0, array.shape[0], 10):
        slice_2d = array[z, :, :]
        muscle_mask = (slice_2d >= muscle_min) & (slice_2d <= muscle_max)
        muscle_area = np.sum(muscle_mask)
        muscle_areas.append(muscle_area)
        if z % 50 == 0:
            print(f"  Slice {z}: {muscle_area} muscle pixels")

    # Find slice with maximum muscle area
    muscle_areas_full = []
    for z in range(array.shape[0]):
        slice_2d = array[z, :, :]
        muscle_mask = (slice_2d >= muscle_min) & (slice_2d <= muscle_max)
        muscle_areas_full.append(np.sum(muscle_mask))

    max_muscle_z = np.argmax(muscle_areas_full)
    print(f"\nSlice with maximum muscle area: {max_muscle_z}")
    print(f"  Muscle pixels: {muscle_areas_full[max_muscle_z]}")

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Show different slices
    slices_to_show = [
        array.shape[0] // 4,  # Upper third
        array.shape[0] // 2,  # Middle
        max_muscle_z,  # Max muscle
    ]

    for idx, z in enumerate(slices_to_show):
        # Original slice
        axes[0, idx].imshow(array[z, :, :], cmap='gray', vmin=0, vmax=1)
        axes[0, idx].set_title(f'Slice {z}')
        axes[0, idx].axis('off')

        # Muscle mask
        slice_2d = array[z, :, :]
        muscle_mask = (slice_2d >= muscle_min) & (slice_2d <= muscle_max)
        axes[1, idx].imshow(muscle_mask, cmap='hot')
        axes[1, idx].set_title(f'Muscle mask (pixels: {np.sum(muscle_mask)})')
        axes[1, idx].axis('off')

    plt.suptitle(f'Debug Visualization for {patient_id}')
    plt.tight_layout()
    plt.show()

    # Plot muscle area profile
    plt.figure(figsize=(12, 6))
    plt.plot(muscle_areas_full)
    plt.axvline(x=max_muscle_z, color='r', linestyle='--', label=f'Max at {max_muscle_z}')
    plt.axvline(x=array.shape[0] // 2, color='g', linestyle='--', label=f'Middle at {array.shape[0] // 2}')
    plt.xlabel('Slice (Z)')
    plt.ylabel('Muscle Pixels')
    plt.title(f'Muscle Area Profile for {patient_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Return suggested L3
    print(f"\n{'=' * 60}")
    print(f"SUGGESTED L3 SLICE: {max_muscle_z}")
    print(f"(Based on maximum muscle area)")
    print(f"{'=' * 60}")

    return max_muscle_z


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        patient_id = sys.argv[1]
    else:
        patient_id = "patient01"

    suggested_l3 = debug_ct_volume(patient_id)

    # Also debug patient02 if running default
    if len(sys.argv) == 1:
        print("\n" + "=" * 60 + "\n")
        debug_ct_volume("patient02")