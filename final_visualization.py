# final_visualization.py
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np


def create_comparison_figure():
    base_dir = Path(r"C:\CT_Project")

    # Load all metrics
    patients = []
    areas = []
    hus = []
    types = []

    for i in range(1, 8):
        patient_id = f"patient{i:02d}"
        metrics_file = base_dir / "results" / patient_id / "muscle_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                patients.append(patient_id)
                areas.append(metrics['total_muscle_area_cm2'])
                hus.append(metrics['mean_muscle_hu'])
                types.append('Donor' if i <= 2 else 'Non-Donor')

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Muscle area comparison
    colors = ['red' if t == 'Donor' else 'blue' for t in types]
    ax1.bar(patients, areas, color=colors, alpha=0.7)
    ax1.axhline(y=170, color='green', linestyle='--', label='Male threshold')
    ax1.axhline(y=130, color='orange', linestyle='--', label='Female threshold')
    ax1.set_ylabel('Muscle Area (cm²)', fontsize=12)
    ax1.set_title('L3 Muscle Area by Patient', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)

    # Muscle quality scatter
    scatter_colors = ['red' if t == 'Donor' else 'blue' for t in types]
    ax2.scatter(areas, hus, c=scatter_colors, s=100, alpha=0.7)

    # Add patient labels
    for i, patient in enumerate(patients):
        ax2.annotate(patient.replace('patient', 'P'),
                     (areas[i], hus[i]),
                     fontsize=8)

    # Add quadrant lines
    ax2.axvline(x=170, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=30, color='gray', linestyle='--', alpha=0.5)

    # Add quadrant labels
    ax2.text(100, 65, 'Low Muscle\nGood Quality', fontsize=9, alpha=0.5)
    ax2.text(250, 65, 'Normal Muscle\nGood Quality', fontsize=9, alpha=0.5)
    ax2.text(100, 20, 'Sarcopenia', fontsize=9, alpha=0.5)
    ax2.text(250, 20, 'Myosteatosis', fontsize=9, alpha=0.5)

    ax2.set_xlabel('Muscle Area (cm²)', fontsize=12)
    ax2.set_ylabel('Mean HU', fontsize=12)
    ax2.set_title('Muscle Quantity vs Quality', fontsize=14, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Kidney Donor'),
                       Patch(facecolor='blue', alpha=0.7, label='Non-Donor')]
    ax2.legend(handles=legend_elements)

    plt.suptitle('Sarcopenia Analysis: CT-Based Muscle Assessment at L3',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save
    plt.savefig(base_dir / "results" / "final_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    create_comparison_figure()