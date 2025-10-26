# analyze_all_patients.py
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def analyze_all_patients():
    base_dir = Path(r"C:\CT_Project")
    results_dir = base_dir / "results"

    data = []
    for patient_dir in sorted(results_dir.iterdir()):
        if patient_dir.is_dir():
            metrics_file = patient_dir / "muscle_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)

                # Determine patient type
                patient_num = int(patient_dir.name.replace('patient', ''))
                patient_type = 'Kidney Donor' if patient_num <= 2 else 'Non-Donor'

                data.append({
                    'Patient': patient_dir.name,
                    'Type': patient_type,
                    'Muscle Area (cm²)': metrics['total_muscle_area_cm2'],
                    'Mean HU': metrics['mean_muscle_hu'],
                    'Low Atten %': metrics['low_attenuation_percentage'],
                })

    df = pd.DataFrame(data)

    # Print summary
    print("\n" + "=" * 80)
    print("COMPLETE PATIENT ANALYSIS")
    print("=" * 80)
    print(df.to_string(index=False))

    # Statistical comparison
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON")
    print("=" * 80)

    donors = df[df['Type'] == 'Kidney Donor']
    non_donors = df[df['Type'] == 'Non-Donor']

    if len(donors) > 0 and len(non_donors) > 0:
        print(f"\nKidney Donors (n={len(donors)}):")
        print(f"  Muscle Area: {donors['Muscle Area (cm²)'].mean():.1f} ± {donors['Muscle Area (cm²)'].std():.1f} cm²")
        print(f"  Mean HU: {donors['Mean HU'].mean():.1f} ± {donors['Mean HU'].std():.1f}")
        print(f"  Low Atten: {donors['Low Atten %'].mean():.1f} ± {donors['Low Atten %'].std():.1f}%")

        print(f"\nNon-Donors (n={len(non_donors)}):")
        print(
            f"  Muscle Area: {non_donors['Muscle Area (cm²)'].mean():.1f} ± {non_donors['Muscle Area (cm²)'].std():.1f} cm²")
        print(f"  Mean HU: {non_donors['Mean HU'].mean():.1f} ± {non_donors['Mean HU'].std():.1f}")
        print(f"  Low Atten: {non_donors['Low Atten %'].mean():.1f} ± {non_donors['Low Atten %'].std():.1f}%")

        # Perform t-test if you have scipy
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(donors['Muscle Area (cm²)'],
                                              non_donors['Muscle Area (cm²)'])
            print(f"\nT-test for muscle area difference:")
            print(f"  t-statistic: {t_stat:.2f}")
            print(f"  p-value: {p_value:.4f}")
            if p_value < 0.05:
                print("  Result: Statistically significant difference (p < 0.05)")
        except ImportError:
            pass

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Box plot of muscle area
    df.boxplot(column='Muscle Area (cm²)', by='Type', ax=axes[0, 0])
    axes[0, 0].set_title('Muscle Area by Patient Type')
    axes[0, 0].set_ylabel('Muscle Area (cm²)')

    # Box plot of Mean HU
    df.boxplot(column='Mean HU', by='Type', ax=axes[0, 1])
    axes[0, 1].set_title('Mean HU by Patient Type')
    axes[0, 1].set_ylabel('Mean HU')

    # Scatter plot
    axes[1, 0].scatter(df['Muscle Area (cm²)'], df['Mean HU'],
                       c=['red' if t == 'Kidney Donor' else 'blue' for t in df['Type']],
                       s=100, alpha=0.6)
    axes[1, 0].set_xlabel('Muscle Area (cm²)')
    axes[1, 0].set_ylabel('Mean HU')
    axes[1, 0].set_title('Muscle Area vs Quality')
    axes[1, 0].legend(['Donor', 'Non-Donor'])

    # Bar chart
    axes[1, 1].bar(df['Patient'], df['Muscle Area (cm²)'],
                   color=['red' if t == 'Kidney Donor' else 'blue' for t in df['Type']])
    axes[1, 1].set_xlabel('Patient')
    axes[1, 1].set_ylabel('Muscle Area (cm²)')
    axes[1, 1].set_title('Individual Patient Results')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.suptitle('Sarcopenia Analysis: Donors vs Non-Donors', fontsize=16)
    plt.tight_layout()

    # Save figure
    output_path = base_dir / "results" / "comparison_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

    # Save CSV
    csv_path = base_dir / "results" / "all_patients_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    return df


if __name__ == "__main__":
    df = analyze_all_patients()