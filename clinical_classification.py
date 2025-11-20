# clinical_classification.py
import json
from pathlib import Path
import pandas as pd


def classify_sarcopenia(muscle_area, mean_hu, sex='M'):
    """Classify sarcopenia status based on metrics."""

    # Thresholds
    male_threshold = 170
    female_threshold = 130

    threshold = male_threshold if sex == 'M' else female_threshold

    if muscle_area < threshold and mean_hu < 30:
        return "Severe Sarcopenia"
    elif muscle_area < threshold:
        return "Sarcopenia"
    elif mean_hu < 30 or muscle_area < threshold * 1.2:
        return "Pre-sarcopenia"
    else:
        return "Normal"


def analyze_with_classification():
    base_dir = Path(r"C:\CT_Project")
    results_dir = base_dir / "results"

    data = []
    for patient_dir in sorted(results_dir.iterdir()):
        if patient_dir.is_dir():
            metrics_file = patient_dir / "muscle_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)

                patient_num = int(patient_dir.name.replace('patient', ''))
                patient_type = 'Kidney Donor' if patient_num <= 2 else 'Non-Donor'

                # Classify
                classification = classify_sarcopenia(
                    metrics['total_muscle_area_cm2'],
                    metrics['mean_muscle_hu']
                )

                data.append({
                    'Patient': patient_dir.name,
                    'Type': patient_type,
                    'Muscle Area': f"{metrics['total_muscle_area_cm2']:.1f}",
                    'Mean HU': f"{metrics['mean_muscle_hu']:.1f}",
                    'Fat %': f"{metrics['low_attenuation_percentage']:.1f}",
                    'Classification': classification
                })

    df = pd.DataFrame(data)

    print("\n" + "=" * 90)
    print("SARCOPENIA CLASSIFICATION RESULTS")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)

    # Count classifications
    print("\nDIAGNOSTIC SUMMARY:")
    print("-" * 40)
    for classification in df['Classification'].unique():
        count = len(df[df['Classification'] == classification])
        percentage = (count / len(df)) * 100
        print(f"{classification}: {count} patients ({percentage:.1f}%)")

    # Save
    df.to_csv(results_dir / "clinical_classifications.csv", index=False)
    print(f"\nSaved to: {results_dir / 'clinical_classifications.csv'}")

    return df


if __name__ == "__main__":
    df = analyze_with_classification()