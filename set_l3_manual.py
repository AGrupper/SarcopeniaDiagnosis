# set_l3_manual.py
"""
Manually set L3 index for a patient when automatic detection fails.
Usage: python set_l3_manual.py patient01 179
"""

import csv
import sys
from pathlib import Path


def set_l3_index(patient_id: str, l3_index: int, base_dir: Path = Path(r"C:\CT_Project")):
    """Manually set L3 index for a patient."""

    csv_path = base_dir / "meta" / "l3_slices_manual.csv"
    csv_path.parent.mkdir(exist_ok=True)

    # Read existing data
    existing = {}
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row['patient_id']] = row['slice_index']

    # Update with new value
    existing[patient_id] = str(l3_index)

    # Write back
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['patient_id', 'slice_index'])
        writer.writeheader()
        for pid, idx in existing.items():
            writer.writerow({'patient_id': pid, 'slice_index': idx})

    print(f"Set L3 index for {patient_id} to {l3_index}")
    print(f"Saved to: {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python set_l3_manual.py <patient_id> <l3_index>")
        print("Example: python set_l3_manual.py patient01 179")
        sys.exit(1)

    patient_id = sys.argv[1]
    l3_index = int(sys.argv[2])

    set_l3_index(patient_id, l3_index)