# sarcopenia_gui.py
"""
Simple GUI Application for Sarcopenia Analysis
This can be easily converted to EXE with PyInstaller
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import json
import tempfile
import shutil


class SarcopeniaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sarcopenia Analysis System")
        self.root.geometry("800x600")

        # Variables
        self.dicom_folder = None
        self.results = None

        self.setup_ui()

    def setup_ui(self):
        # Title
        title = tk.Label(self.root, text="Sarcopenia Analysis System",
                         font=("Arial", 24, "bold"))
        title.pack(pady=20)

        # Input Frame
        input_frame = ttk.LabelFrame(self.root, text="Patient Information", padding=20)
        input_frame.pack(padx=20, pady=10, fill="x")

        # Patient info
        info_grid = ttk.Frame(input_frame)
        info_grid.pack()

        ttk.Label(info_grid, text="Sex:").grid(row=0, column=0, padx=5, pady=5)
        self.sex_var = tk.StringVar(value="M")
        ttk.Radiobutton(info_grid, text="Male", variable=self.sex_var,
                        value="M").grid(row=0, column=1, padx=5)
        ttk.Radiobutton(info_grid, text="Female", variable=self.sex_var,
                        value="F").grid(row=0, column=2, padx=5)

        ttk.Label(info_grid, text="Height (cm):").grid(row=1, column=0, padx=5, pady=5)
        self.height_var = tk.IntVar(value=175)
        ttk.Entry(info_grid, textvariable=self.height_var, width=10).grid(row=1, column=1)

        ttk.Label(info_grid, text="Weight (kg):").grid(row=1, column=2, padx=5, pady=5)
        self.weight_var = tk.IntVar(value=70)
        ttk.Entry(info_grid, textvariable=self.weight_var, width=10).grid(row=1, column=3)

        # File selection
        file_frame = ttk.LabelFrame(self.root, text="CT Scan Selection", padding=20)
        file_frame.pack(padx=20, pady=10, fill="x")

        self.file_label = ttk.Label(file_frame, text="No folder selected")
        self.file_label.pack(pady=5)

        ttk.Button(file_frame, text="Select DICOM Folder",
                   command=self.select_folder).pack(pady=10)

        # Analyze button
        self.analyze_btn = ttk.Button(self.root, text="ANALYZE CT SCAN",
                                      command=self.analyze, state="disabled")
        self.analyze_btn.pack(pady=20)

        # Progress
        self.progress = ttk.Progressbar(self.root, length=400, mode='determinate')
        self.progress.pack(pady=10)

        self.status_label = ttk.Label(self.root, text="Ready")
        self.status_label.pack()

        # Results Frame
        results_frame = ttk.LabelFrame(self.root, text="Results", padding=20)
        results_frame.pack(padx=20, pady=10, fill="both", expand=True)

        self.results_text = tk.Text(results_frame, height=10, width=70)
        self.results_text.pack(fill="both", expand=True)

        # Export button
        self.export_btn = ttk.Button(self.root, text="Export Results",
                                     command=self.export_results, state="disabled")
        self.export_btn.pack(pady=10)

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select DICOM Folder")
        if folder:
            self.dicom_folder = Path(folder)
            self.file_label.config(text=f"Selected: {folder}")
            self.analyze_btn.config(state="normal")

    def update_progress(self, value, status):
        self.progress['value'] = value
        self.status_label.config(text=status)
        self.root.update()

    def analyze(self):
        if not self.dicom_folder:
            messagebox.showerror("Error", "Please select a DICOM folder")
            return

        try:
            self.update_progress(0, "Starting analysis...")

            # Create temp directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Step 1: Read DICOM
                self.update_progress(20, "Reading DICOM files...")
                reader = sitk.ImageSeriesReader()
                series_ids = reader.GetGDCMSeriesIDs(str(self.dicom_folder))

                if not series_ids:
                    raise ValueError("No DICOM series found")

                dicom_files = reader.GetGDCMSeriesFileNames(str(self.dicom_folder), series_ids[0])
                reader.SetFileNames(dicom_files)
                image = reader.Execute()

                # Step 2: Preprocess
                self.update_progress(40, "Preprocessing CT scan...")
                hu_min, hu_max = -150, 300
                clamped = sitk.Clamp(image, sitk.sitkFloat32, hu_min, hu_max)
                normalized = (clamped - hu_min) / float(hu_max - hu_min)

                # Save preprocessed
                volume_path = temp_path / "preprocessed.nii.gz"
                sitk.WriteImage(normalized, str(volume_path))

                # Step 3: Find L3
                self.update_progress(60, "Detecting L3 vertebra...")
                array = sitk.GetArrayFromImage(normalized)
                muscle_areas = []

                for z in range(array.shape[0]):
                    slice_2d = array[z, :, :]
                    muscle_mask = (slice_2d >= 0.269) & (slice_2d <= 0.667)
                    muscle_areas.append(np.sum(muscle_mask))

                l3_index = np.argmax(muscle_areas)

                # Step 4: Segment muscle (simplified)
                self.update_progress(80, "Segmenting muscle tissue...")
                l3_slice = array[l3_index, :, :]

                # Simple muscle segmentation
                muscle_mask = (l3_slice >= 0.269) & (l3_slice <= 0.667)

                # Calculate metrics
                spacing = image.GetSpacing()
                pixel_area_cm2 = (spacing[0] * spacing[1]) / 100
                muscle_area = np.sum(muscle_mask) * pixel_area_cm2

                muscle_values = l3_slice[muscle_mask]
                hu_values = muscle_values * 450 - 150
                mean_hu = np.mean(hu_values) if len(hu_values) > 0 else 0

                # Step 5: Classify
                self.update_progress(90, "Analyzing results...")
                sex = self.sex_var.get()
                threshold = 170 if sex == 'M' else 130

                if muscle_area < threshold:
                    diagnosis = "Sarcopenia Detected"
                elif muscle_area < threshold * 1.2:
                    diagnosis = "Pre-sarcopenia"
                else:
                    diagnosis = "Normal"

                # Calculate SMI
                height_m = self.height_var.get() / 100
                smi = muscle_area / (height_m ** 2)

                # Store results
                self.results = {
                    'muscle_area': muscle_area,
                    'mean_hu': mean_hu,
                    'l3_index': l3_index,
                    'diagnosis': diagnosis,
                    'smi': smi,
                    'threshold': threshold
                }

                # Display results
                self.display_results()

                self.update_progress(100, "Analysis complete!")
                self.export_btn.config(state="normal")

        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.update_progress(0, "Error occurred")

    def display_results(self):
        if not self.results:
            return

        self.results_text.delete(1.0, tk.END)

        report = f"""
SARCOPENIA ANALYSIS REPORT
{'=' * 50}

RESULTS:
  • Muscle Area: {self.results['muscle_area']:.1f} cm²
  • Mean HU: {self.results['mean_hu']:.1f}
  • SMI: {self.results['smi']:.1f} cm²/m²
  • L3 Slice: #{self.results['l3_index']}

DIAGNOSIS: {self.results['diagnosis']}

REFERENCE:
  • Normal threshold: {self.results['threshold']} cm²
  • Patient muscle area: {self.results['muscle_area']:.1f} cm²
  • Status: {'Below' if self.results['muscle_area'] < self.results['threshold'] else 'Above'} threshold

{'=' * 50}
        """

        self.results_text.insert(1.0, report)

    def export_results(self):
        if not self.results:
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            with open(file_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            messagebox.showinfo("Success", f"Results saved to {file_path}")


def main():
    root = tk.Tk()
    app = SarcopeniaApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()