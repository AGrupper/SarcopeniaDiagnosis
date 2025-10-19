# web_dashboard.py
"""
Simplified but complete web dashboard for sarcopenia analysis.
Run with: streamlit run web_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from PIL import Image
import io

# Page config
st.set_page_config(page_title="Sarcopenia Diagnosis", layout="wide", page_icon="🏥")

# Title
st.title("🏥 Sarcopenia Diagnosis System")
st.markdown("---")

# Initialize paths
BASE_DIR = Path(r"C:\CT_Project")

# Sidebar for patient selection
with st.sidebar:
    st.header("Patient Selection")

    # Get list of patients
    preproc_dir = BASE_DIR / "data_preproc"
    patients = []
    if preproc_dir.exists():
        for file in preproc_dir.glob("*_iso_norm.nii.gz"):
            patient_id = file.stem.replace("_iso_norm", "")
            patients.append(patient_id)

    if patients:
        selected_patient = st.selectbox("Select Patient", patients)

        st.header("Patient Information")
        sex = st.radio("Sex", ["M", "F"])
        height = st.number_input("Height (cm)", 150, 200, 175)
        weight = st.number_input("Weight (kg)", 40, 150, 70)
        age = st.number_input("Age", 18, 100, 65)

        patient_info = {
            'sex': sex,
            'height_m': height / 100,
            'weight_kg': weight,
            'age': age
        }
    else:
        st.error("No patients found!")
        st.stop()

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("📊 Analysis Pipeline")

    if st.button("🚀 Run Complete Analysis", type="primary"):
        with st.spinner("Processing..."):
            try:
                # Import modules
                import sys

                sys.path.append(str(BASE_DIR / "code" / "SarcopeniaDiagnosis"))

                # Run pipeline
                st.info("Step 1/4: Detecting L3 vertebra...")
                from automatic_l3_detection import automatic_l3_detection

                l3_idx, confidence = automatic_l3_detection(selected_patient, BASE_DIR)
                st.success(f"✓ L3 detected at slice {l3_idx} (confidence: {confidence:.1%})")

                st.info("Step 2/4: Segmenting muscles...")
                from muscle_segmentation import process_patient

                seg_results = process_patient(selected_patient, l3_idx, BASE_DIR)
                st.success(f"✓ Muscle area: {seg_results['metrics']['total_muscle_area_cm2']:.1f} cm²")

                st.info("Step 3/4: Diagnosing sarcopenia...")
                from sarcopenia_diagnosis import diagnose_patient

                diagnosis = diagnose_patient(selected_patient, patient_info, BASE_DIR)
                st.success(f"✓ Diagnosis: {diagnosis['status']}")

                st.info("Step 4/4: Generating report...")
                st.session_state['results'] = {
                    'l3_idx': l3_idx,
                    'confidence': confidence,
                    'segmentation': seg_results,
                    'diagnosis': diagnosis
                }
                st.success("✓ Analysis complete!")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure all required modules are in place")

with col2:
    st.header("📈 Results")

    if 'results' in st.session_state:
        results = st.session_state['results']

        # Display metrics
        st.subheader("Muscle Metrics")
        metrics = results['segmentation']['metrics']

        col1, col2, col3 = st.columns(3)
        col1.metric("Muscle Area", f"{metrics['total_muscle_area_cm2']:.1f} cm²")
        col2.metric("Mean HU", f"{metrics['mean_muscle_hu']:.1f}")
        col3.metric("Low Atten %", f"{metrics['low_attenuation_percentage']:.1f}%")

        # Display diagnosis
        st.subheader("Diagnosis")
        diagnosis = results['diagnosis']

        # Status indicator with color
        status_color = {
            'Normal': 'green',
            'Pre-sarcopenia': 'yellow',
            'Sarcopenia': 'orange',
            'Severe Sarcopenia': 'red'
        }
        color = status_color.get(diagnosis['status'], 'gray')
        st.markdown(f"**Status:** <span style='color:{color}; font-size:20px'>{diagnosis['status']}</span>",
                    unsafe_allow_html=True)
        st.markdown(f"**Risk Score:** {diagnosis['risk_score']:.0f}/100")
        st.markdown(f"**Explanation:** {diagnosis['explanation']}")

        # Recommendations
        st.subheader("Recommendations")
        for rec in diagnosis['recommendations']:
            st.write(f"• {rec}")

        # Display visualization if exists
        vis_path = BASE_DIR / "results" / selected_patient / "segmentation_vis.png"
        if vis_path.exists():
            st.subheader("Segmentation Visualization")
            img = Image.open(vis_path)
            st.image(img, caption="L3 Muscle Segmentation", use_column_width=True)
    else:
        st.info("👈 Click 'Run Complete Analysis' to start")

# Display existing results if available
st.markdown("---")
st.header("📁 Previous Results")

results_dir = BASE_DIR / "results"
if results_dir.exists():
    available_results = list(results_dir.glob("*/diagnosis.json"))
    if available_results:
        result_data = []
        for result_file in available_results:
            patient_id = result_file.parent.name
            with open(result_file, 'r') as f:
                diag = json.load(f)
                result_data.append({
                    'Patient': patient_id,
                    'Status': diag['status'],
                    'Risk Score': f"{diag['risk_score']:.0f}",
                    'SMI': f"{diag['smi']:.1f}",
                    'Timestamp': diag['timestamp'][:19]
                })

        df = pd.DataFrame(result_data)
        st.dataframe(df, use_container_width=True)

        # Download button for results
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name="sarcopenia_results.csv",
            mime="text/csv"
        )
    else:
        st.info("No previous results found")