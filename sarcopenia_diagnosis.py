# sarcopenia_diagnosis.py
"""
Sarcopenia diagnosis based on muscle metrics from CT imaging.
Implements various diagnostic criteria and thresholds.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json
from enum import Enum
from dataclasses import dataclass


class SarcopeniaStatus(Enum):
    """Sarcopenia diagnosis categories"""
    NORMAL = "Normal"
    PRE_SARCOPENIA = "Pre-sarcopenia"
    SARCOPENIA = "Sarcopenia"
    SEVERE_SARCOPENIA = "Severe Sarcopenia"


@dataclass
class DiagnosticCriteria:
    """
    Diagnostic thresholds for sarcopenia.
    Based on various published criteria (EWGSOP2, AWGS, etc.)
    """
    # Skeletal Muscle Index (SMI) thresholds in cm²/m²
    # These need to be adjusted based on patient height
    male_smi_threshold: float = 52.4  # cm²/m² for males
    female_smi_threshold: float = 38.5  # cm²/m² for females

    # Muscle attenuation thresholds in HU
    low_attenuation_threshold: float = 30.0  # HU
    very_low_attenuation_threshold: float = 20.0  # HU

    # Low attenuation muscle percentage thresholds
    low_atten_percent_threshold: float = 30.0  # %

    # Alternative criteria (can be population-specific)
    # Martin et al. criteria for oncology patients
    martin_male_smi: float = 43.0  # cm²/m² (BMI < 25)
    martin_male_smi_obese: float = 53.0  # cm²/m² (BMI ≥ 25)
    martin_female_smi: float = 41.0  # cm²/m²


class SarcopeniaDiagnostic:
    """
    Performs sarcopenia diagnosis based on CT-derived muscle metrics.
    """

    def __init__(self, criteria: Optional[DiagnosticCriteria] = None):
        """
        Initialize diagnostic system with criteria.

        Args:
            criteria: Diagnostic thresholds to use (defaults to standard values)
        """
        self.criteria = criteria or DiagnosticCriteria()

    def calculate_smi(self, muscle_area_cm2: float, height_m: float) -> float:
        """
        Calculate Skeletal Muscle Index (SMI).

        SMI = muscle area (cm²) / height² (m²)

        Args:
            muscle_area_cm2: Total muscle area at L3 in cm²
            height_m: Patient height in meters

        Returns:
            SMI in cm²/m²
        """
        return muscle_area_cm2 / (height_m ** 2)

    def evaluate_muscle_quantity(self, smi: float, sex: str,
                                 bmi: Optional[float] = None) -> Tuple[bool, str]:
        """
        Evaluate if muscle quantity indicates sarcopenia.

        Args:
            smi: Skeletal Muscle Index in cm²/m²
            sex: 'M' or 'F'
            bmi: Optional BMI for refined criteria

        Returns:
            Tuple of (is_low, description)
        """
        sex = sex.upper()

        # Determine threshold based on sex and BMI
        if sex == 'M':
            if bmi and bmi >= 25:
                threshold = self.criteria.martin_male_smi_obese
                criteria_name = "Martin et al. (obese)"
            elif bmi and bmi < 25:
                threshold = self.criteria.martin_male_smi
                criteria_name = "Martin et al."
            else:
                threshold = self.criteria.male_smi_threshold
                criteria_name = "EWGSOP2"
        else:  # Female
            if bmi is not None:
                threshold = self.criteria.martin_female_smi
                criteria_name = "Martin et al."
            else:
                threshold = self.criteria.female_smi_threshold
                criteria_name = "EWGSOP2"

        is_low = smi < threshold

        if is_low:
            severity = (threshold - smi) / threshold * 100
            if severity > 30:
                description = f"Severely reduced SMI ({smi:.1f} cm²/m², {severity:.1f}% below {criteria_name} threshold of {threshold:.1f})"
            elif severity > 15:
                description = f"Moderately reduced SMI ({smi:.1f} cm²/m², {severity:.1f}% below {criteria_name} threshold of {threshold:.1f})"
            else:
                description = f"Mildly reduced SMI ({smi:.1f} cm²/m², {severity:.1f}% below {criteria_name} threshold of {threshold:.1f})"
        else:
            description = f"Normal SMI ({smi:.1f} cm²/m², above {criteria_name} threshold of {threshold:.1f})"

        return is_low, description

    def evaluate_muscle_quality(self, mean_hu: float,
                                low_atten_percent: float) -> Tuple[bool, str]:
        """
        Evaluate muscle quality based on attenuation.

        Args:
            mean_hu: Mean muscle attenuation in HU
            low_atten_percent: Percentage of low attenuation muscle

        Returns:
            Tuple of (is_poor_quality, description)
        """
        issues = []

        if mean_hu < self.criteria.very_low_attenuation_threshold:
            issues.append(f"very low mean attenuation ({mean_hu:.1f} HU)")
        elif mean_hu < self.criteria.low_attenuation_threshold:
            issues.append(f"low mean attenuation ({mean_hu:.1f} HU)")

        if low_atten_percent > self.criteria.low_atten_percent_threshold:
            issues.append(f"high proportion of fatty infiltration ({low_atten_percent:.1f}%)")

        is_poor_quality = len(issues) > 0

        if issues:
            description = f"Poor muscle quality: {', '.join(issues)}"
        else:
            description = f"Good muscle quality (mean {mean_hu:.1f} HU, {low_atten_percent:.1f}% low attenuation)"

        return is_poor_quality, description

    def diagnose(self, metrics: Dict, patient_info: Dict) -> Dict:
        """
        Perform comprehensive sarcopenia diagnosis.

        Args:
            metrics: Muscle metrics from segmentation
            patient_info: Dictionary with 'height_m', 'sex', optional 'weight_kg', 'age'

        Returns:
            Diagnosis dictionary with status, scores, and explanations
        """
        diagnosis = {
            'timestamp': str(np.datetime64('now')),
            'patient_info': patient_info.copy(),
            'metrics': metrics.copy()
        }

        # Calculate SMI
        height_m = patient_info['height_m']
        smi = self.calculate_smi(metrics['total_muscle_area_cm2'], height_m)
        diagnosis['smi'] = smi

        # Calculate BMI if weight available
        bmi = None
        if 'weight_kg' in patient_info:
            bmi = patient_info['weight_kg'] / (height_m ** 2)
            diagnosis['bmi'] = bmi

        # Evaluate muscle quantity
        quantity_low, quantity_desc = self.evaluate_muscle_quantity(
            smi, patient_info['sex'], bmi
        )
        diagnosis['muscle_quantity_assessment'] = {
            'is_low': quantity_low,
            'description': quantity_desc
        }

        # Evaluate muscle quality
        quality_poor, quality_desc = self.evaluate_muscle_quality(
            metrics['mean_muscle_hu'],
            metrics['low_attenuation_percentage']
        )
        diagnosis['muscle_quality_assessment'] = {
            'is_poor': quality_poor,
            'description': quality_desc
        }

        # Determine overall sarcopenia status
        if quantity_low and quality_poor:
            status = SarcopeniaStatus.SEVERE_SARCOPENIA
            explanation = "Both muscle quantity and quality are significantly reduced"
        elif quantity_low:
            status = SarcopeniaStatus.SARCOPENIA
            explanation = "Muscle quantity is below threshold"
        elif quality_poor:
            status = SarcopeniaStatus.PRE_SARCOPENIA
            explanation = "Muscle quality is compromised (myosteatosis)"
        else:
            status = SarcopeniaStatus.NORMAL
            explanation = "Muscle parameters are within normal range"

        diagnosis['status'] = status.value
        diagnosis['explanation'] = explanation

        # Calculate risk score (0-100)
        risk_score = self.calculate_risk_score(smi, patient_info['sex'],
                                               metrics['mean_muscle_hu'],
                                               metrics['low_attenuation_percentage'])
        diagnosis['risk_score'] = risk_score

        # Add recommendations
        diagnosis['recommendations'] = self.generate_recommendations(status, risk_score)

        return diagnosis

    def calculate_risk_score(self, smi: float, sex: str,
                             mean_hu: float, low_atten_percent: float) -> float:
        """
        Calculate a composite risk score (0-100).

        Args:
            smi: Skeletal Muscle Index
            sex: Patient sex
            mean_hu: Mean muscle attenuation
            low_atten_percent: Percentage of low attenuation muscle

        Returns:
            Risk score from 0 (lowest risk) to 100 (highest risk)
        """
        score = 0.0

        # SMI component (0-50 points)
        threshold = self.criteria.male_smi_threshold if sex.upper() == 'M' else self.criteria.female_smi_threshold
        if smi < threshold:
            smi_deficit = (threshold - smi) / threshold
            score += min(50, smi_deficit * 100)

        # Muscle quality component (0-50 points)
        # Mean HU component (0-25 points)
        if mean_hu < self.criteria.low_attenuation_threshold:
            hu_deficit = (self.criteria.low_attenuation_threshold - mean_hu) / self.criteria.low_attenuation_threshold
            score += min(25, hu_deficit * 50)

        # Low attenuation percentage component (0-25 points)
        if low_atten_percent > self.criteria.low_atten_percent_threshold:
            excess = (low_atten_percent - self.criteria.low_atten_percent_threshold) / 100
            score += min(25, excess * 50)

        return min(100, score)

    def generate_recommendations(self, status: SarcopeniaStatus,
                                 risk_score: float) -> List[str]:
        """
        Generate clinical recommendations based on diagnosis.

        Args:
            status: Sarcopenia status
            risk_score: Calculated risk score

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if status == SarcopeniaStatus.NORMAL:
            recommendations.append("Continue regular physical activity and balanced nutrition")
            if risk_score > 20:
                recommendations.append("Monitor muscle health annually")

        elif status == SarcopeniaStatus.PRE_SARCOPENIA:
            recommendations.append("Consider nutritional assessment and protein supplementation")
            recommendations.append("Initiate resistance training program")
            recommendations.append("Follow-up imaging in 6-12 months")

        elif status == SarcopeniaStatus.SARCOPENIA:
            recommendations.append("Refer to nutrition specialist for dietary intervention")
            recommendations.append("Prescribe structured resistance and balance training")
            recommendations.append("Consider protein supplementation (1.2-1.5 g/kg/day)")
            recommendations.append("Screen for underlying conditions")
            recommendations.append("Follow-up imaging in 3-6 months")

        elif status == SarcopeniaStatus.SEVERE_SARCOPENIA:
            recommendations.append("Urgent multidisciplinary intervention required")
            recommendations.append("Comprehensive geriatric assessment")
            recommendations.append("Intensive nutritional support with protein supplementation")
            recommendations.append("Supervised exercise program with physiotherapy")
            recommendations.append("Evaluate for hormonal deficiencies")
            recommendations.append("Close monitoring with follow-up in 3 months")

        return recommendations

    def generate_report(self, diagnosis: Dict, output_path: Optional[Path] = None) -> str:
        """
        Generate a clinical report from diagnosis.

        Args:
            diagnosis: Diagnosis dictionary
            output_path: Optional path to save report

        Returns:
            Report text
        """
        report = []
        report.append("=" * 60)
        report.append("SARCOPENIA ASSESSMENT REPORT")
        report.append("=" * 60)
        report.append("")

        # Patient information
        report.append("PATIENT INFORMATION")
        report.append("-" * 20)
        info = diagnosis['patient_info']
        report.append(f"Sex: {info.get('sex', 'N/A')}")
        report.append(f"Height: {info.get('height_m', 'N/A')} m")
        if 'weight_kg' in info:
            report.append(f"Weight: {info['weight_kg']} kg")
            report.append(f"BMI: {diagnosis.get('bmi', 'N/A'):.1f} kg/m²")
        if 'age' in info:
            report.append(f"Age: {info['age']} years")
        report.append("")

        # Muscle measurements
        report.append("MUSCLE MEASUREMENTS AT L3")
        report.append("-" * 20)
        metrics = diagnosis['metrics']
        report.append(f"Total Muscle Area: {metrics['total_muscle_area_cm2']:.2f} cm²")
        report.append(f"Skeletal Muscle Index (SMI): {diagnosis['smi']:.2f} cm²/m²")
        report.append(f"Mean Muscle Attenuation: {metrics['mean_muscle_hu']:.1f} HU")
        report.append(f"Low Attenuation Muscle: {metrics['low_attenuation_percentage']:.1f}%")
        report.append("")

        # Diagnosis
        report.append("DIAGNOSIS")
        report.append("-" * 20)
        report.append(f"Status: {diagnosis['status']}")
        report.append(f"Risk Score: {diagnosis['risk_score']:.0f}/100")
        report.append(f"Explanation: {diagnosis['explanation']}")
        report.append("")

        # Assessments
        report.append("DETAILED ASSESSMENT")
        report.append("-" * 20)
        report.append(f"Muscle Quantity: {diagnosis['muscle_quantity_assessment']['description']}")
        report.append(f"Muscle Quality: {diagnosis['muscle_quality_assessment']['description']}")
        report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 20)
        for i, rec in enumerate(diagnosis['recommendations'], 1):
            report.append(f"{i}. {rec}")
        report.append("")

        report.append("-" * 60)
        report.append(f"Report generated: {diagnosis['timestamp']}")
        report.append("Note: This assessment is for clinical support only.")
        report.append("Final diagnosis should incorporate clinical evaluation.")

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)

        return report_text


def diagnose_patient(patient_id: str, patient_info: Dict,
                     base_dir: Path = Path(r"C:\CT_Project")) -> Dict:
    """
    Complete diagnosis pipeline for a patient.

    Args:
        patient_id: Patient identifier
        patient_info: Dict with 'height_m', 'sex', optional 'weight_kg', 'age'
        base_dir: Project base directory

    Returns:
        Complete diagnosis results
    """
    # Load muscle metrics
    metrics_path = base_dir / "results" / patient_id / "muscle_metrics.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics found for {patient_id}. Run segmentation first.")

    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    # Perform diagnosis
    diagnostic = SarcopeniaDiagnostic()
    diagnosis = diagnostic.diagnose(metrics, patient_info)

    # Save diagnosis results
    output_dir = base_dir / "results" / patient_id

    # Save JSON
    with open(output_dir / "diagnosis.json", 'w', encoding='utf-8') as f:
        json.dump(diagnosis, f, indent=2)

    # Generate and save report
    report = diagnostic.generate_report(diagnosis, output_dir / "diagnosis_report.txt")

    print(f"\n{report}")

    return diagnosis


if __name__ == "__main__":
    # Example usage
    patient_id = "patient01"

    # Example patient information (you'll need to get this from your data)
    patient_info = {
        'height_m': 1.75,  # 175 cm
        'weight_kg': 70,
        'sex': 'M',
        'age': 65
    }

    try:
        diagnosis = diagnose_patient(patient_id, patient_info)
        print(f"\nDiagnosis complete for {patient_id}")
        print(f"Status: {diagnosis['status']}")
        print(f"Risk Score: {diagnosis['risk_score']:.0f}/100")
    except Exception as e:
        print(f"Error: {e}")