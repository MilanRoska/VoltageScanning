# VotlageScanning: Voltage Scanning Analysis for CIMS Quantification

VotlageScanning is a Python package for analyzing Voltage Scanning (VS) datasets from Chemical Ionization Mass Spectrometry (CIMS) measurements.  
It allows extraction of meaningful **VS results**, construction of **calibration curves**, and **estimation of compound sensitivities**, including for **uncalibrated compounds**.

> 💡 VotlageScanning provides a structured workflow to derive sensitivities based on collision induced dissociation signal decay — enabling quantification even when no direct calibration exists.

---

## 📦 Included Functions

### `vs_fit(signal_data, timing_data, ...)`
- 🔧 Core analysis function.
- Converts a time-domain scan into a voltage-domain dataset.
- Fits the voltage scan using a double sigmoidal model. 
  - Alterative Gaussian function or a "closest_datapoint" approach with data smoothing are provided.
- Computes the **VS result**: the voltage equivalent where the signal drops to 50% of its initial intensity.
- Returns:
  - Fitted VS result (e.g., dV50)
  - Uncertainty estimate
  - Fit quality metrics (e.g., R²)
  - Full fit parameters

---

### `generate_calibration_curve(vs_results, sensitivities)`
- 🧪 Builds a calibration curve from a set of known **VS results** (analyzed with vs_fit) and their corresponding **conventionally determined sensitivities**.
- Fits a sigmoidal model relating the two.
- Returns:
  - Parameters of the sigmoidal fit (for later use with unknowns)
  - Optionally, a plot of the correlation and fit

---

### `sigmodial(x, ymax, x0, slope)`
- 📈 Sigmoid function used for calibration modeling.
- Can be used to convert new VS results (from uncalibrated compounds) into estimated sensitivities, using the calibration parameters from `generate_calibration_curve`.

---

## 🧪 Example Workflow

1. **Perform a refernce VS** on a set of known calibrant compounds, for which sensitivities were already derived conventionaly, to get a time series signal.
2. Use `vs_fit()` to extract the VS result for these calibration compounds (e.g., dV50).
4. Use `generate_calibration_curve()` to fit a model between these calibration VS results and their sensitivities.
1. **Perform a VS** on an unknown compound to get a time series signal.
2. Use `vs_fit()` to extract the VS result (e.g., dV50).
5. Use the `sigmodial()` function with calibration parameters to estimate the sensitivity of the unknown.

- One example workfliw is provided in testVSfit
- several example cases of potential artifacs in the VS analysis are provided
    - ExampleCompetingIsomers illustrates what happens if several isomers are measured at the same time
    - ExampleSlopeDuringVS illustrates how a rising or falling signal during the VS affects the VS results
    - ExampleTimingOffsetDuringVS illustrates how missalignment of the timing and signal data affect the VS results
---
