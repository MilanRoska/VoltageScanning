# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:09:12 2024

@author: m.roska
"""
# %%

import numpy as np
import matplotlib.pyplot as plt

from VSfit import vs_fit, generate_calibration_curve, double_sigmoid, sigmodial


# %% 🎛️ Generate Example VS Data with Artificial Noise

# Simulated timing data (0–120 in 2 V steps)
timing_data = np.linspace(0, 120, int(120/2) + 1)

# Parameters for a synthetic double sigmoid curve
y_init = 1
fall_center = 0.05
fall_slope = 0.2
rise_center = 0.15
rise_slope = 0.22

# Generate noise-free signal
y_clean = double_sigmoid(timing_data, y_init, fall_center, fall_slope, rise_center, rise_slope)

# Add Gaussian noise
noise = np.random.normal(loc=0, scale=0.1, size=timing_data.shape)
y_noisy = np.clip(y_clean + noise, a_min=0, a_max=None)

# Plot both clean and noisy signal
plt.figure(figsize=(6, 4))
plt.plot(timing_data, y_clean, label="Clean Signal", linestyle="-")
plt.scatter(timing_data, y_noisy, label="Noisy Signal", alpha=0.8)
plt.xlabel("Voltage [V]")
plt.ylabel("Signal Intensity [a.u.]")
plt.legend()
plt.title("Synthetic VS Signal with Noise")
plt.tight_layout()
plt.show()


# %% 🧪 Run VS Fit to Estimate VS Result

signal_data = y_noisy

vs_result, vs_result_err, vs_result_r2, fit_params = vs_fit(
    signal_data, timing_data, convert_to="Volt",
    acquisition_freq=1,
    tunning_freq=None,
    step_size=1,
    max_voltage=120,
    plot_vs=True
)


# %% 📈 Generate Calibration Curve Using Known Standards

# Example calibration data (sensitivity vs dV50)
# artificial pairs of dV50 values and calibration sensitivities
dV50_values = [0, 10, 20, 30, 40, 50, 60, 70, 80]
sensitivity_values = [1, 1, 1, 4, 6, 8, 9, 9, 9]

# Fit sigmoid calibration curve
parameters, param_errors, cal_r2 = generate_calibration_curve(
    np.array(dV50_values),
    np.array(sensitivity_values),
    plot_cal_curve=True
)


# %% 🔁 Predict Sensitivity of Unknown Using Fitted Calibration

fit_y_max, fit_center, fit_slope = parameters
estimated_sensitivity = sigmodial(vs_result, fit_y_max, fit_center, fit_slope)

print(f"VS Result (U²@50%): {vs_result:.2f}")
print(f"Estimated Sensitivity: {estimated_sensitivity:.2f}")
print(f"Calibration R²: {cal_r2:.3f}")
