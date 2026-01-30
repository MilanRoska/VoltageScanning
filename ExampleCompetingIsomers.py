# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:45:04 2024

@author: m.roska

Visualization of Isomer Contributions to a Shared Signal

This script demonstrates how two isomers can contribute to the same measured signal,
each with distinct sensitivities and concentrations.

# Input Parameters
Configure the sensitivities and concentrations for both isomers in the `# %% input` section.

# Output
1. Voltage scanning (VS) curves for each individual isomer signal.
1. Combined (summed) signal curve.
2. A plot showing how the resulting VS results relate to the general correlation curve.

# Purpose
This visualization helps clarify how varying isomer sensitivities affect:
- The total observed signal,
- The resulting VS behavior,
- And their interpretation in quantitative analysis.

Useful for exploring overlapping signals and understanding mixed-contributor response dynamics.

"""

# %% packages

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from VSfit.VSfit import sigmodial, double_sigmoid, vs_fit
# custom figure style functions for layout and colors
from VSfit.FigureStyles import standard_plot_parameters, standard_colors
# import custo colors
orange, orange_light, orange_dark, orange_very_dark, purple, purple_light, purple_dark, purple_very_dark = standard_colors()


# %% input

# set ppb values for hypothetical szenario
isomer1_ppb = 10
isomer2_ppb = 10

# set sensitivitiers for the two isomers in cps/ppb (DCcorrected)
isomer1_sens = 3000
isomer2_sens = 600

# calcualte counts per second max signal detected
isomer1_cps = isomer1_sens*isomer1_ppb
isomer2_cps = isomer2_sens*isomer2_ppb

# example vs parameters for isomer1
isomer1_y_max = isomer1_cps
isomer1_center = 0.00018
isomer1_slope = 0.1

# example vs parameters for isomer2
isomer2_y_max = isomer2_cps
isomer2_center = 0.0014
isomer2_slope = 0.1

# example corealtion curve parameters
y_max = 5799.3
center = 85.3
slope = -8.6


# %% fucntions

def inverted_sigmoidal(y, y_max, center, slope):
    # inverted sigmoid correlation
    x = center + slope * np.log((y_max / y) - 1)
    return x


def double_sigmoid_only_fall(x, y_max, fall_center, fall_slope):
    y = (y_max)/((1+(fall_center*np.exp(fall_slope*x))))
    return y


# %% generate data

# voltage data
voltage_max = 120
voltage_data = np.linspace(0, voltage_max, voltage_max+1)

# calcualte dV50 values based on set sensitivities
# inverted sigmoid correlation to get dV50 values
isomer1_dV50 = inverted_sigmoidal(isomer1_sens, y_max, center, slope)
isomer2_dV50 = inverted_sigmoidal(isomer2_sens, y_max, center, slope)

# generate VS curve for isomer 1
isomer1_xdata = [0, isomer1_dV50]
isomer1_ydata = [isomer1_cps, isomer1_cps/2]
# generate VS curve for isomer 2
isomer2_xdata = [0, isomer2_dV50]
isomer2_ydata = [isomer2_cps, isomer2_cps/2]

# calcualte VS curve for each isomer and then combine to total signal
isomer1_signal_data = double_sigmoid_only_fall(voltage_data, isomer1_y_max, isomer1_center, isomer1_slope)
isomer2_signal_data = double_sigmoid_only_fall(voltage_data, isomer2_y_max, isomer2_center, isomer2_slope)
total_signal_data = isomer1_signal_data + isomer2_signal_data
# get max, dV50 and 0 for total signal
total_signal_dV50, total_signal_dV50_err, total_signal_dV50_r2, total_signal_dV50_parameters = vs_fit(
                                                                                                      total_signal_data,
                                                                                                      voltage_data,
                                                                                                      step_size=1,
                                                                                                      max_voltage=120
                                                                                                      )

total_signal_xdata = [0, total_signal_dV50]
total_signal_ydata = [total_signal_data.max(), total_signal_data.max()/2]
# plot data
figscale = 2.5
fig, ax = plt.subplots(figsize=(figscale * 3, figscale * 2))
# plot isomer 1
ax.scatter(isomer1_xdata, isomer1_ydata, color=purple, edgecolor=purple_dark, s=200, zorder=2)
ax.plot(voltage_data, isomer1_signal_data, color=purple, linewidth=5, label='isomer1', zorder=0)
# plot isomer 2
ax.scatter(isomer2_xdata, isomer2_ydata, color=purple_dark, edgecolor=purple_very_dark, s=200, zorder=2)
ax.plot(voltage_data, isomer2_signal_data, color=purple_dark, linewidth=5, label='isomer2', zorder=0)
# plot sum
ax.scatter(total_signal_xdata, total_signal_ydata, color=orange, edgecolor=purple_very_dark, s=200, zorder=2)
ax.plot(voltage_data, total_signal_data, color=orange, linewidth=5, label='total signal', zorder=1)

plt.xlim((1, 120))
plt.ylim((-1000, 37000))
plt.xlabel('$U$ [$V$]')
plt.ylabel('Signal [cps]')
standard_plot_parameters(ax)
plt.legend()

plt.show()


# %% show in correl plot

correl_fit_ydata = sigmodial(voltage_data, y_max, center, slope)

fig, ax = plt.subplots(figsize=(figscale * 3, figscale * 2))
ax.plot(voltage_data, correl_fit_ydata, color='black', zorder=0, label='correlation curve')
ax.scatter(isomer1_dV50, isomer1_sens,c=purple, s=200, zorder=1, label='isomer 1')
ax.scatter(isomer2_dV50, isomer2_sens, c= purple_dark, s= 200, zorder=1, label='isomer 2')
sens_vs_result=sigmodial(total_signal_dV50, y_max, center, slope)
ax.scatter(total_signal_dV50, sens_vs_result,c= orange, s= 200, zorder=1, label='total')
plt.xlim((0,120))
plt.ylim((-300,6000))
plt.xlabel('$\Delta V_{50}$ [$V$]')
plt.ylabel('Sensitivity [cps/ppbV]')
standard_plot_parameters(ax)
plt.legend()

plt.show()
