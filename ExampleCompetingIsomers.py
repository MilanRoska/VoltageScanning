# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:45:04 2024

@author: m.roska
"""

# %% packages
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from VoltageScanning.VSfit import sigmodial, double_sigmoid, vs_fit
from Common_Functions.StandardSetupAndCommonFunctions import standard_plot_parameters, standard_colors

orange, orange_light, orange_dark, orange_very_dark, purple, purple_light, purple_dark, purple_very_dark = standard_colors()

# %% input
# set ppb values for hypothetical szenario
isomer1_ppb = 10
isomer2_ppb = 10
# set sensitivitiers for the two isomers in cps/ppb (DCcorrected)
isomer1_sens = 3000
isomer2_sens = 600

# example covnersion parameters
# (AEROMMA period 2)
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
voltage_data = np.linspace(0, voltage_max, voltage_max)

# inverted sigmoid correlation to get dV50 values
isomer1_dV50 = inverted_sigmoidal(isomer1_sens, y_max, center, slope)
isomer2_dV50 = inverted_sigmoidal(isomer2_sens, y_max, center, slope)

# calcualte max signal detected
isomer1_cps = isomer1_sens*isomer1_ppb
isomer2_cps = isomer2_sens*isomer2_ppb


# generate VS curve for isomer 1
isomer1_xdata = [0, isomer1_dV50, voltage_max]
isomer1_ydata = [isomer1_cps, isomer1_cps/2, 0]
# guess parameters for isomer1
isomer1_y_max = isomer1_cps
isomer1_center = 0.00018
isomer1_slope = 0.1
# plot data
figscale = 2.5
fig, ax = plt.subplots(figsize=(figscale * 3, figscale * 2))

isomer1_signal_data = double_sigmoid_only_fall(voltage_data, isomer1_y_max, isomer1_center, isomer1_slope)
ax.scatter(isomer1_xdata, isomer1_ydata, color = purple, edgecolor = purple_dark, s = 200, zorder = 2)
ax.plot(voltage_data, isomer1_signal_data, color = purple, linewidth = 5, label='isomer1', zorder = 0)

# generate VS curve for isomer 2
isomer2_xdata = [0, isomer2_dV50, voltage_max]
isomer2_ydata = [isomer2_cps, isomer2_cps/2, 0]
# guess parameters for isomer1
isomer2_y_max = isomer2_cps
isomer2_center = 0.0014
isomer2_slope = 0.1
# plot data
isomer2_signal_data = double_sigmoid_only_fall(voltage_data, isomer2_y_max, isomer2_center, isomer2_slope)
ax.scatter(isomer2_xdata, isomer2_ydata, color = purple_dark, edgecolor = purple_very_dark, s = 200, zorder = 2)
ax.plot(voltage_data, isomer2_signal_data, color = purple_dark, linewidth = 5, label='isomer2', zorder = 0)

# summ up both
total_signal_data = isomer1_signal_data + isomer2_signal_data
ax.plot(voltage_data, total_signal_data, color = orange, linewidth = 5, label='total signal', zorder = 1)

# calcualte dV50 
vs_result, vs_result_r2, parameters, parameters_err = vs_fit(total_signal_data, voltage_data, convert_to="Volt", acquisition_freq=1, tunning_freq=None, StepSize=1, init_fit_parameters=None, max_voltage=voltage_max, fit_used="double_sigmoid", max_fit_iterations=5000, inversion_method="root_scalar", plot_vs=False, plot_title = 'formula not specified')
ax.scatter(vs_result, max(total_signal_data)/2, color = orange, edgecolor = orange_dark, s = 200, zorder = 2)
plt.xlim((1,120))
plt.ylim((-1000,37000))
plt.legend()
standard_plot_parameters(ax)
plt.savefig("C://Users/m.roska/OneDrive - Forschungszentrum Jülich GmbH/Desktop/test.svg", format='svg',bbox_inches = 'tight')

plt.show()


# %% show in correl plot

correl_fit_ydata = sigmodial(voltage_data, y_max, center, slope)

fig, ax = plt.subplots(figsize=(figscale * 3, figscale * 2))
ax.plot(voltage_data, correl_fit_ydata, color = 'black', zorder = 0, label = 'correlation curve')
ax.scatter(isomer1_dV50, isomer1_sens,c=purple, s = 200, zorder = 1, label = 'isomer 1')
ax.scatter(isomer2_dV50, isomer2_sens, c= purple_dark, s= 200, zorder = 1, label = 'isomer 2')
sens_vs_result = sigmodial(vs_result, y_max, center, slope)
ax.scatter(vs_result, sens_vs_result,c= orange, s= 200, zorder = 1, label = 'total')
standard_plot_parameters(ax)
plt.legend(loc = 'best')
plt.xlim((0,120))
plt.ylim((-300,6000))
plt.savefig("C://Users/m.roska/OneDrive - Forschungszentrum Jülich GmbH/Desktop/test.svg", format='svg',bbox_inches = 'tight')

plt.show()
