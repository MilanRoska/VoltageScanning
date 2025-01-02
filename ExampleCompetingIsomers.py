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
isomer1_signal_data = double_sigmoid_only_fall(voltage_data, isomer1_y_max, isomer1_center, isomer1_slope)
plt.scatter(isomer1_xdata, isomer1_ydata)
plt.plot(voltage_data, isomer1_signal_data, label='isomer1')

# generate VS curve for isomer 2
isomer2_xdata = [0, isomer2_dV50, voltage_max]
isomer2_ydata = [isomer2_cps, isomer2_cps/2, 0]
# guess parameters for isomer1
isomer2_y_max = isomer2_cps
isomer2_center = 0.0014
isomer2_slope = 0.1
# plot data
isomer2_signal_data = double_sigmoid_only_fall(voltage_data, isomer2_y_max, isomer2_center, isomer2_slope)
plt.scatter(isomer2_xdata, isomer2_ydata)
plt.plot(voltage_data, isomer2_signal_data, label='isomer2')

# summ up both
total_signal_data = isomer1_signal_data + isomer2_signal_data
plt.plot(voltage_data, total_signal_data, label='total signal')

# calcualte dV50 
vs_result, vs_result_r2, parameters = vs_fit(total_signal_data, voltage_data, convert_to="Volt", acquisition_freq=1, tunning_freq=None, StepSize=1, init_fit_parameters=None, max_voltage=voltage_max, fit_used="double_sigmoid", max_fit_iterations=5000, inversion_method="root_scalar", plot_vs=False, plot_title = 'formula not specified')
plt.scatter(vs_result, max(total_signal_data)/2)
plt.legend()
plt.show()


# %% show in correl plot

correl_fit_ydata = sigmodial(voltage_data, y_max, center, slope)
plt.plot(voltage_data, correl_fit_ydata)
plt.scatter(isomer1_dV50, isomer1_sens)
plt.scatter(isomer2_dV50, isomer2_sens)
sens_vs_result = sigmodial(vs_result, y_max, center, slope)
plt.scatter(vs_result, sens_vs_result)
