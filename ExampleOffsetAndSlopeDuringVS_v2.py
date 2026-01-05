# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 15:27:04 2025

@author: m.roska
"""

# %% packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# custom
from Common_Functions.StandardSetupAndCommonFunctions import standard_plot_parameters, standard_colors
from VoltageScanning.VSfit import sigmodial, vs_fit
orange, orange_light, orange_dark, orange_very_dark, purple, purple_light, purple_dark, purple_very_dark = standard_colors()


# %% functions



# %% input

vs_length = 60

y_max_input = 1
center_input = 15
slope_input = 4

# %% generate example data

xdata_input = np.linspace(0,260, num=260)

# Part 1: y = 1 for x < 100
ydata_1 = np.ones_like(xdata_input[xdata_input < 100])
# Part 2: sigmoid from x = 100 to 160
x_sigmoid = xdata_input[(xdata_input >= 100) & (xdata_input < 160)]
ydata_2 = sigmodial(x_sigmoid-100, y_max_input, center_input, slope_input)
# Part 3: y = 0 for x >= 160
ydata_3 = np.zeros_like(xdata_input[xdata_input >= 160])
# Combine all parts
signal_data_input = np.concatenate([ydata_1, ydata_2, ydata_3])

plt.plot(xdata_input, signal_data_input)


# %% run vs fitting with varying offsets

vs_results_collect = pd.Series()

for i in range(100):
    offset = i-50
    timing_data = np.linspace(0, 60, num=60)
    signal_data = signal_data_input[100 + offset :160 + offset]
    
    vs_result, vs_result_err, vs_result_r2, parameters = vs_fit(signal_data,
                                                                timing_data,
                                                                convert_to="Volt",
                                                                acquisition_freq=1,
                                                                tunning_freq=None,
                                                                StepSize=2,
                                                                init_fit_parameters=None,
                                                                max_voltage=120,
                                                                fit_used="double_sigmoid",
                                                                max_fit_iterations=5000,
                                                                low_bounds=(0, 0, 0, 0, 0),
                                                                up_bounds=(np.inf, np.inf, np.inf, np.inf, np.inf),
                                                                plot_vs=True,
                                                                plot_title='',
                                                                propagate_Error=False)
    
    vs_results_collect[i] = vs_result
    

# %% plot vs resutl versus offset

xdata = np.linspace(-50,len(vs_results_collect)-50,len(vs_results_collect))
ydata = vs_results_collect

plt.scatter(xdata, ydata)


# %% run vs fitting with varying slopes

vs_results_collect = pd.Series()

plt.figure()
for i in range(100):
    # gives slopes from -50 to 50%
    slope = i-50
    timing_data = np.linspace(0, 60, num=60)
    
    center_before = -15
    center_after = +15
    length_slope = center_before + vs_length + center_after
    signal_before = 1
    signal_after = 1 + 0.01*slope
    slope_per_datapoint = (signal_after - signal_before) / length_slope
    lin_rise = slope_per_datapoint * timing_data
    
    signal_data_without_slope = signal_data_input[100 :160] 
    signal_data_with_slope = signal_data_without_slope + lin_rise
    signal_data_with_slope[signal_data_with_slope < 0] = 0
    
    vs_result, vs_result_err, vs_result_r2, parameters = vs_fit(signal_data_with_slope,
                                                                timing_data,
                                                                convert_to="Volt",
                                                                acquisition_freq=1,
                                                                tunning_freq=None,
                                                                StepSize=2,
                                                                init_fit_parameters=None,
                                                                max_voltage=120,
                                                                fit_used="double_sigmoid",
                                                                max_fit_iterations=5000,
                                                                low_bounds=(0, 0, 0, 0, 0),
                                                                up_bounds=(np.inf, np.inf, np.inf, np.inf, np.inf),
                                                                plot_vs=False,
                                                                plot_title='',
                                                                propagate_Error=False)
    vs_results_collect[i] = vs_result
    
    plt.plot(timing_data,signal_data_with_slope,color=purple,alpha=i/100)
    
# %% plot vs resutl versus slope

xdata = np.linspace(-50,len(vs_results_collect)-50,len(vs_results_collect))
ydata = vs_results_collect

scale = 60

plt.scatter(xdata, ydata/scale)
plt.hlines(60/scale, -50, 50)
plt.hlines(60/scale, -50, 50)
plt.hlines(60/scale, -50, 50)

plt.vlines(0, 55/scale, 65/scale)
plt.vlines(-15, 55/scale, 65/scale)
plt.vlines(15, 55/scale, 65/scale)

plt.xlabel('slope (%)')
plt.ylabel('dV50')

# %% example data


# 30 V
dv30 = 30
dv30_n15 = 28.8483
dv30_n30 = 27.7273
dv30_p15 = 31.8802
dv30_p30 = None

# 60 V
dv60 = 60
dv60_n15 = 57.4503
dv60_n30 = 55.108
dv60_p15 = 64.7669
dv60_p30 = 73.7591
