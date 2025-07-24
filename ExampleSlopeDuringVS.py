# -*- coding: utf-8 -*-
"""
Created on Mon May 19 10:32:15 2025

@author: m.roska
"""

# %% packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from VoltageScanning.VSfit import generate_calibration_curve, sigmodial, sigmodial_err_prop, double_sigmoid, vs_fit

# %%input parameters

# for vs curve
x_data = np.linspace(0, 100, num=50)
x_data_seconds = np.linspace(0, 5, num=50)
y_max = 1.68951
fall_center = 0.00223759
fall_slope = 0.100525
rise_center = 0.832066
rise_slope = 1.1392e-22

# for slope
slope = -15
# calcualte actuall change per volt
# averages were taken in the widow -15 to -5 (midpoint -10) and +10 to +20 (midpoint +15) resulting in a timing between -10 and +15 of 25s
slope_per_second = slope/25
# at 10 Hz, 100V, 2V per step we ahve 20 V/s
slope_per_volt = slope_per_second/20


# %% generate simoidal function
y_data = double_sigmoid(x_data, y_max, fall_center, fall_slope, rise_center, rise_slope)
# normalize y data
y_data = y_data/y_data.max()
# lin decay
y_slope = (100 + x_data * slope_per_volt)/100
# curve - slope
y_corrected = y_data - 1 + y_slope
y_corrected = y_corrected.clip(min=0)

# %% plot curve

plt.plot(x_data,y_data)
plt.plot(x_data,y_slope)
plt.plot(x_data,y_corrected)

vs_result = vs_fit(y_data, x_data_seconds, plot_vs=False)
plt.scatter(vs_result[0], 0.5)

# %% iter over slopes and run vs
slope_min = -100
slope_max = 100
iter_range = np.linspace(slope_min, slope_max, num=10)

# Normalize slope values for colormap
slope_min = min(iter_range)
slope_max = max(iter_range)
norm = plt.Normalize(slope_min, slope_max)
cmap = cm.get_cmap('plasma')

plt.figure()

for ind, slope in enumerate(iter_range):
    slope_per_second = slope / 25
    slope_per_volt = slope_per_second / 20

    y_data = double_sigmoid(x_data, y_max, fall_center, fall_slope, rise_center, rise_slope)
    y_data = y_data / y_data.max()

    y_slope = (100 + x_data * slope_per_volt) / 100
    y_corrected = y_data - 1 + y_slope
    y_corrected = y_corrected.clip(min=0)

    # Get color from colormap based on normalized slope
    color = cmap(norm(slope))

    plt.plot(x_data, y_corrected, color=color, linewidth = 3, label=f'slope={slope:.2f}')

plt.xlabel("V")
plt.ylabel("signal")
plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label='Slope [%]')
plt.tight_layout()
plt.show()
    

    
# %% plot dV50 depdencies

slope_min = -50
slope_max = 50
iter_range = np.linspace(slope_min, slope_max, num=100)

for ind, slope in enumerate(iter_range):
    # calcualte actuall change per volt
    # averages were taken in the widow -15 to -5 (midpoint -10) and +10 to +20 (midpoint +15) resulting in a timing between -10 and +15 of 25s
    slope_per_second = slope/25
    # at 10 Hz, 100V, 2V per step we ahve 20 V/s
    slope_per_volt = slope_per_second/20
    y_data = double_sigmoid(x_data, y_max, fall_center, fall_slope, rise_center, rise_slope)
    # normalize y data
    y_data = y_data/y_data.max()
    # lin decay
    y_slope = (100 + x_data * slope_per_volt)/100
    # curve - slope
    y_corrected = y_data - 1 + y_slope
    y_corrected = y_corrected.clip(min=0)

    #plt.plot(x_data,y_data)
    #plt.plot(x_data,y_slope)
    #plt.plot(x_data,y_corrected)

    vs_result = vs_fit(y_corrected, x_data_seconds, plot_vs=False)
    #plt.scatter(vs_result[0], 0.5)
    if abs(slope) <= 15:
        color = 'orange'
    else:
        color = 'grey'
        
    plt.scatter(slope, vs_result[0], c = color)
    
plt.xlim(slope_min,slope_max)
plt.grid()
