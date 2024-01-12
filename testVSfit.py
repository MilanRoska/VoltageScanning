# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:09:12 2024

@author: m.roska
"""
#%%
from VSfit import vs_fit, generate_calibration_curve, sigmodial
import pandas as pd
#%%
# Load the CSV file into a DataFrame
df = pd.read_csv('C://Users/m.roska/OneDrive - Forschungszentrum Jülich GmbH/Desktop/VSExampleData.csv')
df = df[:-15]

timing_data = df['xVS']
signal_data = df['yVS']

#%%
vs_result, vs_result_r2 = vs_fit(signal_data, timing_data, plot_vs=True)

#%% test correlation function

u2 = [0,10,20,30,40,50,60,70,80]
sens = [1,1, 1, 4, 6, 8,9,9,9]

parameters, r2 = generate_calibration_curve(u2,sens,plotCalCurve=True)

#%% use sigmoidal to convert to sens
fit_y_max, fit_center, fit_slope = parameters
sen_rResult = sigmodial(vs_result, fit_y_max, fit_center, fit_slope)
