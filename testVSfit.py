# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:09:12 2024

@author: m.roska
"""
#%%
#These 3 functions are required to import from the VSfit module to perform a VS conversion.
from VSfit import vs_fit, generate_calibration_curve, sigmodial
#Pandas loads and handles example datasets
import pandas as pd
#%%
# Load the CSV file into a DataFrame
# Input Path of DataSet
df = pd.read_csv('C://Users/m.roska/OneDrive - Forschungszentrum Jülich GmbH/Desktop/VSExampleData.csv')
df = df[:-15]

#get timing and signal data from the importat dataset
timing_data = df['xVS']
signal_data = df['yVS']

#%% run VS fit over the chosen timing and signal data to get a VS result value
#this is the core fitting function of the VS method
#the function converts the data to U^2 if not specified otherwise and
#standard parameters for Votlage Stepsize, accquisition and tunning frequency and Voltage range of the scan are set, but can be changed in the fucntion input
#after conversion the data is fitted using per default a double sigmoid function
#the U^2 value at 50% of the starting signal is determined and returned with the R^2 value of the fit (1 being the perfect fit)

vs_result, vs_result_r2 = vs_fit(signal_data, timing_data, plot_vs=True)

#%% test correlation function
#This fucntion generates a correlation function between VSresult data and calibrated sensitivities for calibration compounds
#VS result data (u^2 at 50%) has to be inputted for each calibration compound, as well as the sensitivity derived from a conventional calibration
#the function fits VS result data against sensitvities and fits a sigmoid function
#the function returns the fitting paramers of the sigmoidal function as well as the r^2 value of the fit

#Example dat for calibration compounds
u2 = [0,10,20,30,40,50,60,70,80]
sens = [1,1, 1, 4, 6, 8,9,9,9]

parameters, r2 = generate_calibration_curve(u2,sens,plotCalCurve=True)

#%% use sigmoidal to convert to sens
#Using the parameters derived from the generate_calibration_curve function in the previos section, a sensitivity can be derived from a vs result of an uncalibrated compound
#using the sigmoidal function and the unpacked paramerts, a sensitivity is calcualted


#unpacking the parameters from the generated calibration curve
fit_y_max, fit_center, fit_slope = parameters


sen_rResult = sigmodial(vs_result, fit_y_max, fit_center, fit_slope)
