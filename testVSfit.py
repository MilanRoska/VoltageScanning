# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:09:12 2024

@author: m.roska
"""
#%%
import VSfit
import pandas as pd
#%%
# Load the CSV file into a DataFrame
df = pd.read_csv('C://Users/m.roska/OneDrive - Forschungszentrum Jülich GmbH/Desktop/VSExampleData.csv')
df = df[:-15]

TimingData = df['xVS']
SignalData = df['yVS']

#%%
VSResult, VSResultR2 = VSfit.VSfit(SignalData, TimingData, plotVS=True,FitUsed = "Gauss")