# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:12:30 2024

@author: m.roska
"""

#%%packages
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
#%%custom colors
orange = (244/255 , 153/255 , 62/255 , 100/100)
orange_light = (252/255 , 230/255 , 207/255 , 100/100)
orange_dark = (145/255 , 77/255 , 8/255 , 100/100)
orange_very_dark = (73/255 , 39/255 , 4/255 , 100/100)
purple = (215/255 , 61/255 , 245/255 , 100/100)
purple_light = (252/255 , 240/255 , 254/255 , 100/100)
purple_dark = (122/255 , 8/255 , 145/255 , 100/100)
purple_very_dark = (61/255 , 4/255 , 73/255 , 100/100)

#%%Support Fnctions


def standard_plot_parameters(ax):
    plt.tight_layout()
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 12
    plt.legend(loc='upper right',handlelength=1, scatterpoints=1)
    ax.tick_params(axis='both', direction='in', length=5, width=1)
    ax.tick_params(which='major', size=5)  # Major ticks
    ax.tick_params(which='minor', size=5)   # Minor ticks
    # Show ticks on all four sides of the plot
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    #plt.style.use('dark_background')
    plt.style.use('classic')
    return


def double_sigmoid(x,y_max,fall_center,fall_slope,rise_center,rise_slope):
    y = (y_max)/((1+(fall_center*np.exp(fall_slope*x)))*(1+(rise_center/np.exp(rise_slope*x))))
    return y

def double_sigmoid_find_root_scalar(y_target, y_max, fall_center, fall_slope, rise_center, rise_slope, x_guess=50):
    # Define the objective function to find the root of
    objective_function = lambda x: double_sigmoid(x, y_max, fall_center, fall_slope, rise_center, rise_slope) - y_target

    # Find the root using root_scalar
    result = root_scalar(objective_function, bracket=[0, 200], x0=x_guess)

    if result.converged:
        return result.root
    else:
        raise RuntimeError("Failed to find x for the given y.")

def gauss_amp(x,y_0,x_center,w,area):
    y = y_0+area*np.exp((-((x-x_center)**2))/(2*w**2))
    return y

def gauss_amp_inv(y,y_0,x_center,w,area):
    x = np.sqrt(-2*w**2*np.log((y-y_0)/area))+x_center
    return x

def sigmodial(x,y_max,center,slope):
    y =  y_max / (1 + np.exp((x - center) / slope))
    return y

#%%Main Function for VS fitting

def vs_fit(signal_data, timing_data, convert_to="Volt", acquisition_freq=10, tunning_freq=None, StepSize=2, init_fit_parameters=None, max_voltage=100, fit_used="double_sigmoid", max_fit_iterations=5000, plot_vs=False, plot_title = 'formula not specified'):
    # Set default value for tunning_freq
    if tunning_freq is None:
        tunning_freq = acquisition_freq
    elif tunning_freq > acquisition_freq:
        raise ValueError("VS Tunning Frequency faster than Acquisition Frequency.")
    else:
        pass
    
    #chek if timing and Signal Data have same dimension -> error
    if signal_data.shape != timing_data.shape:
        raise ValueError("Signal and timing Data must have the same dimension.")
        
    #check if any values are negative or NaN -> error
    if any(signal_data < 0) or any(np.isnan(signal_data)):
        raise ValueError("Signal Data contains negative values or NaN.")
    if any(np.isnan(timing_data)):
        raise ValueError("Timing Data contains NaN values.")
    
    #calcualte correction factor for if slower tunning than acquisition
    slowness_correction_factor = tunning_freq/acquisition_freq
    
    #check if Nr of Datapoints matches Expected from VS param -> error
    expected_data_points = max_voltage/StepSize/slowness_correction_factor
    if expected_data_points != timing_data.shape[0]:
        print(f"More Data Points than expected from VS setting. {expected_data_points} expected, {timing_data.shape[0]} found")
        
    #set timing Data to Start at 0
    timing_data = timing_data-min(timing_data)
    
    #convert time (x) to Volt or Ekin
    valid_convert_to_values = ["Volt", "Ekin"]
    if convert_to == "Volt":
    # Code for "Volt" conversion
     timing_dataConverted = (timing_data*StepSize*slowness_correction_factor*tunning_freq)
    elif convert_to == "Ekin":
    # Code for "Ekin" conversion
        raise NotImplementedError("Not yet implemented in package.")
    else:
        raise ValueError(f"Invalid value for convert_to: {convert_to}. Choose from {valid_convert_to_values}")
    
    #Normalize Signal (y) to highest value
    signal_data_normalized = signal_data/max(signal_data)
    
    try:
        #fit TS
        valid_fit_used_values = ["double_sigmoid", "Gauss"]
        if fit_used == "double_sigmoid":
            #set boundary conditions 
            low_bounds = (0,0,0,0,0)
            #low_bounds = (-np.inf,-np.inf,-np.inf,-np.inf,-np.inf)
            up_bounds = (np.inf,np.inf,np.inf,np.inf,np.inf)
            #perform double_sigmoid fit
            parameters, covariance = curve_fit(double_sigmoid, timing_dataConverted, signal_data_normalized, p0=init_fit_parameters, maxfev=max_fit_iterations, bounds= (low_bounds,up_bounds))
            #unpack parameters
            fit_y_max,fit_fall_center, fit_fall_slop, rise_center, rise_slope = parameters  
            #calculate fit curve
            fit_signal_data = double_sigmoid(timing_dataConverted,fit_y_max,fit_fall_center, fit_fall_slop, rise_center, rise_slope)
            #r2 value
            #r-squared value between data and fit curve
            r2 = r2_score(signal_data_normalized,fit_signal_data)
            #get start value from signal_data
            int_start = double_sigmoid(0,fit_y_max,fit_fall_center, fit_fall_slop, rise_center, rise_slope) 
            #get Intensity at at half of int_start
            int_half = int_start/2
            #get Volt_half from int_half value using minimized scalar residual method
            # Perform the optimization to find the best x value
            Volt_half = double_sigmoid_find_root_scalar(int_half,fit_y_max,fit_fall_center, fit_fall_slop, rise_center, rise_slope)
            vs_result = Volt_half
            vs_result_r2 = r2
        elif fit_used == "Gauss":   
            #set boundary conditions 
            low_bounds = (-np.inf,-np.inf,-np.inf,-np.inf)
            up_bounds = (np.inf,np.inf,np.inf,np.inf)
            #perform double_sigmoid fit
            parameters, covariance = curve_fit(gauss_amp, timing_dataConverted, signal_data_normalized, p0=init_fit_parameters, maxfev=max_fit_iterations, bounds= (low_bounds,up_bounds))
            #unpack parameters
            fit_y_0, fit_x_center, fit_w, fit_area = parameters  
            #calculate fit curve
            fit_signal_data = gauss_amp(timing_dataConverted, fit_y_0, fit_x_center, fit_w, fit_area)
            #r2 value
            #r-squared value between data and fit curve
            r2 = r2_score(signal_data_normalized,fit_signal_data)
            #get start value from signal_data
            int_start = gauss_amp(0, fit_y_0, fit_x_center, fit_w, fit_area) 
            #get Intensity at at half of int_start
            int_half = int_start/2
            #get ekin_half from int_half value using minimized scalar residual method
            ekin_half = gauss_amp_inv(int_half, fit_y_0, fit_x_center, fit_w, fit_area)
            vs_result = ekin_half
            vs_result_r2 = r2
        else:
            raise ValueError(f"Invalid value for fit_used: {fit_used}. Choose from {valid_fit_used_values}")

    except (RuntimeError, RuntimeWarning, ValueError) as e:
        raise RuntimeError("fit not possible") from e
    
    #plot VS
    if plot_vs == True:
        fig, ax = plt.subplots(figsize=(5.5,4.125))
        ax.scatter(timing_dataConverted, signal_data_normalized, label='Data',color =orange, s=50, zorder=1)
        ax.plot(timing_dataConverted, fit_signal_data, label = 'fit. (r$^{2}=$'+str(round(r2,3))+')',color =purple_very_dark, linewidth = 3, zorder=2)
        ax.scatter(0,int_start,color =purple_very_dark,label='100%',s=100, marker = 'H',zorder=3)
        ax.scatter(vs_result,int_half,color =purple_very_dark,label='50% ('+convert_to+'='+str(round(vs_result,2))+')',s=100, marker = 'h',zorder=3)       
        standard_plot_parameters(ax)
        plt.ylabel('Intensity [AU]')
        if convert_to == "Volt":
            plt.xlabel('$U$ [$V$]')
        plt.title(plot_title)
    #output vs_result
    return vs_result, vs_result_r2, parameters

#%%Main Function for Correlating Calcualting Sensitivities and generating Conversion functuin

#Input vs_result and CalResult Data and Names for Calibration compounds
#fit parameters in order: y_max, center, slope
def generate_calibration_curve(cal_vs_values, cal_sens_values, plot_cal_curve=False, converted_to="Volt", low_bounds_input = (0,-np.inf,-np.inf), up_bounds_input = (np.inf,np.inf,0)):
    #set boundary conditions 
    low_bounds = low_bounds_input
    up_bounds = up_bounds_input
    #perform Sigm fit
    parameters, covariance = curve_fit(sigmodial, cal_vs_values, cal_sens_values, maxfev=10000, bounds= (low_bounds,up_bounds))
    #unpack parameters
    fit_y_max, fit_center, fit_slope = parameters
    #calculate fit curve
    fit_sens_data = sigmodial(cal_vs_values, fit_y_max, fit_center, fit_slope)
    #r2 value
    #r-squared value between data and fit curve
    cal_curve_r2 = r2_score(cal_sens_values,fit_sens_data)
    #plot
    if plot_cal_curve == True:
        #generate plot x dataaxis Dataset
        plot_fit_vs_values = np.linspace(0, max(cal_vs_values)+1.1,1000)
        Plotfit_sens_data = sigmodial(plot_fit_vs_values,fit_y_max, fit_center, fit_slope)
        fig, ax = plt.subplots(figsize=(5.5,3.44))
        ax.scatter(cal_vs_values, cal_sens_values, c = orange, edgecolor =  'None' ,s=60,zorder=2,label = 'cal. compounds')
        ax.plot(plot_fit_vs_values, Plotfit_sens_data, label= 'sigm. fit (r$^{2}=$'+str(round(cal_curve_r2,3))+')',color =purple, linewidth = 1, zorder=3)
        standard_plot_parameters(ax)
        plt.ylabel('Sensitivity')
        if converted_to == "Volt":
            plt.xlabel('$U$ [$V$]')
        plt.rcParams['legend.fontsize'] = 10
        plt.legend(loc='best',handlelength=1)
    #output fit parameters 
    return parameters, cal_curve_r2 
