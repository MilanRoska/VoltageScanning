# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:12:30 2024

@author: m.roska
"""

# %%packages
import sys
import numpy as np
from scipy.optimize import curve_fit
from scipy import odr
from sklearn.metrics import r2_score
from scipy.optimize import root_scalar
from scipy.optimize import newton
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

sys.path.append('C:/Users/exp.voc/PythonRepo/')


# %%custom colors
orange = (244/255 , 153/255 , 62/255 , 100/100)
orange_light = (252/255 , 230/255 , 207/255 , 100/100)
orange_dark = (145/255 , 77/255 , 8/255 , 100/100)
orange_very_dark = (73/255 , 39/255 , 4/255 , 100/100)
purple = (215/255 , 61/255 , 245/255 , 100/100)
purple_light = (252/255 , 240/255 , 254/255 , 100/100)
purple_dark = (122/255 , 8/255 , 145/255 , 100/100)
purple_very_dark = (61/255 , 4/255 , 73/255 , 100/100)


# %%Support Fnctions


def standard_plot_parameters(ax):
    plt.tight_layout()
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 12
    plt.legend(loc='upper right', handlelength=1, scatterpoints=1)
    ax.tick_params(axis='both', direction='in', length=5, width=1)
    ax.tick_params(which='major', size=5)  # Major ticks
    ax.tick_params(which='minor', size=5)   # Minor ticks
    # Show ticks on all four sides of the plot
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    #plt.style.use('dark_background')
    plt.style.use('classic')
    return


def double_sigmoid(x, y_max, fall_center, fall_slope, rise_center, rise_slope):
    y = (y_max)/((1+(fall_center*np.exp(fall_slope*x)))*(1+(rise_center/np.exp(rise_slope*x))))
    return y


def double_sigmoid_err(x, y_max, fall_center, fall_slope, rise_center, rise_slope, y_max_err, fall_center_err, fall_slope_err, rise_center_err, rise_slope_err):
    y_err = np.sqrt(
        (y_max_err*(1 / ((rise_center*np.exp(-rise_slope*x) + 1)*(fall_center*np.exp(fall_slope*x) + 1))))**2 +
        (fall_center_err*(-(y_max*np.exp(fall_slope*x)) / ((rise_center*np.exp(-rise_slope*x) + 1)*(np.exp(fall_slope*x)*fall_center + 1)**2)))**2 +
        (fall_slope_err*(-(y_max*fall_center*x*np.exp(x*fall_slope)) / ((rise_center*np.exp(-rise_slope*x) + 1)*(fall_center*np.exp(x*fall_slope) + 1)**2)))**2 +
        (rise_center_err*(-(y_max*np.exp(rise_slope*x)) / ((fall_center*np.exp(fall_slope*x) + 1)*(rise_center + np.exp(rise_slope*x))**2)))**2 +
        (rise_slope_err*((y_max*rise_center*x*np.exp(x*rise_slope)) / ((fall_center*np.exp(fall_slope*x) + 1)*(np.exp(x*rise_slope) + rise_center)**2)))**2
        )
    return y_err


def double_sigmoid_find_root_scalar(y_target, y_max, fall_center, fall_slope, rise_center, rise_slope, x_guess=40):
    """
    Finds the x-value such that double_sigmoid(x) = y_target.

    Tries the bracketing method first. If that fails, it falls back to Newton's method.
    """
    # Define the objective function to find the root of
    def objective_function(x):
        return double_sigmoid(x, y_max, fall_center, fall_slope, rise_center, rise_slope) - y_target

    # Check function values at bracket edges
    f_a = objective_function(0)
    f_b = objective_function(300)

    if np.sign(f_a) != np.sign(f_b):
        # Bracketing method: Root must be within [0, 300]
        try:
            result = root_scalar(objective_function, bracket=[0, 300], x0=x_guess)
            if result.converged:
                return result.root
        except ValueError:
            pass  # Fall back to Newton's method if bracketing fails

    # If no sign change in the bracket, try Newton's method
    try:
        root = newton(objective_function, x_guess)
        return root
    except RuntimeError as e:
        raise RuntimeError(f"Failed to find x for the given y_target using both methods: {e}")

    # If all fails, raise an error
    raise RuntimeError("Failed to find x for the given y_target.")


def gauss_amp(x, y_0, x_center, w, area):
    y = y_0+area*np.exp((-((x-x_center)**2))/(2*w**2))
    return y


def gauss_amp_inv(y, y_0, x_center, w, area):
    x = np.sqrt(-2*w**2*np.log((y-y_0)/area))+x_center
    return x


def sigmodial(x ,y_max, center, slope):
    y = y_max / (1 + np.exp((x - center) / slope))
    return y


def sigmodial_gaussian_conv(x ,y_max_sigm, center_sigm, slope_sigm, y_max_gauss, area_gauss, center_gauss, width_gauss):
    y = y_max_sigm / (1 + np.exp((x - center_sigm) / slope_sigm)) + y_max_gauss + area_gauss *np.exp((-((x- center_gauss)**2))/(2*width_gauss**2))
    return y


def box_lucas(x, a, b):
    y = (a/(a-b))*(np.exp(-a*x) - np.exp(-b*x))
    return y


def box_lucas_inv(y, a, b):
    x = 2 * (np.log(a) - np.log(b)) / (a-b)
    return x


def sigmodial_for_odr(B, x):
    y_max, center, slope = B
    return y_max / (1 + np.exp((x - center) / slope))


def sigmodial_err_prop(x, y_max, center, slope, x_err=0, y_max_err=0, center_err=0, slope_err=0):
    y_err = np.sqrt(
        (x_err * (y_max * np.exp((x - center) / slope)) / (slope * (1 + np.exp((x - center) / slope))**2))**2 +
        (center_err * (y_max * np.exp((x - center) / slope)) / (slope * (1 + np.exp((x - center) / slope))**2))**2 +
        (slope_err * (y_max * (x - center) * np.exp((x - center) / slope)) / ((slope**2) * (1 + np.exp((x - center) / slope))**2))**2 +
        (y_max_err * (1 / (1 + np.exp((x - center) / slope))))**2
        )
    return y_err

# %% main Function for VS fitting

# vs fit function generating a dV50 value and an error
def vs_fit(signal_data,
           timing_data,
           convert_to="Volt",
           acquisition_freq=10,
           tunning_freq=None,
           StepSize=2,
           init_fit_parameters=None,
           max_voltage=100,
           fit_used="double_sigmoid",
           max_fit_iterations=5000,
           low_bounds=(0, 0, 0, 0, 0),
           up_bounds=(np.inf, np.inf, np.inf, np.inf, np.inf),
           plot_vs=False,
           plot_title='formula not specified',
           propagate_Error=False
           ):

    # Set default value for tunning_freq if not specified otherwise
    if tunning_freq is None:
        tunning_freq = acquisition_freq
    # failsafe for when tuning was faster then acquisition. This should be avoided
    elif tunning_freq > acquisition_freq:
        raise ValueError("VS Tunning Frequency faster than Acquisition Frequency.")
    # pass if tunning freq is defined
    else:
        pass

    # chek if timing and Signal Data have same dimension -> error
    if signal_data.shape != timing_data.shape:
        raise ValueError("Signal and timing Data must have the same dimension.")

    # check if any values are negative or NaN -> error
    if any(signal_data < 0) or any(np.isnan(signal_data)):
        raise ValueError("Signal Data contains negative values or NaN.")
    if any(np.isnan(timing_data)):
        raise ValueError("Timing Data contains NaN values.")

    # calcualte correction factor for if slower tunning than acquisition
    slowness_correction_factor = tunning_freq/acquisition_freq

    # check if Nr of Datapoints matches Expected from VS param -> error
    expected_data_points = max_voltage / StepSize / slowness_correction_factor
    if expected_data_points != timing_data.shape[0]:
        print(f"More Data Points than expected from VS setting. {expected_data_points} expected, {timing_data.shape[0]} found")

    # set timing Data to Start at 0
    timing_data = timing_data-min(timing_data)

    # convert time (x) to Volt or Ekin
    valid_convert_to_values = ["Volt", "Ekin"]
    if convert_to == "Volt":
        # Code for "Volt" conversion
        timing_dataConverted = (timing_data*StepSize*slowness_correction_factor*tunning_freq)
    elif convert_to == "Ekin":
        # Code for "Ekin" conversion
        raise NotImplementedError("Not yet implemented in package.")
    else:
        raise ValueError(f"Invalid value for convert_to: {convert_to}. Choose from {valid_convert_to_values}")

    # Normalize Signal (y) to start value value
    signal_data_normalized = signal_data / signal_data[0]

    try:
        # fit Voltage scanning curve
        valid_fit_used_values = ["double_sigmoid", "Gauss", "closest_datapoint", "closest_datapoint"]

        if fit_used == "double_sigmoid":
            # perform double_sigmoid fit
            parameters, covariance = curve_fit(double_sigmoid,
                                               timing_dataConverted,
                                               signal_data_normalized,
                                               p0=init_fit_parameters,
                                               maxfev=max_fit_iterations,
                                               bounds=(low_bounds, up_bounds)
                                               )
            # parameter errros
            parameters_errors = np.sqrt(np.diag(covariance))
            # unpack parameters
            fit_y_max, fit_fall_center, fit_fall_slope, fit_rise_center, fit_rise_slope = parameters
            # calculate fit curve
            fit_signal_data = double_sigmoid(timing_dataConverted,
                                             fit_y_max,
                                             fit_fall_center,
                                             fit_fall_slope,
                                             fit_rise_center,
                                             fit_rise_slope
                                             )
            # r-squared value between data and fit curve
            r2 = r2_score(signal_data_normalized, fit_signal_data)
            # get start value from signal_data
            int_start = double_sigmoid(0,
                                       fit_y_max,
                                       fit_fall_center,
                                       fit_fall_slope,
                                       fit_rise_center,
                                       fit_rise_slope
                                       )
            # get Intensity at at half of int_start
            int_half = int_start/2

            # Perform the optimization to find the best x value
            volt_half = double_sigmoid_find_root_scalar(int_half,
                                                        fit_y_max,
                                                        fit_fall_center,
                                                        fit_fall_slope,
                                                        fit_rise_center,
                                                        fit_rise_slope
                                                        )
            # calcualte an error if set active
            if propagate_Error:
                # unpack parameter errors
                fit_y_max_err, fit_fall_center_err, fit_fall_slope_err, fit_rise_center_err, fit_rise_slope_err = parameters_errors
                # calcualte int_half error for volt_half
                # calcuale upper volt error and lower volt error and take the maximum of both
                int_half_err = double_sigmoid_err(volt_half,
                                                  fit_y_max,
                                                  fit_fall_center,
                                                  fit_fall_slope,
                                                  fit_rise_center,
                                                  fit_rise_slope,
                                                  fit_y_max_err,
                                                  fit_fall_center_err,
                                                  fit_fall_slope_err,
                                                  fit_rise_center_err,
                                                  fit_rise_slope_err
                                                  )
                volt_half_err_upper = double_sigmoid_find_root_scalar(int_half+int_half_err,
                                                                      fit_y_max,
                                                                      fit_fall_center,
                                                                      fit_fall_slope,
                                                                      fit_rise_center,
                                                                      fit_rise_slope
                                                                      )
                volt_half_err_lower = double_sigmoid_find_root_scalar(int_half-int_half_err,
                                                                      fit_y_max,
                                                                      fit_fall_center,
                                                                      fit_fall_slope,
                                                                      fit_rise_center,
                                                                      fit_rise_slope
                                                                      )
                volt_half_err = abs(volt_half-max(volt_half_err_upper, volt_half_err_lower))
                vs_result_err = volt_half_err
            else:
                vs_result_err = None

            # set vs results and r2
            vs_result = volt_half
            vs_result_r2 = r2
  
        elif fit_used == "Gauss":
            # overwrite boundary conditions
            low_bounds = (-np.inf,
                          -np.inf,
                          -np.inf,
                          -np.inf
                          )
            up_bounds = (np.inf,
                         np.inf,
                         np.inf,
                         np.inf
                         )
            # perform gauss fit
            parameters, covariance = curve_fit(gauss_amp,
                                               timing_dataConverted,
                                               signal_data_normalized,
                                               p0=init_fit_parameters,
                                               maxfev=max_fit_iterations,
                                               bounds=(low_bounds, up_bounds)
                                               )
            # unpack parameters
            fit_y_0, fit_x_center, fit_w, fit_area = parameters
            # calculate fit curve
            fit_signal_data = gauss_amp(timing_dataConverted,
                                        fit_y_0, fit_x_center,
                                        fit_w,
                                        fit_area
                                        )
            # r2 value
            # r-squared value between data and fit curve
            r2 = r2_score(signal_data_normalized,
                          fit_signal_data
                          )
            # get start value from signal_data
            int_start = gauss_amp(0,
                                  fit_y_0,
                                  fit_x_center,
                                  fit_w,
                                  fit_area
                                  )
            # get Intensity at at half of int_start
            int_half = int_start/2
            # get ekin_half from int_half value using minimized scalar residual method
            ekin_half = gauss_amp_inv(int_half,
                                      fit_y_0,
                                      fit_x_center,
                                      fit_w,
                                      fit_area
                                      )

            if propagate_Error:
                print('Errorproapgation not yet implemented for Gauss fit')

            vs_result = ekin_half
            vs_result_err = None
            vs_result_r2 = r2
           
        # experimental fit curve to adress gauss rises in the slope
        elif fit_used == "sigmodial_gaussian_conv":
            # set boundary conditions
            low_bounds = (0,
                          0,
                          -100,
                          0,
                          0,
                          -200,
                          0)
            up_bounds = (2,
                         200,
                         0,
                         10,
                         np.inf,
                         200,
                         np.inf)
            # perform double_sigmoid fit
            parameters, covariance = curve_fit(sigmodial_gaussian_conv,
                                               timing_dataConverted,
                                               signal_data_normalized,
                                               p0=init_fit_parameters,
                                               maxfev=max_fit_iterations,
                                               bounds=(low_bounds, up_bounds)
                                               )
            # unpack parameters
            y_max_sigm, center_sigm, slope_sigm, y_max_gauss, area_gauss, center_gauss, width_gauss = parameters
            # calculate fit curve
            fit_signal_data = sigmodial_gaussian_conv(timing_dataConverted,
                                                      y_max_sigm,
                                                      center_sigm,
                                                      slope_sigm,
                                                      y_max_gauss,
                                                      area_gauss,
                                                      center_gauss,
                                                      width_gauss
                                                      )
            # r2 value
            # r-squared value between data and fit curve
            r2 = r2_score(signal_data_normalized, fit_signal_data)
            # get start value from signal_data
            int_start = sigmodial_gaussian_conv(0,
                                                y_max_sigm,
                                                center_sigm,
                                                slope_sigm,
                                                y_max_gauss,
                                                area_gauss,
                                                center_gauss,
                                                width_gauss
                                                )
            # get Intensity at at half of int_start
            int_half = int_start/2
            
            # !!!!!!!!!!!!!!!!!! impelemnt inversion fucntion
            # get ekin_half from int_half value using minimized scalar residual method
            ekin_half = sigmodial_gaussian_conv_inv(int_half,
                                                    y_max_sigm,
                                                    center_sigm,
                                                    slope_sigm,
                                                    y_max_gauss,
                                                    area_gauss,
                                                    center_gauss,
                                                    width_gauss
                                                    )
            vs_result = ekin_half
            vs_result_err = None
            vs_result_r2 = r2
            
        elif fit_used == "closest_datapoint":
            # smoothen data
            # set savgol window to 40% of length
            savgol_window = round(0.4 * len(signal_data_normalized),0)
            # apply savgol
            signal_data_normalized_smooth = savgol_filter(signal_data_normalized,
                                                          savgol_window,
                                                          2,mode='nearest')
            # get start value from signal_data
            int_start = signal_data_normalized_smooth[0]
            # get Intensity at at half of int_start
            int_half = int_start/2
            # Find the index where the difference is minimal
            closest_idx = np.argmin(np.abs(signal_data_normalized_smooth - int_half))
            # Use that index to get the corresponding timing_dataConverted value
            ekin_half = timing_dataConverted[closest_idx]
            # set fit signal to be the same as signal itself, sicne we are not fitting
            fit_signal_data = signal_data_normalized_smooth

            vs_result = ekin_half
            vs_result_err = None
            vs_result_r2 = -999
            parameters = [-999,-999,-999,-999,-999]
            
        else:
            raise ValueError(f"Invalid value for fit_used: {fit_used}. Choose from {valid_fit_used_values}")

    except (RuntimeError, RuntimeWarning, ValueError) as e:
        raise type(e)(f"fit not possible: {str(e)}").with_traceback(e.__traceback__)

    # plot VS
    if plot_vs == True:
        fig, ax = plt.subplots(figsize=(5.5, 4.125))
        ax.scatter(timing_dataConverted,
                   signal_data_normalized,
                   label='Data',
                   color=orange,
                   s=50,
                   zorder=1
                   )
        ax.plot(timing_dataConverted,
                fit_signal_data,
                label='fit. (r$^{2}=$'+str(round(vs_result_r2, 3))+')',
                color=purple_very_dark,
                linewidth=3,
                zorder=2
                )
        ax.scatter(0,
                   int_start,
                   color=purple_very_dark,
                   label='100%',
                   s=100,
                   marker='H',
                   zorder=3
                   )
        ax.scatter(vs_result,
                   int_half,
                   color=purple_very_dark,
                   label='50% ('+convert_to+'='+str(round(vs_result, 2))+')',
                   s=100,
                   marker='h',
                   zorder=3
                   )
        standard_plot_parameters(ax)

        plt.ylabel('Intensity [AU]')
        if convert_to == "Volt":
            plt.xlabel('$U$ [$V$]')

        plt.title(plot_title)

    # output vs_result
    return vs_result, vs_result_err, vs_result_r2, parameters

# %%Main Function for Correlating Calcualting Sensitivities and generating Conversion functuin

# Input vs_result and CalResult Data and Names for Calibration compounds
# fit parameters in order: y_max, center, slope
def generate_calibration_curve(cal_vs_values, cal_sens_values, cal_vs_err = None, cal_sens_err = None, plot_cal_curve=False, converted_to="Volt",fitter_used ='curve_fit', low_bounds_input = (0,-np.inf,-np.inf), up_bounds_input = (np.inf,np.inf,0)):
    # set boundary conditions 
    low_bounds = low_bounds_input
    up_bounds = up_bounds_input
    # perform Sigm fit
    # with curve fit linear regression
    if fitter_used == 'curve_fit':
        parameters, covariance = curve_fit(sigmodial, cal_vs_values, cal_sens_values, maxfev=10000, bounds= (low_bounds,up_bounds))
        # parameter errors
        parameters_errors = np.sqrt(np.diag(covariance))
        # unpack parameters
        fit_y_max, fit_center, fit_slope = parameters
        # calculate fit curve
        fit_sens_data = sigmodial(cal_vs_values, fit_y_max, fit_center, fit_slope)
        
    # with odr Orthogonal distance regression to also accoutn for errors
    elif fitter_used =='odr':
        # Create a model for ODR
        sigmodial_model = odr.Model(sigmodial_for_odr)
        # Set up the data with errors
        data = odr.RealData(cal_vs_values, cal_sens_values, sx=cal_vs_err, sy=cal_sens_err)
        # Initial parameter guesses: [y_max, center, slope]
        initial_params = [np.max(cal_sens_values), np.median(cal_vs_values), -10.0]
        # Run the ODR fitting
        odr_instance = odr.ODR(data, sigmodial_model, beta0=initial_params)
        output = odr_instance.run()
        # Get fitted parameters
        parameters = output.beta
        parameters_errors = output.sd_beta  # Standard deviation (uncertainties)
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
        ax.plot(plot_fit_vs_values, Plotfit_sens_data, label= 'sigm. fit (r$^{2}=$'+str(round(cal_curve_r2,3))+')',color =purple_dark, linewidth = 1, zorder=3)

        # use fit parameter errors to generat error corridor
        y_max_upper = parameters[0] + parameters_errors[0]
        center_upper = parameters[1] - parameters_errors[1]
        slope_upper = parameters[2] - parameters_errors[2]
        ydata_fit_upper = sigmodial(plot_fit_vs_values, y_max_upper, center_upper, slope_upper)
        plt.plot(plot_fit_vs_values, ydata_fit_upper, color = purple, linewidth = 1, zorder=3, label = "fit error")
        y_max_lower = parameters[0] - parameters_errors[0]
        center_lower = parameters[1] + parameters_errors[1]
        slope_lower = parameters[2] + parameters_errors[2]
        ydata_fit_lower = sigmodial(plot_fit_vs_values, y_max_lower, center_lower, slope_lower)
        plt.plot(plot_fit_vs_values, ydata_fit_lower, color = purple, linewidth = 1, zorder=3)
        
        standard_plot_parameters(ax)
        plt.ylabel('Sensitivity')
        if converted_to == "Volt":
            plt.xlabel('$U$ [$V$]')
        plt.rcParams['legend.fontsize'] = 10
        plt.legend(loc='best',handlelength=1)
    #output fit parameters 
    return parameters, parameters_errors, cal_curve_r2 
