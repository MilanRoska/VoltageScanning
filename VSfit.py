# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:12:30 2024

@author: m.roska
"""

# %%packages

import numpy as np
from scipy.optimize import curve_fit
from scipy import odr
from sklearn.metrics import r2_score
from scipy.optimize import root_scalar
from scipy.optimize import newton
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from typing import Union, Optional, Tuple
import pandas as pd

# custom figure style functions for layout and colors
from FigureStyles import standard_plot_parameters, standard_colors
# import custo colors
orange, orange_light, orange_dark, orange_very_dark, purple, purple_light, purple_dark, purple_very_dark = standard_colors()


# %% Support Functions

def double_sigmoid(
    x: float,
    y_init: float,
    fall_center: float,
    fall_slope: float,
    rise_center: float,
    rise_slope: float
) -> float:
    """
    Double sigmoid decay curve function with rising and falling components.

    Used to fit a VS curve.
    - Falling slope models loss due to colission induced declustering (central VS mechanism)
    - Rising slope likely models increase due to focusing effects at high masses (secondary artifact)
    
    High precision but higher computational effort and potentially to many variables for smaller datasets (VS with to few steps)

    Args:
        x (float): Independent variable (e.g., mass or energy).
        y_init (float): Initial amplitude of the response.
        fall_center (float): Scaling factor for falling slope.
        fall_slope (float): Slope of falling edge.
        rise_center (float): Scaling factor for rising slope.
        rise_slope (float): Slope of rising edge.

    Returns:
        float: Output y-value from the double sigmoid model.
    """
    denom = (1 + fall_center * np.exp(fall_slope * x)) * (1 + rise_center / np.exp(rise_slope * x))
    y = y_init / denom
    return y


def double_sigmoid_err(
    x: float,
    y_init: float,
    fall_center: float,
    fall_slope: float,
    rise_center: float,
    rise_slope: float,
    y_init_err: float,
    fall_center_err: float,
    fall_slope_err: float,
    rise_center_err: float,
    rise_slope_err: float
) -> float:
    """
    Error propagation for the double sigmoid function using partial derivatives.

    Args:
        x (float): Independent variable
        y_init (float): Initial amplitude of the response
        fall_center (float): Fall center parameter
        fall_slope (float): Fall slope parameter
        rise_center (float): Rise center parameter
        rise_slope (float): Rise slope parameter
        y_init_err (float): Uncertainty in y_init
        fall_center_err (float): Uncertainty in fall_center
        fall_slope_err (float): Uncertainty in fall_slope
        rise_center_err (float): Uncertainty in rise_center
        rise_slope_err (float): Uncertainty in rise_slope

    Returns:
        float: Propagated uncertainty (1σ) on double_sigmoid(x)
    """
    exp_fall = np.exp(fall_slope * x)
    exp_rise = np.exp(rise_slope * x)

    denom = (1 + fall_center * exp_fall) * (1 + rise_center / exp_rise)

    term_y = y_init_err / denom

    term_fall_center = (
        -y_init * exp_fall / ((1 + rise_center / exp_rise) * (1 + fall_center * exp_fall) ** 2)
    ) * fall_center_err

    term_fall_slope = (
        -y_init * fall_center * x * exp_fall /
        ((1 + rise_center / exp_rise) * (1 + fall_center * exp_fall) ** 2)
    ) * fall_slope_err

    term_rise_center = (
        -y_init * exp_rise /
        ((1 + fall_center * exp_fall) * (rise_center + exp_rise) ** 2)
    ) * rise_center_err

    term_rise_slope = (
        y_init * rise_center * x * exp_rise /
        ((1 + fall_center * exp_fall) * (rise_center + exp_rise) ** 2)
    ) * rise_slope_err

    # Combine all terms in quadrature
    y_err = np.sqrt(term_y**2 + term_fall_center**2 + term_fall_slope**2 +
                    term_rise_center**2 + term_rise_slope**2)

    return y_err


def double_sigmoid_find_root_scalar(
    y_target: float,
    y_init: float,
    fall_center: float,
    fall_slope: float,
    rise_center: float,
    rise_slope: float,
    x_guess: float = 40.0
) -> float:
    """
    Estimates the x-value such that `double_sigmoid(x)` equals a given y_target.
    Used to get the Voltage (y) value at which half of the signal of a VS curve is remaining.

    This function numerically inverts a non-analytically invertible double sigmoid function using:
    - Bracketing method (preferred for safety)
    - Newton-Raphson fallback if bracketing fails

    Args:
        y_target (float): Target y-value for which to find the corresponding x.
        y_init (float): Maximum y-value of the sigmoid.
        fall_center (float): Center of the falling slope.
        fall_slope (float): Slope of the falling edge.
        rise_center (float): Center of the rising slope.
        rise_slope (float): Slope of the rising edge.
        x_guess (float, optional): Initial guess for x (used in Newton's method). Defaults to 40.

    Returns:
        float: x-value such that double_sigmoid(x) ≈ y_target

    Raises:
        RuntimeError: If neither root-finding method converges.
    """

    def objective_function(x: float) -> float:
        return double_sigmoid(x, y_init, fall_center, fall_slope, rise_center, rise_slope) - y_target

    try:
        # Try bracketing within [0, 300]
        f_a, f_b = objective_function(0), objective_function(300)
        if np.sign(f_a) != np.sign(f_b):
            result = root_scalar(objective_function, bracket=[0, 300], x0=x_guess)
            if result.converged:
                return result.root
    except Exception:
        pass  # Fall through to Newton

    try:
        return newton(objective_function, x_guess)
    except Exception as e:
        raise RuntimeError(f"Failed to find x for y_target={y_target}: {e}")


def gauss_amp(x: float, y_0: float, x_center: float, w: float, area: float) -> float:
    """
    Gaussian peak with specified area.
    Used to fit a VS curve.
    Simplified fitting with less precision but computationally faster since it is ivnertible.

    Args:
        x (float): Input value.
        y_0 (float): Baseline offset.
        x_center (float): Center of the Gaussian peak.
        w (float): Standard deviation (width).
        area (float): Peak area (scales the amplitude).

    Returns:
        float: Computed y-value.
    """
    return y_0 + area * np.exp(-((x - x_center) ** 2) / (2 * w ** 2))


def gauss_amp_inv(y: float, y_0: float, x_center: float, w: float, area: float) -> float:
    """
    Inverse of the Gaussian (positive root only).
    Used to get the Voltage (y) value at which half of the signal of a VS curve is remaining.

    Args:
        y (float): Output value (must be > y_0).
        y_0 (float): Baseline offset.
        x_center (float): Center of the Gaussian peak.
        w (float): Width (standard deviation).
        area (float): Peak area.

    Returns:
        float: Estimated x-value (positive root).
    """
    if y <= y_0:
        raise ValueError("y must be greater than y_0 for inversion to be valid.")
    return np.sqrt(-2 * w ** 2 * np.log((y - y_0) / area)) + x_center


def sigmodial(x: float, y_max: float, center: float, slope: float) -> float:
    """
    Sigmoid function.
    Used for correlating conventionally calibrated sensitivities to their corresponding VS results

    Args:
        x (float): Input value, VS result (usually dV50).
        y_max (float): upper limit of sensitivity.
        center (float): Midpoint of the sigmoid.
        slope (float): Controls the steepness.

    Returns:
        float: Sigmoid output.
    """
    return y_max / (1 + np.exp((x - center) / slope))


def sigmodial_err_prop(
    x: float,
    y_max: float,
    center: float,
    slope: float,
    x_err: float = 0.0,
    y_max_err: float = 0.0,
    center_err: float = 0.0,
    slope_err: float = 0.0
) -> float:
    """
    Error propagation for the sigmoidal function using partial derivatives.

    The sigmoidal function is defined as:
        y = y_max / (1 + exp((x - center) / slope))

    Args:
        x (float): Independent variable.
        y_max (float): upper limit of sensitivity.
        center (float): Midpoint of the sigmoid.
        slope (float): Controls the steepness of the curve.
        x_err (float, optional): Uncertainty in x. Defaults to 0.
        y_max_err (float, optional): Uncertainty in y_max. Defaults to 0.
        center_err (float, optional): Uncertainty in center. Defaults to 0.
        slope_err (float, optional): Uncertainty in slope. Defaults to 0.

    Returns:
        float: Propagated uncertainty (1σ) of the sigmoid output.
    """
    exp_term = np.exp((x - center) / slope)
    denom = (1 + exp_term)
    denom_sq = denom ** 2

    dy_dx = (y_max * exp_term) / (slope * denom_sq)
    dy_dcenter = (y_max * exp_term) / (slope * denom_sq)
    dy_dslope = (y_max * (x - center) * exp_term) / (slope ** 2 * denom_sq)
    dy_dymax = 1 / denom

    y_err = np.sqrt(
        (dy_dx * x_err) ** 2 +
        (dy_dcenter * center_err) ** 2 +
        (dy_dslope * slope_err) ** 2 +
        (dy_dymax * y_max_err) ** 2
    )

    return y_err


def sigmodial_for_odr(B, x):
    y_max, center, slope = B
    return y_max / (1 + np.exp((x - center) / slope))


# %% main Function for VS fitting

def vs_fit(
    signal_data: Union[pd.Series, np.ndarray, list],
    timing_data: Union[pd.Series, np.ndarray, list],
    convert_to: str = "Volt",
    acquisition_freq: float = 1,
    tunning_freq: Optional[float] = None,
    step_size: float = 2,
    max_voltage: float = 100,
    init_fit_parameters: Optional[Tuple[float, ...]] = None,
    fit_used: str = "double_sigmoid",
    max_fit_iterations: int = 5000,
    low_bounds: Tuple[float, float, float, float, float] = (0, 0, 0, 0, 0),
    up_bounds: Tuple[float, float, float, float, float] = (np.inf, np.inf, np.inf, np.inf, np.inf),
    plot_vs: bool = False,
    plot_title: str = "formula not specified",
    propagate_Error: bool = False
) -> Tuple[float, float, float, Tuple[float, ...]]:
    """
    Fit voltage scan (VS) signal using specified curve (default: double sigmoid).

    Args:
        signal_data (Series/array/list): 1D signal intensity data. Any unit is fine, since it will be nomalized anyways. All values should be positive
        timing_data (Series/array/list): 1D time or voltage scan data. This should be in seconds after start of the VS. The Voltage values will be calcualted based on the provided VS parameters. A direct input of Votlage values is planned for future updates.
        convert_to (str, optional): Whether to convert to 'Volt' or other units. Defaults to "Volt". Alternative converstion "Ekin" is not yet implemented and not recomended and although common in literature, not recomended by us currelty.
        acquisition_freq (float, optional): Frequency of signal acquisition (Hz). Defaults to 1.
        tunning_freq (float, optional): Optional tuning frequency (Hz). Defaults to None. If None is selected tunning_freq is assumed to be the same as the acquisition_freq
        step_size (float, optional): Step size in volts per tunning step. Defaults to 2V.
        max_voltage (float, optional): Max votlage value to which the scan is tuned up. Defaults to 100.
        init_fit_parameters (tuple, optional): Initial guess for fit parameters. Defaults to None.
        fit_used (str, optional): Name of fit model to use ('double_sigmoid', etc.). Defaults to "double_sigmoid" for best fitting, "Gauss" for faster computation but less optimized, "closest_datapoint" for approach wehre data is not fitted but smoothed with sav. golay. algorithm and then the closest datapoint to 50% is picked.
        max_fit_iterations (int, optional): Max iterations allowed in fit. Defaults to 5000.
        low_bounds (tuple, optional): Lower bounds for fit parameters. Defaults to (0, 0, 0, 0, 0).
        up_bounds (tuple, optional): Upper bounds for fit parameters. Defaults to (inf, inf, inf, inf, inf). Only implemented for "double_sigmoid", other fits will overwrite
        plot_vs (bool, optional): Whether to plot the fit result. Defaults to False.
        plot_title (str, optional): Title to use for plot. Defaults to "formula not specified".
        propagate_Error (bool, optional): Whether to propagate fit uncertainty. Defaults to False.

    Returns:
    Tuple[float, float, float, Tuple[float, ...]]: A 4-element tuple containing:
        - vs_result (float): Computed voltage scan response. The value at which half of the initial signal is remaining usually in V (dV50).
        - vs_result_err (float): Propagated error of the fit result.
        - vs_result_r2 (float): R² of the fit.
        - parameters (tuple): Fitted parameters.
    """

    # Set acquisition_freq value for tunning_freq if not specified otherwise
    if tunning_freq is None:
        tunning_freq = acquisition_freq
    # failsafe for when tuning was faster then acquisition. This should be avoided
    elif tunning_freq > acquisition_freq:
        raise ValueError("VS Tunning Frequency faster than Acquisition Frequency.")

    # chek if timing and Signal Data have same dimension -> error
    if signal_data.shape != timing_data.shape:
        raise ValueError("Signal and timing Data must have the same dimension.")

    # check if any values are negative or NaN -> error
    if any(signal_data < 0) or any(np.isnan(signal_data)):
        raise ValueError("Signal Data contains negative values or NaN.")
    if any(np.isnan(timing_data)):
        raise ValueError("Timing Data contains NaN values.")

    # calcualte correction factor for if slower tunning than acquisition
    slowness_correction_factor = tunning_freq / acquisition_freq

    # check if Nr of Datapoints matches Expected from VS param -> error
    expected_data_points = max_voltage / step_size / slowness_correction_factor
    if expected_data_points != timing_data.shape[0]:
        print(f"More Data Points than expected from VS setting. {expected_data_points} expected, {timing_data.shape[0]} found")

    # set timing Data to Start at 0
    timing_data = timing_data - min(timing_data)

    # convert time (x) to Volt or Ekin
    valid_convert_to_values = ["Volt", "Ekin"]
    if convert_to == "Volt":
        # Code for "Volt" conversion
        timing_dataConverted = (timing_data * step_size * slowness_correction_factor * tunning_freq)
    elif convert_to == "Ekin":
        # Code for "Ekin" conversion
        raise NotImplementedError("Not yet implemented in package.")
    else:
        raise ValueError(f"Invalid value for convert_to: {convert_to}. Choose from {valid_convert_to_values}")

    # Normalize Signal (y) to start value value
    signal_data_normalized = signal_data / signal_data[0]

    # fit data
    try:
        # fit Voltage scanning curve
        valid_fit_used_values = ["double_sigmoid", "Gauss", "closest_datapoint"]

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
            fit_y_init, fit_fall_center, fit_fall_slope, fit_rise_center, fit_rise_slope = parameters
            # calculate fit curve
            fit_signal_data = double_sigmoid(timing_dataConverted,
                                             fit_y_init,
                                             fit_fall_center,
                                             fit_fall_slope,
                                             fit_rise_center,
                                             fit_rise_slope
                                             )
            # r-squared value between data and fit curve
            r2 = r2_score(signal_data_normalized, fit_signal_data)
            # get start value from signal_data
            int_start = double_sigmoid(0,
                                       fit_y_init,
                                       fit_fall_center,
                                       fit_fall_slope,
                                       fit_rise_center,
                                       fit_rise_slope
                                       )
            # get Intensity at at half of int_start
            int_half = int_start / 2
            # Perform the optimization to find the best x value
            volt_half = double_sigmoid_find_root_scalar(int_half,
                                                        fit_y_init,
                                                        fit_fall_center,
                                                        fit_fall_slope,
                                                        fit_rise_center,
                                                        fit_rise_slope
                                                        )
            # calcualte an error if set active
            if propagate_Error:
                # unpack parameter errors
                fit_y_init_err, fit_fall_center_err, fit_fall_slope_err, fit_rise_center_err, fit_rise_slope_err = parameters_errors
                # calcualte int_half error for volt_half
                # calcuale upper volt error and lower volt error and take the maximum of both
                int_half_err = double_sigmoid_err(volt_half,
                                                  fit_y_init,
                                                  fit_fall_center,
                                                  fit_fall_slope,
                                                  fit_rise_center,
                                                  fit_rise_slope,
                                                  fit_y_init_err,
                                                  fit_fall_center_err,
                                                  fit_fall_slope_err,
                                                  fit_rise_center_err,
                                                  fit_rise_slope_err
                                                  )
                volt_half_err_upper = double_sigmoid_find_root_scalar(int_half+int_half_err,
                                                                      fit_y_init,
                                                                      fit_fall_center,
                                                                      fit_fall_slope,
                                                                      fit_rise_center,
                                                                      fit_rise_slope
                                                                      )
                volt_half_err_lower = double_sigmoid_find_root_scalar(int_half-int_half_err,
                                                                      fit_y_init,
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
            int_half = int_start / 2
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

        elif fit_used == "closest_datapoint":
            # smoothen data
            # set savgol window to 40% of length
            savgol_window = round(0.4 * len(signal_data_normalized), 0)
            # apply savgol
            signal_data_normalized_smooth = savgol_filter(signal_data_normalized, 
                                                          savgol_window,
                                                          2, mode='nearest')
            # get start value from signal_data
            int_start = signal_data_normalized_smooth[0]
            # get Intensity at at half of int_start
            int_half = int_start / 2
            # Find the index where the difference is minimal
            closest_idx = np.argmin(np.abs(signal_data_normalized_smooth - int_half))
            # Use that index to get the corresponding timing_dataConverted value
            ekin_half = timing_dataConverted[closest_idx]
            # set fit signal to be the same as signal itself, sicne we are not fitting
            fit_signal_data = signal_data_normalized_smooth

            vs_result = ekin_half
            vs_result_err = None
            vs_result_r2 = -999
            parameters = [-999, -999, -999, -999, -999]
 
        else:
            raise ValueError(f"Invalid value for fit_used: {fit_used}. Choose from {valid_fit_used_values}")

    except (RuntimeError, RuntimeWarning, ValueError) as e:
        raise type(e)(f"fit not possible: {str(e)}").with_traceback(e.__traceback__)

    # plot VS if activated
    if plot_vs:
        fig, ax = plt.subplots(figsize=(5.5, 4.125))
        # datapoints as scatter
        ax.scatter(timing_dataConverted,
                   signal_data_normalized,
                   label='Data',
                   color=orange,
                   s=50,
                   zorder=1
                   )
        # vs curve as fit
        ax.plot(timing_dataConverted,
                fit_signal_data,
                label='fit. (r$^{2}=$'+str(round(vs_result_r2, 3))+')',
                color=purple_very_dark,
                linewidth=3,
                zorder=2
                )
        # init signal values as purple hex
        ax.scatter(0,
                   int_start,
                   color=purple_very_dark,
                   label='100%',
                   s=100,
                   marker='H',
                   zorder=3
                   )
        # vs results as pruple hex
        ax.scatter(vs_result,
                   int_half,
                   color=purple_very_dark,
                   label='50% ('+convert_to+'='+str(round(vs_result, 2))+')',
                   s=100,
                   marker='h',
                   zorder=3
                   )
        # ploting styles
        standard_plot_parameters(ax)
        # axis labels and title
        plt.ylabel('Intensity [AU]')
        if convert_to == "Volt":
            plt.xlabel('$U$ [$V$]')
        plt.title(plot_title)

    # output vs_result
    return vs_result, vs_result_err, vs_result_r2, parameters


# %% Main Function for Correlating conventionally derived Sensitivities to voltage scanning results and generating a Correlation function

def generate_calibration_curve(
    cal_vs_values: Union[pd.Series, np.ndarray, list],
    cal_sens_values: Union[pd.Series, np.ndarray, list],
    cal_vs_err: Optional[Union[pd.Series, np.ndarray, list]] = None,
    cal_sens_err: Optional[Union[pd.Series, np.ndarray, list]] = None,
    plot_cal_curve: bool = True,
    converted_to: str = "Volt",
    fitter_used: str = 'curve_fit',
    low_bounds_input: Tuple[float, float, float] = (0, -np.inf, -np.inf),
    up_bounds_input: Tuple[float, float, float] = (np.inf, np.inf, 0)
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit VS correlation curve using sigmoid function via curve_fit or ODR.
    The function fits the covnentionally derived sensitivities against the VS results for corresponding VS measurements.

    Args:
        cal_vs_values (np.ndarray): Input VS values (e.g., dV50).
        cal_sens_values (np.ndarray): Corresponding sensitivity values from conventional calibration (unit of choice).
        cal_vs_err (np.ndarray, optional): Errors in VS values (used in ODR).
        cal_sens_err (np.ndarray, optional): Errors in sensitivity (used in ODR).
        plot_cal_curve (bool, optional): Whether to generate a plot of the calibration curve. Defaults to True.
        converted_to (str, optional): Axis label unit (e.g. 'Volt'). Defaults to "Volt".
        fitter_used (str, optional): 'curve_fit' or 'odr'. Defaults to 'curve_fit'.
        low_bounds_input (tuple): Lower bounds for fitting parameters. Defaults to (0, -inf, -inf).
        up_bounds_input (tuple): Upper bounds for fitting parameters. Defaults to (inf, inf, 0).

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: 
            - parameters: Best fit parameters (y_max, center, slope)
            - parameters_errors: 1σ uncertainties on fit parameters
            - cal_curve_r2: R² of the fit
    """

    # Fit the data
    if fitter_used == 'curve_fit':
        parameters, covariance = curve_fit(
            sigmodial, cal_vs_values, cal_sens_values,
            bounds=(low_bounds_input, up_bounds_input), maxfev=10000
        )
        parameters_errors = np.sqrt(np.diag(covariance))
    elif fitter_used == 'odr':
        model = odr.Model(sigmodial_for_odr)
        data = odr.RealData(cal_vs_values, cal_sens_values, sx=cal_vs_err, sy=cal_sens_err)
        initial_params = [np.max(cal_sens_values), np.median(cal_vs_values), -10.0]
        odr_fit = odr.ODR(data, model, beta0=initial_params)
        output = odr_fit.run()
        parameters = output.beta
        parameters_errors = output.sd_beta
    else:
        raise ValueError(f"Unsupported fitter method: {fitter_used}")

    # Unpack for readability
    fit_y_max, fit_center, fit_slope = parameters

    # Compute fitted sensitivity values
    fit_sens_data = sigmodial(cal_vs_values, fit_y_max, fit_center, fit_slope)
    cal_curve_r2 = r2_score(cal_sens_values, fit_sens_data)

    # Plot if needed
    if plot_cal_curve:
        x_fit = np.linspace(0, np.max(cal_vs_values) + 1.1, 1000)
        y_fit = sigmodial(x_fit, *parameters)

        fig, ax = plt.subplots(figsize=(5.5, 3.44))
        ax.scatter(cal_vs_values, cal_sens_values, c=orange, edgecolor='none', s=60, zorder=2, label='Cal. compounds')
        ax.plot(x_fit, y_fit, label=f'Sigm. fit (R² = {cal_curve_r2:.3f})', color=purple_dark, linewidth=1)

        standard_plot_parameters(ax)
        ax.set_ylabel('Sensitivity')
        ax.set_xlabel('$U$ [$V$]' if converted_to == "Volt" else converted_to)
        ax.legend(loc='best', handlelength=1)
        plt.tight_layout()

    return parameters, parameters_errors, cal_curve_r2
