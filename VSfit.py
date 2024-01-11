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
orangelight = (252/255 , 230/255 , 207/255 , 100/100)
orangedark = (145/255 , 77/255 , 8/255 , 100/100)
orangeverydark = (73/255 , 39/255 , 4/255 , 100/100)
purple = (215/255 , 61/255 , 245/255 , 100/100)
purplelight = (252/255 , 240/255 , 254/255 , 100/100)
purpledark = (122/255 , 8/255 , 145/255 , 100/100)
purpleverydark = (61/255 , 4/255 , 73/255 , 100/100)

#%%Support Fnctions
def standardPlotParameters(ax):
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

def DoubleSigm (x,ymax,A,a,B,b):
    y = (ymax)/((1+(A*np.exp(a*x)))*(1+(B/np.exp(b*x))))
    return y

def DoubleSigmInvResidual(y_target, ymax, A, a, B, b, x_guess=50):
    # Define the objective function to find the root of
    objective_function = lambda x: DoubleSigm(x, ymax, A, a, B, b) - y_target

    # Find the root using root_scalar
    result = root_scalar(objective_function, bracket=[0, 200], x0=x_guess)

    if result.converged:
        return result.root
    else:
        raise RuntimeError("Failed to find x for the given y.")

def GaussAmp(x,y0,xc,w,A):
    y = y0+A*np.exp((-((x-xc)**2))/(2*w**2))
    return y

def GaussAmpInv(y,y0,xc,w,A):
    x = np.sqrt(-2*w**2*np.log((y-y0)/A))+xc
    return x

#%%Main Function for VS fitting

def VSfit(SignalData, TimingData, ConvertTo="U2", AcquisitionFreq=10, TunningFreq=None, StepSize=2, Range=100, FitUsed="DoubleSigm", maxFitIterations=5000, plotVS=False):
    
    # Set default value for TunningFreq
    if TunningFreq is None:
        TunningFreq = AcquisitionFreq
    elif TunningFreq > AcquisitionFreq:
        raise ValueError("VS Tunning Frequency faster than Acquisition Frequency.")
    else:
        pass
    
    #chek if timing and Signal Data have same dimension -> error
    if SignalData.shape != TimingData.shape:
        raise ValueError("Signal and timing Data must have the same dimension.")
        
    #check if any values are negative or NaN -> error
    if any(SignalData < 0) or any(np.isnan(SignalData)):
        raise ValueError("Signal Data contains negative values or NaN.")
    if any(np.isnan(TimingData)):
        raise ValueError("Timing Data contains NaN values.")
    
    #calcualte correction factor for if slower tunning than acquisition
    SlownessCorrectionFactor = TunningFreq/AcquisitionFreq
    
    #check if Nr of Datapoints matches Expected from VS param -> error
    ExpectedDataPoints = Range/StepSize/SlownessCorrectionFactor
    if ExpectedDataPoints != TimingData.shape[0]:
        raise ValueError(f"More Data Points than expected from VS setting. {ExpectedDataPoints} expected, {TimingData.shape[0]} found")
        
    #set timing Data to Start at 0
    TimingData = TimingData-min(TimingData)
    
    #convert time (x) to U2 or Ekin
    valid_ConvertTo_values = ["U2", "Ekin"]
    if ConvertTo == "U2":
    # Code for "U2" conversion
     TimingDataConverted = (TimingData*StepSize*SlownessCorrectionFactor)**2
    elif ConvertTo == "Ekin":
    # Code for "Ekin" conversion
        raise NotImplementedError("Not yet implemented in package.")
    else:
        raise ValueError(f"Invalid value for ConvertTo: {ConvertTo}. Choose from {valid_ConvertTo_values}")
    
    #Normalize Signal (y) to highest value
    SignalDataNormalized = SignalData/max(SignalData)
    
    try:
        #fit TS
        valid_FitUsed_values = ["DoubleSigm", "Gauss"]
        if FitUsed == "DoubleSigm":
            #set boundary conditions 
            lowBounds = (0,0,0,0,0)
            upBounds = (np.inf,np.inf,np.inf,np.inf,np.inf)
            #perform DoubleSigm fit
            parameters, covariance = curve_fit(DoubleSigm, TimingDataConverted, SignalDataNormalized, maxfev=maxFitIterations, bounds= (lowBounds,upBounds))
            #unpack parameters
            fit_ymax, fit_A, fit_a, fit_B, fit_b = parameters  
            #calculate fit curve
            fitSignalData = DoubleSigm(TimingDataConverted,fit_ymax,fit_A,fit_a,fit_B,fit_b)
            #r2 value
            #r-squared value between data and fit curve
            r2 = r2_score(SignalDataNormalized,fitSignalData)
            #get start value from SignalData
            IntStart = DoubleSigm(0,fit_ymax,fit_A,fit_a,fit_B,fit_b) 
            #get Intensity at at half of IntStart
            IntHalf = IntStart/2
            
            
            #get U2Half from IntHalf value using minimized scalar residual method
            # Perform the optimization to find the best x value
            U2Half = DoubleSigmInvResidual(IntHalf,fit_ymax,fit_A,fit_a,fit_B,fit_b)
            VSResult = U2Half
            VSResultR2 = r2
        elif FitUsed == "Gauss":   
            #set boundary conditions 
            lowBounds = (-np.inf,-np.inf,-np.inf,-np.inf)
            upBounds = (np.inf,np.inf,np.inf,np.inf)
            #perform DoubleSigm fit
            parameters, covariance = curve_fit(GaussAmp, TimingDataConverted, SignalDataNormalized, maxfev=maxFitIterations, bounds= (lowBounds,upBounds))
            #unpack parameters
            fit_y0, fit_xc, fit_w, fit_A = parameters  
            #calculate fit curve
            fitSignalData = GaussAmp(TimingDataConverted, fit_y0, fit_xc, fit_w, fit_A)  
            #r2 value
            #r-squared value between data and fit curve
            r2 = r2_score(SignalDataNormalized,fitSignalData)
            #get start value from SignalData
            IntStart = GaussAmp(0, fit_y0, fit_xc, fit_w, fit_A) 
            #get Intensity at at half of IntStart
            IntHalf = IntStart/2
            #get EkinHalf from IntHalf value using minimized scalar residual method
            EkinHalf = GaussAmpInv(IntHalf, fit_y0, fit_xc, fit_w, fit_A)
            VSResult = EkinHalf
            VSResultR2 = r2
        else:
            raise ValueError(f"Invalid value for FitUsed: {FitUsed}. Choose from {valid_FitUsed_values}")

    except (RuntimeError, RuntimeWarning, ValueError) as e:
        raise RuntimeError("fit not possible") from e
    
    #plot VS
    if plotVS == True:
        fig, ax = plt.subplots(figsize=(5.5,4.125))
        ax.scatter(TimingDataConverted, SignalDataNormalized, label='Data',color =orange, s=50, zorder=1)
        ax.plot(TimingDataConverted, fitSignalData, label = 'fit. (r$^{2}=$'+str(round(r2,3))+')',color =purpleverydark, linewidth = 3, zorder=2)
        ax.scatter(0,IntStart,color =purpleverydark,label='100%',s=100, marker = 'H',zorder=3)
        ax.scatter(VSResult,IntHalf,color =purpleverydark,label='50% ('+ConvertTo+'='+str(round(VSResult,2))+')',s=100, marker = 'h',zorder=3)       
        standardPlotParameters(ax)
        plt.ylabel('Intensity [AU]')
        if ConvertTo == "U2":
            plt.xlabel('$U^{2}$ [$V^{2}$]')
        
    #output VSresult
    return VSResult, VSResultR2

#%%Main Function for Correlating Calcualting Sensitivities and generating Conversion functuin

#Input VSresult and CalResult Data for Calibration compounds
#plot
#fit
#return fit parameters

    
