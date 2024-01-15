# VoltageScanning
Package to be used for calculating VS datasets from CIMS measurements. VS can be used for quantification of uncalibrated Compounds

VSfit.py contains several functions for calculating VS result data from a given Voltage Scan dataset and converting these values into sensitivities using data from calibration compounds. 
This includes sensitivities from calibration measurements, as well as VS results calculated from VS measurements of the calibration compounds.

from VSfit import 

vs_fit
Converts timing data into (default) U^2 (Voltage^2) data
fits the dataset 
calcualtes the U^2 value at 50% of the starting signal

generate_calibration_curve

generates a calibration curve from given Sensitivity and VSresult values of a set of calibration compounds
returns the fit parameters of the sigmodial fit used

sigmodial

can be used to convert VSresult data derived with vs_fit by converting them using the fit parameters derived with generate_calibration_curve