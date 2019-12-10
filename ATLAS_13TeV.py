from __future__ import division
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
from math import erf

analysis_name = 'ATLAS_13TeV'
lumi = 139.           # in fb^1-

ee_data = np.loadtxt(analysis_name+'/ee_data.dat',delimiter='\t')
mm_data = np.loadtxt(analysis_name+'/mm_data.dat',delimiter='\t')

ee_bin_low = ee_data[:,0]
ee_bin_high = ee_data[:,1]
ee_observed = ee_data[:,2]
ee_expected = ee_data[:,3]

mm_bin_low = mm_data[:,0]
mm_bin_high = mm_data[:,1]
mm_observed = mm_data[:,2]
mm_expected = mm_data[:,3]

ee_resolution_data = np.loadtxt(analysis_name+'/ee_resolutions.dat')
mm_resolution_data = np.loadtxt(analysis_name+'/mm_resolutions.dat')

ee_res_x = ee_resolution_data[:,0]
ee_res_y = ee_resolution_data[:,1]

mm_res_x = mm_resolution_data[:,0]
mm_res_y = mm_resolution_data[:,1]

# Returns the resolution for given dilepton invariant mass
def ee_resolution(Mll):
  if Mll < ee_res_x[0]: return ee_res_y[0]
  if Mll > ee_res_x[-1]: return ee_res_y[-1]
  i = np.searchsorted(ee_res_x, Mll)
  return (ee_res_y[i] - ee_res_y[i-1]) * (Mll - ee_res_x[i-1])/(ee_res_x[i] - ee_res_x[i-1]) + ee_res_y[i-1]

def mm_resolution(Mll):
  if Mll < mm_res_x[0]: return mm_res_y[0]
  if Mll > mm_res_x[-1]: return mm_res_y[-1]
  i = np.searchsorted(mm_res_x, Mll)
  return (mm_res_y[i] - mm_res_y[i-1]) * (Mll - mm_res_x[i-1])/(mm_res_x[i] - mm_res_x[i-1]) + mm_res_y[i-1]

# Returns the probability that a signal at mll ends up contributing to the bin [mll_low, mll_high] due to resolution effects
def ee_response_function(mll, mll_low, mll_high):
  return (erf((mll_high - mll)/(np.sqrt(2)*ee_resolution(mll))) - erf((mll_low - mll)/(np.sqrt(2)*ee_resolution(mll))))/2.

def mm_response_function(mll, mll_low, mll_high):
  return (erf((mll_high - mll)/(np.sqrt(2)*mm_resolution(mll))) - erf((mll_low - mll)/(np.sqrt(2)*mm_resolution(mll))))/2

# For a given input value of mll this function finds the smallest value of mll for which the probability of upward fluctuation above the input value is greater than 0.135% (3 sigma)
# This function is used to set the lower bound of the convolution integral
def ee_upward_fluctuation(mll):
  if mll < ee_resolution_data[0,0] or mll > ee_resolution_data[-1,0]:
    print('Warning! Energy resolution undefined for mll = ',mll)
    #exit ()
  f = lambda x: x + 3*ee_resolution(x) - mll
  if f(ee_resolution_data[0,0]) >= 0:
    return ee_resolution_data[0,0]
  else:
    return optimize.brentq(f, ee_resolution_data[0,0], mll)

def mm_upward_fluctuation(mll):
  if mll < mm_resolution_data[0,0] or mll > mm_resolution_data[-1,0]:
    print('Warning! Muon energy resolution undefined for mll = ',mll)
    #exit ()
  f = lambda x: x + 3*mm_resolution(x) - mll
  if f(mm_resolution_data[0,0]) >= 0:
    return mm_resolution_data[0,0]
  else:
    return optimize.brentq(f, mm_resolution_data[0,0], mll)

# For a given input value of mll this function finds the largest value of mll for which the probability of downward fluctuation below the input value is greater than 0.135% (3 sigma)
# This function is used to set the upper bound of the convolution integral
def ee_downward_fluctuation(mll):
  if mll < ee_resolution_data[0,0] or mll > ee_resolution_data[-1,0]:
    print('Energy resolution undefined for mll = ',mll)
    #exit ()
  f = lambda x: x - 3*ee_resolution(x) - mll
  if f(ee_resolution_data[-1,0]) <= 0:
    return ee_resolution_data[-1,0]
  else:
    return optimize.brentq(f, mll, ee_resolution_data[-1,0])

def mm_downward_fluctuation(mll):
  if mll < mm_resolution_data[0,0] or mll > mm_resolution_data[-1,0]:
    print('Energy resolution undefined for mll = ',mll)
    #exit ()
  f = lambda x: x - 3*mm_resolution(x) - mll
  if f(mm_resolution_data[-1,0]) <= 0:
    return mm_resolution_data[-1,0]
  else:
    return optimize.brentq(f, mll, mm_resolution_data[-1,0])

# Returns -2 log L, where L is the Poisson likelihood to observe n_observed events for expectation value n_predicted
def likelihood(n_predicted, n_observed):
  if n_predicted <= 0:
    return 10000
  if n_observed > 0:
    return 2. * ((n_predicted - n_observed) + n_observed * np.log(n_observed / n_predicted))
  else:
    return 2. * n_predicted

# Input:
# ee_signal: Callable function of one variable (mll) that returns the expected differential cross section in the di-electron channel
# mm_signal: Callable function of one variable (mll) that returns the expected differential cross section in the di-muon channel
# signal_range: Range of mll to be included in the analysis *after* convolution with energy resolution
# Note that ee_signal and mm_signal will also be evaluated outside of signal_range in order to perform the convolution integral
# Output:
# chi2: Callable function of one variable (mu) that returns the chi2 test statistic as function of the signal strength mu
# chi2_Asimov: Same as chi2 but for the Asimove data set rather than the actually observed data
def calculate_chi2(ee_signal, mm_signal, signal_range = [0,5000]):

  # Identify bins that cover the requested signal range
  i_low = 0 
  while ee_bin_low[i_low+1] < signal_range[0] and i_low < len(ee_bin_low)-2: i_low = i_low + 1

  i_high = 0
  while ee_bin_high[i_high] < signal_range[1] and i_high < len(ee_bin_high)-1: i_high = i_high + 1

  ee_integrand = lambda x, mll_low, mll_high: lumi * ee_signal(x) * ee_response_function(x, mll_low, mll_high)
  mm_integrand = lambda x, mll_low, mll_high: lumi * mm_signal(x) * mm_response_function(x, mll_low, mll_high)

  # Calculate the signal expectation in the signal range
  ee_bincounts = np.array([integrate.quad(ee_integrand, ee_upward_fluctuation(ee_bin_low[i]), ee_downward_fluctuation(ee_bin_high[i]), args=(ee_bin_low[i], ee_bin_high[i]), epsabs=1e-30, epsrel = 0.01)[0] for i in range(i_low, i_high + 1)])

  mm_bincounts = np.array([integrate.quad(mm_integrand, mm_upward_fluctuation(mm_bin_low[i]), mm_downward_fluctuation(mm_bin_high[i]), args=(mm_bin_low[i], mm_bin_high[i]), epsabs=1e-30, epsrel = 0.01)[0] for i in range(i_low, i_high + 1)])

  # Calculate the weights for bins at the edge of the signal region
  ee_weight = np.ones(i_high-i_low +1)
  mm_weight = np.ones(i_high-i_low +1)
  
  if ee_bincounts[0] != 0:
    ee_weight[0] = integrate.quad(ee_integrand, ee_upward_fluctuation(signal_range[0]), ee_downward_fluctuation(ee_bin_high[i_low]), args=(signal_range[0], ee_bin_high[i_low]), epsabs=1e-30, full_output = 1)[0] / ee_bincounts[0]
    ee_weight[0] = min(np.abs(ee_weight[0]),1)
  else :
    ee_weight[0] = 0
  if ee_bincounts[i_high-i_low] != 0:
    ee_weight[i_high-i_low] = integrate.quad(ee_integrand, ee_upward_fluctuation(ee_bin_low[i_high]), ee_downward_fluctuation(signal_range[1]), args=(ee_bin_low[i_high], signal_range[1]), epsabs=1e-30, full_output = 1)[0] / ee_bincounts[i_high-i_low]
    ee_weight[i_high-i_low] = min(np.abs(ee_weight[i_high-i_low]),1)
  else: 
    ee_weight[i_high-i_low] = 0

  if mm_bincounts[0] != 0:
    mm_weight[0] = integrate.quad(mm_integrand, mm_upward_fluctuation(signal_range[0]), mm_downward_fluctuation(mm_bin_high[i_low]), args=(signal_range[0], mm_bin_high[i_low]), epsabs=1e-30, full_output = 1)[0] / mm_bincounts[0]
    mm_weight[0] = min(np.abs(mm_weight[0]),1)
  else:
    mm_weight[0] = 0
  if mm_bincounts[i_high-i_low]   != 0:
    mm_weight[i_high-i_low] = integrate.quad(mm_integrand, mm_upward_fluctuation(mm_bin_low[i_high]), mm_downward_fluctuation(signal_range[1]), args=(mm_bin_low[i_high], signal_range[1]), epsabs=1e-30, full_output = 1)[0] / mm_bincounts[i_high-i_low]  
    mm_weight[i_high-i_low] = min(np.abs(mm_weight[i_high-i_low]),1)
  else:
    mm_weight[i_high-i_low] = 0

  # Define chi2 functions
  def chi2(mu):
        
    chi2_ee = np.sum([ee_weight[i - i_low] * likelihood(ee_expected[i] + mu * ee_bincounts[i - i_low], ee_observed[i]) for i in range(i_low, i_high + 1)]) 

    chi2_mm = np.sum([mm_weight[i - i_low] * likelihood(mm_expected[i] + mu * mm_bincounts[i - i_low], mm_observed[i]) for i in range(i_low, i_high + 1)]) 

    return chi2_ee + chi2_mm

  def chi2_Asimov(mu):
        
    chi2_ee = np.sum([ee_weight[i - i_low] * likelihood(ee_expected[i] + mu * ee_bincounts[i - i_low], ee_expected[i]) for i in range(i_low, i_high + 1)]) 

    chi2_mm = np.sum([mm_weight[i - i_low] * likelihood(mm_expected[i] + mu * mm_bincounts[i - i_low], mm_expected[i]) for i in range(i_low, i_high + 1)]) 

    return chi2_ee + chi2_mm

  return chi2, chi2_Asimov
