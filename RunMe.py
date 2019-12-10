###############################################
# ZPEED: Z' Exclusions from Experimental Data # 
###############################################
# By Felix Kahlhoefer and Stefan Schulte, 2019

from __future__ import division
import numpy as np
import scipy.integrate as integrate
from chi2_CLs import get_likelihood
from ATLAS_13TeV_calibration import xi_function
from ATLAS_13TeV import calculate_chi2
import dileptons_functions as df 
import time

start_time = time.clock()

#Step 1: Define model parameter point
Zp_model = {
  'MZp': 1000.,  #Zp mass
  'mDM': 100.,   #Dark matter mass

  'gxv': 1.,     #Zp-DM vector coupling
  'guv': 0.1,    #Zp-up-type-quark vector coupling
  'gdv': 0.1,    #Zp-down-type-quark vector coupling
  'glv': 0.01,   #Zp-lepton vector coupling

  'gxa': 0.,     #Zp-DM axial coupling
  'gua': 0.,     #Zp-up-type-quark axial coupling
  'gda': 0.,     #Zp-down-type-quark axial coupling
  'gla': 0.,     #Zp-lepton axial coupling
}

# The couplings to neutrinos follow from SM gauge invariance and the fact that right-handed neutrinos do not exist
Zp_model['gnv'] = 0.5 * (Zp_model['glv'] - Zp_model['gla'])
Zp_model['gna'] = 0.5 * (Zp_model['gla'] - Zp_model['glv'])

Zp_model['Gamma'] = df.DecayWidth(Zp_model)


step1_time = time.clock()

#Step 2: Calculate differential cross section (including detector efficiency)
ee_signal = lambda x : xi_function(x, "ee") * df.dsigmadmll(x, Zp_model, "ee")
mm_signal = lambda x : xi_function(x, "mm") * df.dsigmadmll(x, Zp_model, "mm")	

ee_signal_with_interference = lambda x : xi_function(x, "ee") * df.dsigmadmll_wint(x, Zp_model, "ee")
mm_signal_with_interference = lambda x : xi_function(x, "mm") * df.dsigmadmll_wint(x, Zp_model, "mm")	

step2_time = time.clock()

#Step 3: Create likelihood functions
Mlow = Zp_model['MZp'] - 3.*Zp_model['Gamma']
Mhigh = Zp_model['MZp'] + 3.*Zp_model['Gamma']

sig_range = [Mlow,Mhigh]
chi2, chi2_Asimov = calculate_chi2(ee_signal, mm_signal, signal_range=sig_range)
chi2_with_interference, chi2_Asimov_with_interference = calculate_chi2(ee_signal_with_interference, mm_signal_with_interference, signal_range=sig_range)

step3_time = time.clock()

# Step 4: Evaluate test statistic
result = get_likelihood(chi2, chi2_Asimov)
result_with_interference = get_likelihood(chi2_with_interference, chi2_Asimov_with_interference)

print("Without interference")
print("-2 log L:       ", result[0])
print("-2 Delta log L: ", result[1])
print("CLs:            ", result[2])

print("With interference")
print("-2 log L:       ", result_with_interference[0])
print("-2 Delta log L: ", result_with_interference[1])
print("CLs:            ", result_with_interference[2])

step4_time = time.clock()

print("Timing information")
print("Step 1: ", step1_time - start_time)
print("Step 2: ", step2_time - step1_time)
print("Step 3: ", step3_time - step2_time)
print("Step 4: ", step4_time - step3_time)
print("Total:  ", step4_time - start_time)

exit()

