from __future__ import division
import numpy as np

# load interpolation data
xi_ee_cuts = np.loadtxt('xi_functions/ee_cuts.dat')
xi_mm_cuts = np.loadtxt('xi_functions/mm_cuts.dat')

def xi_function(mll, ll=0):
  '''
  returns the value at mll of the rescaling function that effectively accounts for detector efficiencies and NLO effects
  inputs:
  mll = dilepton invariant mass in GeV
  ll = lepton species (0 for electrons, 1 for muons)
  '''
  if ll == "ee":
    grid = xi_ee_cuts[:,0]
    data = xi_ee_cuts[:,1]
  elif ll == "mm":
    grid = xi_mm_cuts[:,0]
    data = xi_mm_cuts[:,1]
  else:
    print('invalid input')
    return 0

  if mll < grid[0]:
    int_value = 0
  elif mll >= grid[0] and mll <= grid[1]:
    int_value = data[1]
  elif mll >= grid[-1]:
    int_value = data[-1]
  else:
    i = np.searchsorted(grid, mll)
    int_value = (data[i] - data[i-1]) * (mll - grid[i-1])/(grid[i] - grid[i-1]) + data[i-1]
  return int_value










