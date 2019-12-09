import numpy as np

'''
Load PDF data
'''

# For each possible mll value, interpolate PDFs bilinearly in y3 and y4. bilinearInterp takes mll, y3 and y4 as arguments and returns the value of the interpolated PDF at these inputs.
# Note the difference from the original version: The interpolator does not return an interpolated function, but a single value of the interpolation function. This makes bilinearInterp compatible with numba.

T_mll_list = np.loadtxt("T_Lists/T_mll_list.dat")

T2_ATLAS_ee_u_cuts_list = np.loadtxt("T_Lists/T2_ATLAS_ee_u_cuts_list.dat")
T2_ATLAS_mm_u_cuts_list = np.loadtxt("T_Lists/T2_ATLAS_mm_u_cuts_list.dat")
T2_ATLAS_ee_d_cuts_list = np.loadtxt("T_Lists/T2_ATLAS_ee_d_cuts_list.dat")
T2_ATLAS_mm_d_cuts_list = np.loadtxt("T_Lists/T2_ATLAS_mm_d_cuts_list.dat")


def T_interp_cuts(mll, power, species, set_index = 0):
  xvalues = T_mll_list

  # set index determines, which of the PDF sums should be considered	
  if set_index == 0:
    if species == 0:
      yvalues = T2_ATLAS_ee_u_cuts_list
    elif species == 1:
      yvalues = T2_ATLAS_mm_u_cuts_list
    else:
      print('invalid input')
      return 0
		
  elif set_index == 1:
    if species == 0:
      yvalues = T2_ATLAS_ee_d_cuts_list
    elif species == 1:
      yvalues = T2_ATLAS_mm_d_cuts_list
    else:
      print('invalid input')
      return 0
		
  else:
    print('invalid input')
    return 0

  if mll < xvalues[0]:
    int_value = 0
  elif mll > xvalues[-1]:
    int_value = 0
  else:
    i = np.searchsorted(xvalues, mll)
    int_value = (yvalues[i] - yvalues[i-1]) * (mll - xvalues[i-1])/(xvalues[i] - xvalues[i-1]) + yvalues[i-1]
  return int_value
