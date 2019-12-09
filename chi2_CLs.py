from __future__ import division
import numpy as np
import scipy.optimize as optimize
from scipy.stats import norm

#Makes sure no square root of a negative value is taken
def sqrt_safe(x):
  return np.sqrt(max(x,0))

#For a given function chi2(mu) this function finds the value of mu that minimizes chi2, ensuring mu >= 0
def minimum(chi2):

  chi2_min = optimize.minimize(chi2, 1, method = 'Nelder-Mead', options = {'ftol':.01, 'maxiter':100} )

  if not(chi2_min['success']):
    print('Warning: Failed to find minimal chi2')

  if chi2_min['x'][0] > 0:
    mu_min = chi2_min['x'][0]
  else:
    mu_min = 0

  return mu_min

#For given input chi2(mu) and chi2_Asimov(mu) this function returns a list of three values:
#Position 0:   chi2 (or -2 log L) of the signal hypothesis (mu = 1)
#Position 1:   Delta chi2 (or -2 Delta log L) between the signal hypothesis and the minimal chi2 with 0 < mu < 1
#              If the signal hypothesis is correct, this function should follow a chi2 distribution with 1 d.o.f.
#              under random fluctuations in the data in the asymptotic limit
#              The signal hypothesis can be excluded at 95% C.L. if Delta chi2 < 2.71 (one-sided limit)
#Position 2:   CLs value for the signal hypothesis in the asymptotic limit
#              The signal hypothesis can be excluded at 95% C.L. if CLs < 0.05
def get_likelihood(chi2, chi2_Asimov):

  muhat = minimum(chi2)

  Delta_chi2 = lambda mu: chi2(mu) - chi2(min(muhat,mu))

  CLs = lambda mu: (1 - norm.cdf(sqrt_safe(Delta_chi2(mu))))/norm.cdf(sqrt_safe(chi2_Asimov(mu)) - sqrt_safe(Delta_chi2(mu)))

  return [chi2(1),Delta_chi2(1), CLs(1)]

