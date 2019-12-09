import numpy as np
import scipy.integrate as scint
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import time
import pdf_interpolation as pdf
from scipy import LowLevelCallable
from scipy.stats import norm

'''
A few conventions ahead:
Indices for lepton species (electrons and muons must be distinguished due to different cuts):
  Electrons = 0
  Muons = 1
Set indices for quarks:
  Up type quarks = 1
  Down type quarks = 2
'''

'''
Define some global constants that should be changed consistently (in case one wants to )
'''
# Fine Structure Constant:
alpha_e = 1/128.0# PDG (Olive et al 2014) value at Q^2 = m_W^2

# SM Z mass and Width
M_Z = 91.1876 # Z boson mass in GeV
Gamma_Z = 2.4952 # Z boson decay width in GeV

# Weak Mixing Angle
sw2 = 0.23126# PDG (Olive et al 2014) value at Q^2 = m_Z^2

'''
First define auxiliary functions.
'''

def gammats2(gqv, gfv, gqa, gfa):
  return 2*(gqv**2+gqa**2)*(gfv**2+gfa**2)

def DecayWidth_full(mll, mz, gfv, gfa, mf):
  '''
  Couplings and fermion masses have to be input in the order: DM, t, b, c, s, d, u, tau, mu, e, nu_tau, nu_mu, nu_e
  e.g.:
  gfv_list = np.array([gxv, gqv, gqv, gqv, gqv, gqv, gqv, glv, glv, glv, gnv, gnv, gnv])#
  gfa_list = np.array([gxa, gqa, gqa, gqa, gqa, gqa, gqa, gla, gla, gla, gna, gna, gna])#
  mf_list = np.array([0., 173.0, 4.18, 1.275, 0.095, 0.0047, 0.0022, 1.77686, 0.1056583745, 0.0005109989461, 0., 0., 0.])#
  '''
  Gamma = 0.0
  for i in range(len(mf)):
    if mll >= (2*mf[i]):
      if (i > 0) & (i < 7):
        Gamma += 3*np.sqrt(1-(4*mf[i]**2)/(mll**2)) * ( mll**2 * (gfv[i]**2 + gfa[i]**2) + mf[i]**2 * ( 2 * gfv[i]**2 - 4 * gfa[i]**2) )
      else:
        Gamma += np.sqrt(1-(4*mf[i]**2)/(mll**2)) * ( mll**2 * (gfv[i]**2 + gfa[i]**2) + mf[i]**2 * ( 2 * gfv[i]**2 - 4 * gfa[i]**2) )
    else:
      Gamma += 0
  return (1/(12*np.pi*mz)) * Gamma

def DecayWidth(Zp_model):
#FK: Not sure about the neutrino couplings...
  gfv = np.array([Zp_model['gxv'], Zp_model['guv'], Zp_model['gdv'], Zp_model['guv'], Zp_model['gdv'], Zp_model['gdv'], Zp_model['guv'], Zp_model['glv'], Zp_model['glv'], Zp_model['glv'], 0, 0, 0])
  gfa = np.array([Zp_model['gxa'], Zp_model['gua'], Zp_model['gda'], Zp_model['gua'], Zp_model['gda'], Zp_model['gda'], Zp_model['gua'], Zp_model['gla'], Zp_model['gla'], Zp_model['gla'], 0, 0, 0])
  return DecayWidth_full(Zp_model['MZp'], Zp_model['MZp'], gfv, gfa, np.array([Zp_model['mDM'], 173.0, 4.18, 1.275, 0.095, 0.0047, 0.0022, 1.77686, 0.1056583745, 0.0005109989461, 0., 0., 0.]))

def dsigmadmll_full(mll, mz, Gamma, gqv, gfv, gqa, gfa, species = 0):
  '''
  Returns differential cross section dsigma/dmll of the Z' only signal.
  The result is returned in fb/GeV.
  Inputs: 
  mll = dilepton invariant mass in GeV, 
  mz = Z' mass in GeV), Gamma (total Z' decay width in GeV, 
  gqv = vector couplings of the Z' to quarks; must be a two-component array including coupling to up- and down-type quarks (in that order)
  glv = vector coupling of the Z' to leptons
  gqa = axial couplings of the Z' to quarks; must be a two-component array including coupling to up- and down-type quarks (in that order)
  gla = axial coupling of the Z' to leptons
  species = lepton species (0 for electrons, 1 for muons)
  '''

  # define rapidity integrals at input mll value
  T2_u = pdf.T_interp_cuts(mll, 2,  species, 0)
  T2_d = pdf.T_interp_cuts(mll, 2,  species, 1)
  
  N_c = 3 # colour factor
  return (1e12/2.56819)* 0.5*mll* (1.0/(8.0*np.pi*N_c)) * (1.0/((mll**2-mz**2)**2+(mz*Gamma)**2)) * ( ( gammats2(gqv[0], gfv, gqa[0], gfa) * T2_u ) + ( gammats2(gqv[1], gfv, gqa[1], gfa) * T2_d ) )  

def dsigmadmll(mll, Zp_model, species = "ee"):
  if species == "ee":
    sp = 0
  elif species == "mm":
    sp = 1
  else:
    print("Unknown species requested")
    return 0
  return dsigmadmll_full(mll, Zp_model['MZp'], Zp_model['Gamma'], np.array([Zp_model['guv'],Zp_model['gdv']]), Zp_model['glv'], np.array([Zp_model['gua'],Zp_model['gda']]), Zp_model['gla'], sp)

def dsigmadmll_wint_full(mll, mz, Gamma, gqv, gfv, gqa, gfa, species = 0):
  '''
  Returns differential cross section dsigma/dmll of the full BSM signal, i.e. Z' only + Z' intereference with the SM.
  The result is returned in fb/GeV.
  Inputs: 
  mll = dilepton invariant mass in GeV, 
  mz = Z' mass in GeV), Gamma (total Z' decay width in GeV, 
  gqv = vector couplings of the Z' to quarks; must be a two-component array including coupling to up- and down-type quarks (in that order)
  glv = vector coupling of the Z' to leptons
  gqa = axial couplings of the Z' to quarks; must be a two-component array including coupling to up- and down-type quarks (in that order)
  gla = axial coupling of the Z' to leptons
  species = lepton species (0 for electrons, 1 for muons)
  '''
  # define couplings of EW gauge bosons to SM fermions
  N_c = 3 # colour factor
  Q_u = 2./3. # photon coupling to up-type quarks
  Q_d = -1./3. # photon coupling to down-type quarks
  cw2 = 1-sw2 # cosine^2 of weak mixing angle
  gw2 = (4*np.pi*alpha_e)/sw2 # weak coupling constant squared
  Vl = (-0.5 + 2*sw2) # vector coupling of the Z boson to leptons
  Al = -0.5 # axial coupling of the Z boson to leptons
  V_u =  0.5- (4.0/3.0)*sw2 # vector coupling of the Z boson to up-type quarks
  V_d = -0.5+ (2.0/3.0)*sw2 # vector coupling of the Z boson to down-type quarks
  A_u =  0.5 # axial coupling of the Z boson to up-type quarks
  A_d = -0.5 # axial coupling of the Z boson to down-type quarks

  # define rapidity integrals at input mll value
  T2_u = pdf.T_interp_cuts(mll, 2, species, 0)
  T2_d = pdf.T_interp_cuts(mll, 2, species, 1)

  sum = 0.
  # add Z' only term:
  sum +=  (1.0/(8.0*np.pi*N_c)) * (1.0/((mll**2-mz**2)**2+(mz*Gamma)**2)) * ( ( gammats2(gqv[0], gfv, gqa[0], gfa) * T2_u ) + ( gammats2(gqv[1], gfv, gqa[1], gfa) * T2_d ) )

  # add Z' photon interference:
  sum += ((-alpha_e )/(N_c * mll**2)) * ((mll**2 - mz**2)/( (mll**2 - mz**2)**2 + (mz*Gamma)**2 )) * (   Q_u*( 2*gqv[0]*gfv* T2_u )  +  Q_d* ( 2*gqv[1]*gfv* T2_d )  )  

  # add Z' Z interference:
  I_u =  V_u*(gqv[0]*gfv*Vl - gqv[0]*gfa*Al) + A_u*(gqa[0]*gfa*Al - gqa[0]*gfv*Vl)

  I_d =  V_d*(gqv[1]*gfv*Vl - gqv[1]*gfa*Al) + A_d*(gqa[1]*gfa*Al - gqa[1]*gfv*Vl)

  sum += (gw2/(16*cw2*np.pi * N_c)) * ( ((mll**2-mz**2) * (mll**2 - M_Z**2) + mz*M_Z*Gamma*Gamma_Z )/( ((mll**2 - mz**2)**2 + (mz*Gamma)**2 ) * ( (mll**2 - M_Z**2)**2 + (M_Z*Gamma_Z)**2 )) ) * (    (  2*I_u*T2_u)  +   (  2*I_d*T2_d)       )

  return (1e12/2.56819)* 0.5*mll*sum


def dsigmadmll_wint(mll, Zp_model, species = "ee"):
  if species == "ee":
    sp = 0
  elif species == "mm":
    sp = 1
  else:
    print("Unknown species requested")
    return 0
  return dsigmadmll_wint_full(mll, Zp_model['MZp'], Zp_model['Gamma'], np.array([Zp_model['guv'],Zp_model['gdv']]), Zp_model['glv'], np.array([Zp_model['gua'],Zp_model['gda']]), Zp_model['gla'], sp)

