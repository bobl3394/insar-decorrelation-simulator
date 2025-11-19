import numpy as np 
from . import fresnel

def RunForwardPropagation(sat,pos,eps,theta_inc,theta_norm,antenna_pattern,pol,wl):
    lia = (theta_inc - theta_norm)
    lia[np.abs(lia) > antenna_pattern] = 0.0 # VERY BASIC ANTENNA PATTERN GAIN :D

    t = fresnel.GetReflectionCoefficient(1.0,eps,lia,pol)

    dist = np.linalg.norm(pos - sat,axis=-1)

    prop_phase_delay = (4 * np.pi * dist) / wl
    signals = t * np.exp(-1j * prop_phase_delay)

    return signals
