import numpy as np
from numpy.typing import NDArray

def TemporalDecorrelation(wl:float,theta_inc:NDArray,ymotion:NDArray=0.0,zmotion:NDArray=0.0)->NDArray:
        
    ph_ymotion = (ymotion ** 2) * (np.sin(theta_inc) ** 2)
    ph_zmotion = (zmotion ** 2) * (np.cos(theta_inc) ** 2)

    return np.exp(-0.5 * ((4 * np.pi / wl) ** 2) * (ph_ymotion + ph_zmotion))
