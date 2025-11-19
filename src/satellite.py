import numpy as np 
from numpy.typing import NDArray

def GetPositionFromHeightAndLookAngle(height:float,look_angle:float)->NDArray:
    d = height * np.tan(look_angle)
    return np.array([[-d,0.0,height]])

