import numpy as np
from numpy.typing import NDArray

def GetReflectionCoefficient(eps_rel1:NDArray,eps_rel2:NDArray,inc:NDArray,pol:str):
    assert pol in ('s','p')

    n1 = np.sqrt(eps_rel1)
    n2 = np.sqrt(eps_rel2)

    a = n1 * np.cos(inc)
    b = n2 * np.cos(inc)
    del n1,n2

    if pol == 's':
        t =  (a - b) / (a + b)
    elif pol == 'p':
        t =  (b - a) / (a + b)

    return t
