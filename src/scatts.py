import numpy as np
from numpy.typing import NDArray

def GetScatterers(
        num_looks:int,
        num_scatts:int,
        scene_shape:tuple[int,int],
        scene_buffer:int,
        slope_mean:float, 
        slope_stdd:float,
        eps_mean:complex,
        eps_stdd:complex
    )->tuple[NDArray,NDArray]:

    h2 = (scene_shape[0] // 2) + scene_buffer
    w2 = (scene_shape[1] // 2) + scene_buffer
    shape = (num_looks,num_scatts)
    
    # POSITION VECTOR
    # uniformely and indipendently samples the scene
    #
    # TODO:
    # - add support for 2D/3D case
    # - add support for predifined distribution
    # - add support for custom distribution 
    pos = np.stack([
        np.random.uniform(-h2,h2,size=shape), 
        np.random.uniform(-w2,w2,size=shape),
        np.zeros(shape,dtype=np.float64)],    
        axis=2)
    
    # ORIENTATION VECTOR
    theta_norm = np.random.normal(slope_mean,slope_stdd,shape)

    # DIELECTRIC VECTOR   
    # samples the dielectric constanst from a Gaussian distribution 
    eps_real = np.random.normal(eps_mean.real,eps_stdd.real,shape)
    eps_imag = np.random.normal(eps_mean.imag,eps_stdd.imag,shape)
    eps = eps_real + 1j * eps_imag

    return pos,theta_norm,eps