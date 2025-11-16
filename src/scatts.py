import numpy as np
from numpy.typing import NDArray

pi2 = (np.pi / 2)

def GetScatterers(
        size:tuple,
        xlim:tuple,
        ylim:tuple,
        zlim:tuple,
        buffer:int,
        norm_kwargs:dict,
        diel_kwargs:dict
    )->tuple[NDArray,NDArray]:
        
    # POSITION VECTOR
    pos = SamplePosition(size,xlim,ylim,zlim,buffer)
    
    # ORIENTATION VECTOR
    norm = SampleOrientations(size,**norm_kwargs)

    # DIELECTRIC VECTOR   
    # samples the dielectric constanst from a Gaussian distribution 
    eps = SampleDielectricConstants(size,**diel_kwargs)

    return pos,norm,eps


def SamplePosition(size:tuple,xlim:tuple,ylim:tuple,zlim:tuple=(0.0,0.0),buffer:int=0):    
    assert buffer >= 0
    
    dims = []
    for dlim in [xlim,ylim,zlim]:
        if (dlim[0] > dlim[1]) or (buffer > 0):
            dpos = np.random.uniform(dlim[0]-buffer,dlim[1]+buffer,size=size)
        else:
            dpos = np.zeros(size,dtype=np.float64)

        dims.append(dpos)
    
    return np.stack(dims,axis=2)

def SampleOrientations(shape:tuple,distribution:str,**kwargs)->NDArray:    
    # NOTE: expresses the angle between the surface normal and the nadir in [rads]
    key:str = distribution.lower()

    if key == 'constant':
        value:float = kwargs.get('value',0.0)
        norm = np.full(shape,value,dtype=np.float64)
    elif key in ('gaussian','normal'):
        norm_mean:float = kwargs.get('mean',0.0)
        norm_stdd:float = kwargs.get('stdd',0.0)
        norm = np.random.normal(norm_mean,norm_stdd,shape)

    else:
        raise NotImplementedError(f'Unsupported distribution type "{distribution}".')

    # wrap around to preserve physical sense
    norm[norm > +pi2] -= np.pi
    norm[norm < -pi2] += np.pi

    return norm

def SampleDielectricConstants(shape:tuple,distribution:str,**kwargs)->NDArray:
    key:str = distribution.lower()

    if key == 'constant':
        eps = np.ones(shape,dtype=np.complex64)
    elif key in ('gaussian','normal'):
        diel_mean:complex = kwargs.get('mean',1.0)
        diel_stdd:complex = kwargs.get('stdd',0.0)
        eps_real = np.random.normal(diel_mean.real,diel_stdd.real,shape)
        eps_imag = np.random.normal(diel_mean.imag,diel_stdd.imag,shape)
        eps = eps_real + 1j * eps_imag
    else:
        raise NotImplementedError(f'Unsupported distribution type "{distribution}".')

    return eps