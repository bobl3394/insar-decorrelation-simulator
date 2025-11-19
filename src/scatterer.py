import numpy as np
from numpy.typing import NDArray

pi2 = (np.pi / 2)


def _SampleDistribution(shape:tuple,name:str,**kwargs):
    key:str = name.lower()

    if key == 'constant':
        value:complex = kwargs.get('value',0.0)

        if np.iscomplex(value):
            samples = (
                np.full(shape,value.real,dtype=np.float64) + 1j * 
                np.full(shape,value.imag,dtype=np.float64))
        else:
            samples = np.full(shape,value,dtype=np.float64)
    if key == 'uniform':
        vlim:tuple[complex,complex] = kwargs.get('interval',[0.0,0.0])

        if np.iscomplex(vlim[0]) or np.iscomplex(vlim[1]):
            samples = (
                np.random.uniform(vlim[0].real,vlim[1].real,shape) + 1j * 
                np.random.uniform(vlim[0].imag,vlim[1].imag,shape))
        else:
            samples = np.random.uniform(vlim[0],vlim[1],shape)
    elif key in ('gaussian','normal'):
        mean:complex = kwargs.get('mean',0.0)
        stdd:complex = kwargs.get('stdd',0.0)

        if np.iscomplex(mean) or np.iscomplex(stdd):
            samples = (
                np.random.normal(mean.real,stdd.real,shape) + 1j * 
                np.random.normal(mean.imag,stdd.imag,shape))
        else:
            samples = np.random.normal(mean,stdd,shape)
    else:
        raise NotImplementedError(f'Unsupported distribution type "{name}".')
    
    return samples
    

def SamplePosition(size:tuple,x:dict,y:dict,z:dict)->NDArray:  
    pos = np.stack([
            _SampleDistribution(size,**x),
            _SampleDistribution(size,**y),
            _SampleDistribution(size,**z)],
        axis=2)

    return pos

def SampleOrientations(shape:tuple,**kwargs)->NDArray:
    norm = _SampleDistribution(shape,**kwargs)

    # wrap around to preserve physical sense
    norm[norm > +pi2] -= np.pi
    norm[norm < -pi2] += np.pi

    return norm

def SampleDielectricConstants(shape:tuple,**kwargs)->NDArray:
    eps = _SampleDistribution(shape,**kwargs)

    return eps

def GetScatterers(
        size:tuple,
        pos_kwargs:dict,
        norm_kwargs:dict,
        diel_kwargs:dict
    )->tuple[NDArray,NDArray]:
        
    # POSITION VECTOR
    pos = SamplePosition(size,**pos_kwargs)
    
    # ORIENTATION VECTOR
    norm = SampleOrientations(size,**norm_kwargs)

    # DIELECTRIC VECTOR   
    # samples the dielectric constanst from a Gaussian distribution 
    eps = SampleDielectricConstants(size,**diel_kwargs)

    return pos,norm,eps