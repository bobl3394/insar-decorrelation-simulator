import matplotlib
matplotlib.use('tkagg') 
del matplotlib

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from src import fresnel,scatts

LUT_ESP_REL = {
    "dry":  3.0 - 0.1j,
    "wet": 15.0 - 2.1j
}

def GetSatellite(height,inc):
    d = height * np.tan(inc)
    return np.array([[-d,0,height]])

def RunForwardPropagation(sat,pos,eps,theta_inc,theta_norm,antenna_pattern,pol,wl):
    # - orientation elemental scatterers    
    lia = (theta_inc - theta_norm)
    lia[np.abs(lia) > antenna_pattern] = 0.0 # VERY BASIC ANTENNA PATTERN GAIN :D

    t = fresnel.GetReflectionCoefficient(1.0,eps,lia,pol)

    dist = np.linalg.norm(pos - sat,axis=-1)

    prop_phase_delay = (4 * np.pi * dist) / wl
    signals = t * np.exp(-1j * prop_phase_delay)

    return signals

def GetSceneMask(pos:NDArray,scene_shape:tuple)->NDArray:
    h2 = (scene_shape[0] // 2); w2 = (scene_shape[1] // 2)
    m_scene = (np.abs(pos[...,0]) <= h2) & (np.abs(pos[...,1]) <= w2)
    return m_scene

def ApplyRandomGaussianShifts(pos:NDArray,xmotion:float=0.0,ymotion:float=0.0,zmotion:float=0.0)->NDArray:    
    _,num_scatts,_ = pos.shape

    pos_shift = pos.copy()
    if xmotion: pos_shift[...,0] += np.random.normal(0.0,xmotion,num_scatts)
    if ymotion: pos_shift[...,1] += np.random.normal(0.0,ymotion,num_scatts)
    if zmotion: pos_shift[...,2] += np.random.normal(0.0,zmotion,num_scatts)

    return pos_shift

def ComputeSampleCoherence(pri:NDArray,sec:NDArray)->NDArray:
    num = np.sum(pri * sec.conj())
    den = np.sqrt(np.sum(np.abs(pri) ** 2)) * np.sqrt(np.sum(np.abs(sec) ** 2))

    return num / den

def SimulSampleCoherence(
        num_looks,
        num_scatts,
        scene_shape,
        scene_buffer,
        eps_mean,
        eps_stdd,
        slope_mean,
        slope_stdd,
        height,
        theta_inc,        
        antenna_pattern_rad,
        pol,
        wl,
        motion_stdd
    ):
    
    #====================================================================================#
    # COMPUTE SATELLITE POSITION
    sat = GetSatellite(height,theta_inc)

    #====================================================================================#
    # RUN CELL SIMULATION
    #------------------------------------------------------------------------------------#
    # PRIMARY
    pos1,theta_norm,eps = scatts.GetScatterers(num_looks,num_scatts,scene_shape,scene_buffer,slope_mean,slope_stdd,eps_mean,eps_stdd)        
    scatts1 = RunForwardPropagation(sat,pos1,eps,theta_inc,theta_norm,antenna_pattern_rad,pol,wl)
    
    m_buffer = ~GetSceneMask(pos1,scene_shape) # Determine scatterers inside scene (i.e. remove buffer)
    scatts1[m_buffer] = np.nan        
    signal1 = np.nansum(scatts1,axis=-1)

    #------------------------------------------------------------------------------------#
    # SECONDARY
    pos2 = ApplyRandomGaussianShifts(pos1,*motion_stdd)
    scatts2 = RunForwardPropagation(sat,pos2,eps,theta_inc,theta_norm,antenna_pattern_rad,pol,wl)

    m_buffer = ~GetSceneMask(pos2,scene_shape) # Determine scatterers inside scene (i.e. remove buffer)        
    scatts2[m_buffer] = np.nan        
    signal2 = np.nansum(scatts2,axis=-1)

    #====================================================================================#
    # COMPUTE SAMPLE COHERENCE
    coh = ComputeSampleCoherence(signal1,signal2)

    return coh

def PlotGammaTempVsMotion(x:NDArray,lut_coh:dict[str,NDArray],theta_inc):
    xlim = (x[0],x[-1])

    fig,axs = plt.subplots(1,1,dpi=100,figsize=(12,9))
    ax = axs
    ax.set_facecolor((.9,.9,.9))
    for wl,ll in lut_coh.items():
        y_simul = np.abs(np.array(ll))
        y_theory = np.exp(-0.5 * ((4 * np.pi / wl) ** 2) * ((x ** 2) * (np.sin(theta_inc) ** 2)))

        l = ax.plot(x,y_simul,label=f'{wl:.3f} [m]')
        l = ax.plot(x,y_theory,label=f'{wl:.3f} [m]',linestyle=':',color=l[-1].get_color())

    ax.set_xscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim((-0.05,1.05))
    ax.grid()
    fig.tight_layout()
    plt.show(block=True)

    return 

def run():
    #================================================================#
    # SETTINGS
    # Satellite
    sat_height = 10E3 # [m]
    theta_inc_deg = 45 # [deg]
    antenna_pattern_deg = 5
    wl = 0.031

    # Scene
    scene_shape = (20,20) # [m,m]
    scene_buffer = 2 # [m]
    slope_mean_deg = 0 # [deg]

    # Scatterers
    num_scatts = 10000
    num_looks = 100
    slope_stdd_deg = 90 # [deg]
    eps_mean = 3.00 + 1j * 0.10
    eps_stdd = 0.10 + 1j * 0.01
    pol = 's' # {s,p} --> {}

    #================================================================#
    # CONVERT DEG2RAD
    theta_inc_rad = np.deg2rad(theta_inc_deg)
    slope_mean_rad = np.deg2rad(slope_mean_deg)
    slope_stdd_rad = np.deg2rad(slope_stdd_deg)
    antenna_pattern_rad = np.deg2rad(antenna_pattern_deg)

    #================================================================#
    # DEFINE SCENE
    num_steps = 100
    x = 10 ** np.linspace(np.log10(0.001),np.log10(1.0),num_steps)

    lut_coh = {}
    for wl in [0.031,0.055,0.24,0.69]:
        opid = f'WL={wl:.3f} [m]'
        coh_wl = []
        for i,motion_stdd in enumerate(x):
            print(f'{opid}: {i+1}/{num_steps}',end='\r')
            coh = SimulSampleCoherence(
                num_looks,num_scatts,scene_shape,scene_buffer,
                eps_mean,eps_stdd,slope_mean_rad,slope_stdd_rad,
                sat_height,theta_inc_rad,
                antenna_pattern_rad,pol,wl,
                (motion_stdd,0.0,0.0))
            
            coh_wl.append(coh)

        lut_coh[wl] = coh_wl

        print(f'{opid}: COMPLETED.')

    PlotGammaTempVsMotion(x,lut_coh)

    return

if __name__ == '__main__':
    run()