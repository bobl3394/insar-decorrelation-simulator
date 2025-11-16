import matplotlib
matplotlib.use('tkagg') 
del matplotlib

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src import fresnel,scatts,models
from src.plotting import SetRC
from os import path,makedirs

LUT_ESP_REL = {
    "dry":  3.0 - 0.1j,
    "wet": 15.0 - 2.1j
}


def PlotScattererOrientation(savepath:str,theta_scatt:NDArray,theta_norm:float=None,theta_inc:float=None,n_bins:int=50,plot_semicircle:bool=False):
    SetRC(legendFontSize=18,yTickLabelSize=18)
    lw = 3
    c2 = 'red'

    hist,edges = np.histogram(theta_scatt,bins=n_bins,range=(-np.pi,+np.pi))
    # hist = hist / hist.sum()
    width = (2*np.pi) / n_bins

    fig,axs = plt.subplots(figsize=(11,10),dpi=100,subplot_kw={"projection": "polar"})
    ax:plt.Axes = axs
    ax.set_theta_zero_location("N")
    ax.bar(edges[:-1],hist,width=width,bottom=0.0,align='edge')
    rmax = ax.get_rmax()
    if theta_norm is not None: ax.plot([theta_norm,theta_norm],[0,rmax],color=c2,lw=lw)
    if theta_inc is not None: ax.plot([theta_inc,theta_inc],[0,rmax],color=c2,lw=lw,ls=':')
    ax.set_rlabel_position(0) 

    # UNCOMMENT TO ONLY PLOT SEMI-CIRCLE
    if plot_semicircle:
        ax.set_thetamin(-90)
        ax.set_thetamax(90)
    
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath)
        plt.close(fig)
    else:
        plt.show(block=True)

    return

def PlotGammaTempVsMotion(savepath:str,x:NDArray,lut_coh:dict[str,NDArray],theta_inc):
    SetRC(legendFontSize=18)
    lw = 3
    xlabel = r'$\sigma_{motion}$ [$m$]'
    ylabel = r'$\gamma_{temp}$'

    xlim = (x[0],x[-1])

    fig,axs = plt.subplots(1,1,dpi=100,figsize=(12,9))
    ax = axs
    ax.set_facecolor((.9,.9,.9))
    for wl,ll in lut_coh.items():
        y_simul = np.abs(np.array(ll))
        y_theory = models.TemporalDecorrelation(wl,theta_inc,ymotion=x,zmotion=0.0)

        label = rf'$\lambda={wl:.3f}$ [$m$]'

        # Option-1: data as dots, model as solid line
        l = ax.plot(x,y_theory,label=label,lw=lw,linestyle='-')        
        ax.scatter(x,y_simul,s=15,color=l[-1].get_color())

        # Option-2: data as solid line, model as dashed line
        # l = ax.plot(x,y_simul,label=label+' (data)',lw=lw)
        # l = ax.plot(x,y_theory,label=label+' (model)',lw=lw,linestyle=':',color=l[-1].get_color())

    ax.set_xscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim((-0.05,1.05))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color='white')
    ax.legend(framealpha=0.0,loc='upper right')

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath)
        plt.close(fig)
    else:
        plt.show(block=True)

    return


def PlotScattererDistribution(savepath:str,data:NDArray,*,n_bins:int=100):
    SetRC()
    
    data = data[np.isfinite(data)]

    fig,axs = plt.subplots(2,2,dpi=100,figsize=(16,9))
    
    ax:plt.Axes = axs[0,0]
    ax.set_facecolor((.9,.9,.9))
    ax.hist(data.real,bins=n_bins)
    ax.set_xlabel(r'$\mathcal{Re}(s)$')
    ax.set_ylabel('Samples')
    ax.grid(color='white')

    ax:plt.Axes = axs[0,1]
    ax.set_facecolor((.9,.9,.9))
    ax.hist(data.imag,bins=n_bins)
    ax.set_xlabel(r'$\mathcal{Im}(s)$')
    ax.set_ylabel('Samples')
    ax.grid(color='white')

    ax:plt.Axes = axs[1,0]
    ax.set_facecolor((.9,.9,.9))
    ax.hist(np.abs(data),bins=n_bins)
    ax.set_xlabel(r'$|s|$')
    ax.set_ylabel('Samples')
    ax.grid(color='white')

    ax:plt.Axes = axs[1,1]
    ax.set_facecolor((.9,.9,.9))
    ax.hist(np.angle(data),bins=n_bins,range=(-np.pi,+np.pi))
    ax.set_xlabel(r'$\angle s$')
    ax.set_ylabel('Samples')
    ax.grid(color='white')

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath)
        plt.close(fig)
    else:
        plt.show(block=True)

    return

def Plot2DScattererMotion(savepath:str,p1:NDArray,p2:NDArray,*,extent:tuple=None,buffer:int=0):
    SetRC(legendFontSize=18)

    d = p2 - p1

    # Create figure + axis with subplots
    fig, ax = plt.subplots(figsize=(12,9))
    ax.set_facecolor((.9,.9,.9))

    if extent is not None:
        xmin,xmax,ymin,ymax = extent
        width = (xmax - xmin)
        height = (ymax - ymin)

        square = patches.Rectangle((xmin,ymin),width,height,fill=False,edgecolor='red',linewidth=2)
        ax.add_patch(square)

        if buffer > 0:
            height += (2 * buffer)
            width  += (2 * buffer)
            xmin -= buffer
            ymin -= buffer
            square = patches.Rectangle((xmin,ymin),width,height,fill=False,edgecolor='red',linewidth=2,linestyle=':')
            ax.add_patch(square)        

    # Plot points
    ax.scatter(p1[:,0],p1[:,1],label='Before',color='blue')
    ax.scatter(p2[:,0],p2[:,1],label='After',color='red')

    # Draw arrows
    for i in range(len(p1)):
        ax.arrow(
            p1[i,0],p1[i,1],d[i,0],d[i,1],
            length_includes_head=True,
            head_width=0.005, head_length=0.01,
            color='gray')

    # Styling
    ax.set_xlabel("Along-Track [m]")
    ax.set_ylabel("Across-Track [m]")
    ax.set_title("2D Positions and Motion Vectors")
    ax.axis('equal')
    ax.grid(color='white')
    ax.legend(framealpha=0.0,loc='upper right')

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath)
        plt.close(fig)
    else:
        plt.show(block=True)

    return

def Plot3DScattererMotion(savepath:str,p1:NDArray,p2:NDArray):
    SetRC(legendFontSize=18)
    fc = (.9,.9,.9)
    lp = 20

    d = p2 - p1

    # Create figure + axis with subplots
    fig,axs = plt.subplots(figsize=(12,9),subplot_kw={"projection":"3d"})
    fig.set_facecolor(fc)
    ax:plt.Axes = axs
    ax.set_facecolor(fc)

    # Plot points
    ax.scatter(p1[...,0],p1[...,1],p1[...,2],label='Before',color='blue')
    ax.scatter(p2[...,0],p2[...,1],p2[...,2],label='After',color='red')

    # Draw arrows
    ax.quiver(
        p1[...,0],p1[...,1],p1[...,2],d[...,0],d[...,1],d[...,2], 
        length=1.0,normalize=False,color='gray')

    # Styling
    ax.set_xlabel("Along-Track [m]",labelpad=lp)
    ax.set_ylabel("Across-Track [m]",labelpad=lp)
    ax.set_zlabel("Elevation [m]",labelpad=lp)
    ax.set_title("3D Positions and Motion Vectors")
    ax.axis('equal')
    ax.grid(color='white')
    ax.legend(framealpha=0.0,loc='upper right')

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath)
        plt.close(fig)
    else:
        plt.show(block=True)

    return


def GetSatellite(height,inc):
    d = height * np.tan(inc)
    return np.array([[-d,0,height]])

def RunForwardPropagation(sat,pos,eps,theta_inc,theta_norm,antenna_pattern,pol,wl):
    lia = (theta_inc - theta_norm)
    lia[np.abs(lia) > antenna_pattern] = 0.0 # VERY BASIC ANTENNA PATTERN GAIN :D

    t = fresnel.GetReflectionCoefficient(1.0,eps,lia,pol)

    dist = np.linalg.norm(pos - sat,axis=-1)

    prop_phase_delay = (4 * np.pi * dist) / wl
    signals = t * np.exp(-1j * prop_phase_delay)

    return signals

def GetSceneMask(pos:NDArray,xlim:tuple,ylim:tuple,zlim:tuple)->NDArray:
    m_scene = (
        (pos[...,0] >= xlim[0]) & (pos[...,0] <= xlim[1]) &
        (pos[...,1] >= ylim[0]) & (pos[...,1] <= ylim[1]) &
        (pos[...,2] >= zlim[0]) & (pos[...,2] <= zlim[1]))

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
        num_looks:int,
        num_scatts:int,
        xlim:tuple,
        ylim:tuple,
        zlim:tuple,
        buffer:int,
        norm_kwargs:dict,
        diel_kwargs:dict,
        motion_stdd:tuple,
        height:float,
        theta_inc:float,        
        antenna_pattern_rad:float,        
        wl:float,     
        pol:str,   
        savedir:str=None
    ):
    
    size = (num_looks,num_scatts)

    #====================================================================================#
    # COMPUTE SATELLITE POSITION
    sat = GetSatellite(height,theta_inc)

    #====================================================================================#
    # RUN CELL SIMULATION
    #------------------------------------------------------------------------------------#
    # PRIMARY
    pos1,norm1,diel1 = scatts.GetScatterers(size,xlim,ylim,zlim,buffer,norm_kwargs,diel_kwargs)        
    scatts1 = RunForwardPropagation(sat,pos1,diel1,theta_inc,norm1,antenna_pattern_rad,pol,wl)
    
    m_buffer = ~GetSceneMask(pos1,xlim,ylim,zlim) # Determine scatterers inside scene (i.e. remove buffer)
    # scatts1[m_buffer] = np.nan        
    signal1 = np.nansum(scatts1,axis=-1)

    #------------------------------------------------------------------------------------#
    # SECONDARY
    pos2 = ApplyRandomGaussianShifts(pos1,*motion_stdd)
    scatts2 = RunForwardPropagation(sat,pos2,diel1,theta_inc,norm1,antenna_pattern_rad,pol,wl)

    m_buffer = ~GetSceneMask(pos2,xlim,ylim,zlim) # Determine scatterers inside scene (i.e. remove buffer)        
    # scatts2[m_buffer] = np.nan        
    signal2 = np.nansum(scatts2,axis=-1)

    #====================================================================================#
    # COMPUTE SAMPLE COHERENCE
    coh = ComputeSampleCoherence(signal1,signal2)
    
    if savedir:
        sp = path.join(savedir,'elemscatt_distribution.jpg')
        PlotScattererDistribution(sp,scatts1[0],n_bins=100)

        sp = path.join(savedir,'scattlooks_distribution.jpg')
        PlotScattererDistribution(sp,signal1,n_bins=100)
    
        sp = path.join(savedir,'elemscatt_orientation.jpg')
        PlotScattererOrientation(sp,norm1[0],norm_kwargs['mean'],theta_inc)

        sp = path.join(savedir,'elemscatt_motion2d.jpg')
        Plot2DScattererMotion(sp,pos1[0],pos2[0],extent=(*xlim,*ylim),buffer=buffer)

        sp = path.join(savedir,'elemscatt_motion3d.jpg')
        Plot3DScattererMotion(sp,pos1[0],pos2[0])

    return coh


def run():
    #================================================================#
    # SETTINGS
    # Plots
    savedir:str = 'plots'

    # Satellite
    sat_height = 10E3 # [m]
    theta_inc_deg = 45 # [deg]
    antenna_pattern_deg = 5
    wl = 0.031

    # Scene
    xlim = [-10,10]
    ylim = [-10,10]
    zlim = [0,20]
    buffer = 2 # [m]
    slope_mean_deg = 0 # [deg]
    slope_stdd_deg = 45 # [deg]

    # Scatterers
    num_scatts = 100
    num_looks = 1000
    eps_mean = 3.00 + 1j * 0.10
    eps_stdd = 0.10 + 1j * 0.01    
    motion_stdd = (1.0,1.0,1.0)
    pol = 's' # {s,p} --> {}

    #================================================================#
    # CONVERT DEG2RAD
    theta_inc_rad = np.deg2rad(theta_inc_deg)
    slope_mean_rad = np.deg2rad(slope_mean_deg)
    slope_stdd_rad = np.deg2rad(slope_stdd_deg)
    antenna_pattern_rad = np.deg2rad(antenna_pattern_deg)

    norm_kwargs = {'distribution':'Gaussian','mean':slope_mean_rad,'stdd':slope_stdd_rad}  
    diel_kwargs = {'distribution':'Gaussian','mean':eps_mean,'stdd':eps_stdd}

    makedirs(savedir,exist_ok=True)

    #================================================================#
    # DEFINE SCENE
    num_steps = 100
    x = 10 ** np.linspace(np.log10(0.001),np.log10(1.0),num_steps)

    lut_coh = {}
    for wl in [0.031,0.055,0.24,0.69]:
        opid = f'WL={wl:.3f} [m]'
        coh_wl = []
        for i,_ in enumerate(x):
            print(f'{opid}: {i+1}/{num_steps}',end='\r')
            coh = SimulSampleCoherence(
                num_looks,num_scatts,xlim,ylim,zlim,buffer,
                norm_kwargs,diel_kwargs,motion_stdd, # (motion_stdd,0.0,0.0)
                sat_height,theta_inc_rad,antenna_pattern_rad,
                wl,pol,None)
            
            coh_wl.append(coh)

        lut_coh[wl] = coh_wl

        print(f'{opid}: COMPLETED.')

    
    sp = path.join(savedir,'gamma_temp_simul_vs_model.jpg')
    PlotGammaTempVsMotion(sp,x,lut_coh,theta_inc_rad)

    return

if __name__ == '__main__':
    run()