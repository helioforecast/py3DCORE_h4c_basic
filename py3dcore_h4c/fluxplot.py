import os

import numpy as np

import datetime as datetime
from datetime import timedelta
import py3dcore_h4c
from py3dcore_h4c.fitter.base import custom_observer

from py3dcore_h4c.models.toroidal import thin_torus_gh, thin_torus_qs, thin_torus_sq

import matplotlib.pyplot as plt

from itertools import product

import py3dcore_h4c.measure as ms

import logging

logger = logging.getLogger(__name__)

def loadpickle(path = None, number = -1):

    """ Loads the filepath of a pickle file. """

    # Get the list of all files in path
    dir_list = sorted(os.listdir(path))

    resfile = []
    respath = []
    # we only want the pickle-files
    for file in dir_list:
        if file.endswith(".pickle"):
            resfile.append(file) 
            respath.append(os.path.join(path,file))
            
    filepath = path + resfile[number]

    return filepath

def standardplot(observer= 'solo', t_fit=None, start = None, end=None, filepath=None, custom_data=False, save_fig = True):
    """
    Plots the insitu data plus the fitted results.

    Arguments:
        observer          name of the observer
        t_fit             datetime points used for fitting
        start             starting point of the plot
        end               ending point of the plot
        path              where to find the fitting results
        number            which result to use
        custom_data       path to custom data, otherwise heliosat is used
        save_fig          whether to save the created figure

    Returns:
        None
    """
         
    if start == None:
        start = t_fit[0]

    if end == None:
        end = t_fit[-1]
    
    
    if custom_data == False:
        observer_obj = getattr(heliosat, observer)() # get observer obj
        logger.info("Using HelioSat to retrieve observer data")
    else:
        observer_obj = custom_observer(custom_data)
        logger.info("Using custom datafile: %s", custom_data)
        
    t, b = observer_obj.get([start, end], "mag", reference_frame="HEEQ", as_endpoints=True)
    
    # get ensemble_data
    ed = py3dcore_h4c.generate_ensemble(filepath, t, reference_frame="HEEQ",reference_frame_to="HEEQ", max_index=128, custom_data=custom_data)

    plt.figure(figsize=(28, 12))
    plt.title(observer+ " fitting result")
    plt.plot(t, np.sqrt(np.sum(b**2, axis=1)), "k", alpha=0.5)
    plt.plot(t, b[:, 0], "r", alpha=0.5)
    plt.plot(t, b[:, 1], "g", alpha=0.5)
    plt.plot(t, b[:, 2], "b", alpha=0.5)
    plt.fill_between(t, ed[0][3][0], ed[0][3][1], alpha=0.25, color="k")
    plt.fill_between(t, ed[0][2][0][:, 0], ed[0][2][1][:, 0], alpha=0.25, color="r")
    plt.fill_between(t, ed[0][2][0][:, 1], ed[0][2][1][:, 1], alpha=0.25, color="g")
    plt.fill_between(t, ed[0][2][0][:, 2], ed[0][2][1][:, 2], alpha=0.25, color="b")
    plt.ylabel("B [nT]")
    plt.xlabel("Time [MM-DD HH]")
    for _ in t_fit:
        plt.axvline(x=_, lw=1, alpha=0.25, color="k", ls="--")
    if save_fig == True:
        plt.savefig('%s.png' %filepath)    
    plt.show()


def plot_configure(ax, **kwargs):
    view_azim = kwargs.pop("view_azim", -25)
    view_elev = kwargs.pop("view_elev", 25)
    view_radius = kwargs.pop("view_radius", .5)
    
    ax.view_init(azim=view_azim, elev=view_elev)

    ax.set_xlim([-view_radius, view_radius])
    ax.set_ylim([-view_radius, view_radius])
    ax.set_zlim([-view_radius, view_radius])
    
    ax.set_axis_off()

def plot_3dcore(ax, obj, t_snap, **kwargs):
    kwargs["alpha"] = kwargs.pop("alpha", .05)
    kwargs["color"] = kwargs.pop("color", "k")
    kwargs["lw"] = kwargs.pop("lw", 1)

    ax.scatter(0, 0, 0, color="y", s=500)

    obj.propagator(t_snap)
    wf_model = obj.visualize_shape(0,)#visualize_wireframe(index=0)
    ax.plot_wireframe(*wf_model.T, **kwargs)

    
def plot_3dcore_field(ax, obj, step_size=0.005, q0=[0.8, .1, np.pi/2],**kwargs):

    #initial point is q0
    q0i =np.array(q0, dtype=np.float32)
    fl = ms.visualize_fieldline(obj, q0, index=0, steps=1000, step_size=step_size)
    #fl = model_obj.visualize_fieldline_dpsi(q0i, dpsi=2*np.pi-0.01, step_size=step_size)
    ax.plot(*fl.T, **kwargs)
    
    
def plot_traj(ax, sat, t_snap, frame="HEEQ", traj_pos=True, traj_major=4, traj_minor=None, **kwargs):
    kwargs["alpha"] = kwargs.pop("alpha", 1)
    kwargs["color"] = kwargs.pop("color", "k")
    kwargs["lw"] = kwargs.pop("lw", 1)
    kwargs["s"] = kwargs.pop("s", 25)
    
    inst = getattr(heliosat, sat)()

    _s = kwargs.pop("s")

    if traj_pos:
        pos = inst.trajectory(t_snap, frame)

        ax.scatter(*pos.T, s=_s, **kwargs)
        
    if traj_major and traj_major > 0:
        traj = inst.trajectory([t_snap + datetime.timedelta(hours=i) for i in range(-traj_major, traj_major)], frame)
        ax.plot(*traj.T, **kwargs)
        
    if traj_minor and traj_minor > 0:
        traj = inst.trajectory([t_snap + datetime.timedelta(hours=i) for i in range(-traj_minor, traj_minor)], frame)
        
        if "ls" in kwargs:
            kwargs.pop("ls")

        _ls = "--"
        _lw = kwargs.pop("lw") / 2
        
        ax.plot(*traj.T, ls=_ls, lw=_lw, **kwargs)

        
def plot_circle(ax,dist,**kwargs):        

    thetac = np.linspace(0, 2 * np.pi, 100)
    xc=dist*np.sin(thetac)
    yc=dist*np.cos(thetac)
    zc=0
    ax.plot(xc,yc,zc,ls='--',color='black',lw=0.3,**kwargs)
      

def plot_satellite(ax,satpos1,**kwargs):

    xc=satpos1[0]*np.cos(np.radians(satpos1[1]))
    yc=satpos1[0]*np.sin(np.radians(satpos1[1]))
    zc=0
    #print(xc,yc,zc)
    ax.scatter3D(xc,yc,zc,**kwargs)
    
    
def visualize_wireframe(obj, index=0, r=1.0, d=10):
        """Generate model wireframe.

        Parameters
        ----------
        index : int, optional
            Model run index, by default 0.
        r : float, optional
            Surface radius (r=1 equals the boundary of the flux rope), by default 1.0.

        Returns
        -------
        np.ndarray
            Wireframe array (to be used with plot_wireframe).
        """
        r = np.array([np.abs(r)], dtype=obj.dtype)

        c = 360 // d + 1
        u = np.radians(np.r_[0:360. + d:d])
        v = np.radians(np.r_[0:360. + d:d])

        # combination of wireframe points in (q)
        arr = np.array(list(product(r, u, v)), dtype=obj.dtype).reshape(c ** 2, 3)

        for i in range(0, len(arr)):
            thin_torus_qs(arr[i], arr[i], obj.iparams_arr[index], obj.sparams_arr[index], obj.qs_xs[index])

        return arr.reshape((c, c, 3))