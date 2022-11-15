import numpy as np

import datetime as datetime
from datetime import timedelta

from py3dcore_h4c.models.toroidal import thin_torus_gh, thin_torus_qs, thin_torus_sq

from itertools import product

import measure as ms

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