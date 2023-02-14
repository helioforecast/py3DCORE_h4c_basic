def full3d(spacecraftlist=['solo', 'psp'], planetlist =['Earth'], t=None, traj = False, filepath=None, custom_data=False, save_fig = True, legend = True, title = True,**kwargs):
    
    """
    Plots 3d.
    """
    
    #colors for 3dplots

    c0 = 'mediumseagreen'
    c1 = "xkcd:red"
    c2 = "xkcd:blue"
    
    #Color settings    
    C_A = "xkcd:red"
    C_B = "xkcd:blue"

    C0 = "xkcd:black"
    C1 = "xkcd:magenta"
    C2 = "xkcd:orange"
    C3 = "xkcd:azure"

    earth_color='blue'
    solo_color='orange'
    venus_color='mediumseagreen'
    mercury_color='grey'
    psp_color='black'
    sta_color='red'
    bepi_color='coral' 
    
    sns.set_context("talk")     

    sns.set_style("ticks",{'grid.linestyle': '--'})
    fsize=15

    fig=plt.figure(figsize=(15,12),dpi=200)
    ax = fig.add_subplot(111, projection='3d')
    
    plot_configure(ax, view_azim=0, view_elev=90, view_radius=0.8)
    
    model_obj = returnmodel(filepath)
    
    plot_3dcore(ax, model_obj, t, color=c2)
    plot_3dcore_field(ax, model_obj, color=c2, step_size=0.005, lw=1.1, ls="-")
    

    
    if 'solo' in spacecraftlist:

        plot_traj(ax, sat = 'Solar Orbiter', t_snap = t, frame="HEEQ", traj_pos=True, traj_minor=None, custom_data = 'sunpy', color=solo_color,**kwargs)

        
    if 'psp' in spacecraftlist:
        
        plot_traj(ax, sat = 'Parker Solar Probe', t_snap = t, frame="HEEQ", traj_pos=True, traj_minor=None, custom_data = 'sunpy', color=psp_color,**kwargs)
        
    if 'STEREO-A' in spacecraftlist:
        t_solo, pos_solo, traj_solo = getpos('STEREO-A', t.strftime('%Y-%m-%d-%H'), start, end)
        plot_satellite(ax,pos_solo,color=sta_color,alpha=0.9, label = 'STEREO-A')
        
        
    
    if 'Earth' in planetlist:
        earthpos = np.asarray([1,0, 0])
        plot_planet(ax,earthpos,color=earth_color,alpha=0.9, label = 'Earth')
        plot_circle(ax,earthpos[0])
        
    if 'Venus' in planetlist:
        t_ven, pos_ven, traj_ven  = getpos('Venus Barycenter', t.strftime('%Y-%m-%d-%H'), start, end)
        plot_planet(ax,pos_ven,color=venus_color,alpha=0.9, label = 'Venus')
        plot_circle(ax,pos_ven[0])
        
    if 'Mercury' in planetlist:
        t_mer, pos_mer, traj_mer  = getpos('Mercury Barycenter', t.strftime('%Y-%m-%d-%H'), start, end)
        plot_planet(ax,pos_mer,color=mercury_color,alpha=0.9, label = 'Mercury')
        plot_circle(ax,pos_mer[0])
        
        
    
    if legend == True:
        ax.legend(loc='lower left')
    if title == True:
        plt.title('3DCORE fitting result - ' + t.strftime('%Y-%m-%d-%H'))
    if save_fig == True:
        plt.savefig('%s_3d.pdf' %filepath, dpi=300)  
    
    return fig