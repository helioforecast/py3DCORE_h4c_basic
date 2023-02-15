def full3d_multiview(t_launch, filepath):
    
    """
    Plots 3d from multiple views.
    """
    
    TP_A =  t_launch + datetime.timedelta(hours=2)
    TP_B =  t_launch + datetime.timedelta(hours=46)

    C_A = "xkcd:red"
    C_B = "xkcd:blue"

    C0 = "xkcd:black"
    C1 = "xkcd:magenta"
    C2 = "xkcd:orange"
    C3 = "xkcd:azure"

    sns.set_style('whitegrid')

    fig = plt.figure(figsize=(15, 11),dpi=100)

    #define subplot grid
    ax1 = plt.subplot2grid((2, 3), (0, 0),rowspan=2,colspan=2,projection='3d')  
    ax2 = plt.subplot2grid((2, 3), (0, 2),projection='3d')  
    ax3 = plt.subplot2grid((2, 3), (1, 2),projection='3d')  
    
    model_obj = returnmodel(filepath)
    
    ######### tilted view
    plot_configure(ax1, view_azim=150, view_elev=25, view_radius=.2,light_source=True) #view_radius=.08

    plot_3dcore(ax1, model_obj, TP_A, color=C_A,light_source = True)
    plot_3dcore_field(ax1, model_obj, color=C_A, step_size=0.0005, lw=1.0, ls="-")
    plot_traj(ax1, "Parker Solar Probe", t_snap = TP_A, frame="HEEQ", custom_data = 'sunpy', color=C_A)
    
    plot_3dcore(ax1, model_obj, TP_B, color=C_B, light_source = True)
    plot_3dcore_field(ax1, model_obj, color=C_B, step_size=0.001, lw=1.0, ls="-")
    plot_traj(ax1, "Solar Orbiter", t_snap = TP_B, frame="HEEQ", custom_data = 'sunpy', color=C_B)
    
    #dotted trajectory
    #plot_traj(ax1, "PSP", TP_B, frame="ECLIPJ2000", color="k", traj_pos=False, traj_major=None, traj_minor=144,lw=1.5)

    #shift center
    plot_shift(ax1,0.31,-0.25,0.0,-0.2)
    