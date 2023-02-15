#!/usr/bin/env python
# coding: utf-8


def insituprofiles(observer, date=None, start=None, end=None, filepath=None, save_fig=True, best=True, mean=False, legend=True, fixed=None):
    
    """
    Plots the synthetic insitu data plus the measured insitu data and ensemble fit.

    Arguments:
        observer          name of the observer
        start             starting point of the plot
        end               ending point of the plot
        filepath              where to find the fitting results
        save_fig          whether to save the created figure
        best              whether to plot run with min(eps)
        mean              whether to plot run with mean parameter values
        legend            whether to plot legend 

    Returns:
        None
    """
    
    if start == None:
        logger.info("Please specify start time of plot")

    if end == None:
        logger.info("Please specify end time of plot")
        
    t = [start + datetime.timedelta(hours=i) for i in range(96)]
    
    pos_temp = getpos(observer, date, start, end)
    
    pos = [pos_temp for _ in t]
    
    if best == True:
        model_obj = returnfixedmodel(filepath)
        
        outa = np.squeeze(np.array(model_obj.simulator(t, pos))[0])
        outa[outa==0] = np.nan
        
    if fixed is not None:
        model_obj = returnfixedmodel(filepath, fixed)
        
        outa = np.squeeze(np.array(model_obj.simulator(t, pos))[0])
        outa[outa==0] = np.nan
    
    if mean == True:
        model_obj = returnfixedmodel(filepath, fixed_iparams_arr='mean')
        
        means = np.squeeze(np.array(model_obj.simulator(t, pos))[0])
        means[means==0] = np.nan
    
    lw_best = 3  # linewidth for plotting the min(eps) run
    lw_mean = 3  # linewidth for plotting the mean run
    
    if observer == 'solo':
        obs_title = 'Solar Orbiter'
    
    if observer == 'psp':
        obs_title = 'Parker Solar Probe'

    plt.figure(figsize=(20, 10))
    plt.title("3DCORE synthetic profiles - "+obs_title)
        
    if (best == True) or (fixed is not None):
        if best == True:
            plt.plot(t, np.sqrt(np.sum(outa**2, axis=1)), "k", alpha=0.5, linestyle='dashed', lw=lw_best)#, label ='run with min(eps)')
        else:
            plt.plot(t, np.sqrt(np.sum(outa**2, axis=1)), "k", alpha=0.5, linestyle='dashed', lw=lw_best)#, label ='run with fixed iparams')
        plt.plot(t, outa[:, 0], "r", alpha=0.5,linestyle='dashed', lw=lw_best)
        plt.plot(t, outa[:, 1], "g", alpha=0.5,linestyle='dashed', lw=lw_best)
        plt.plot(t, outa[:, 2], "b", alpha=0.5,linestyle='dashed', lw=lw_best)
        
    if mean == True:
        plt.plot(t, np.sqrt(np.sum(means**2, axis=1)), "k", alpha=0.5, linestyle='dashdot', lw=lw_mean)#, label ='run with mean iparams')
        plt.plot(t, means[:, 0], "r", alpha=0.75,linestyle='dashdot', lw=lw_mean)
        plt.plot(t, means[:, 1], "g", alpha=0.75,linestyle='dashdot', lw=lw_mean)
        plt.plot(t, means[:, 2], "b", alpha=0.75,linestyle='dashdot', lw=lw_mean)
        
        
        
    date_form = mdates.DateFormatter("%h %d %H")
    plt.gca().xaxis.set_major_formatter(date_form)
           
    plt.ylabel("B [nT]")
    # plt.xlabel("Time")
    plt.xticks(rotation=25, ha='right')
    if legend == True:
        plt.legend(loc='lower right')
    if save_fig == True:
        plt.savefig('%s_fullinsitu.pdf' %filepath, dpi=300)    
    plt.show()
        

