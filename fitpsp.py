import heliosat as heliosat
import logging as logging
import datetime as datetime
import numpy as np
import os as os
import pickle as pickle
import py3dcore_h4c as py3dcore_h4c
import matplotlib.pyplot as plt
import shutil as shutil
import pandas as pds
import event as evt
from heliosat.util import sanitize_dt


if __name__ == "__main__":
    

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("heliosat.spice").setLevel("WARNING")
    logging.getLogger("heliosat.spacecraft").setLevel("WARNING")
    
    wincat,stacat,stbcat,pspcat,solocat,bepicat,ulycat,messcat,vexcat = evt.get_cat()

    i1au = wincat + stacat + stbcat
    print('ICMECAT events near 1 AU',len(i1au))
    
    pspevent = evt.findevent(pspcat, year=2022, month=6, day=2)
    
    #t_launch = winevent[0].begin-datetime.timedelta(days=4)

    t_launch = datetime.datetime(2022, 6, 2, 6, tzinfo=datetime.timezone.utc)

    t_s_psp = datetime.datetime(2022, 6, 2, 12, tzinfo=datetime.timezone.utc)
    t_e_psp = datetime.datetime(2022, 6, 3, 6, tzinfo=datetime.timezone.utc)

    t_psp = [
        datetime.datetime(2022, 6, 2, 13, tzinfo=datetime.timezone.utc),
        datetime.datetime(2022, 6, 2, 14, tzinfo=datetime.timezone.utc) ,
        datetime.datetime(2022, 6, 2, 15, tzinfo=datetime.timezone.utc),
        datetime.datetime(2022, 6, 2, 16, tzinfo=datetime.timezone.utc),
    ]
    
    
    model_kwargs = {
        "ensemble_size": int(2**17), #2**17
        "iparams": {
           "cme_longitude": {
               "maximum": -110,
               "minimum": -130
           },
           "cme_latitude": {
               "maximum": -10,
               "minimum": -50
           },
           "cme_inclination": {
               "maximum": 10,
               "minimum": 0
           }, 
            "cme_launch_velocity": {
                "maximum": 750,
                "minimum": 250
            },
            "cme_launch_radius": {
                "maximum": 30,
                "minimum": 5
            }
        }
    }
    
    output = 'psp02062022_heeq_512_4FP_ts12/'

    fitter = py3dcore_h4c.ABC_SMC()
    fitter.initialize(t_launch, py3dcore_h4c.ToroidalModel, model_kwargs)
    fitter.add_observer("PSP", t_psp, t_s_psp, t_e_psp)


    fitter.run(ensemble_size=512, reference_frame="HEEQ", sampling_freq=3600, output=output, use_multiprocessing=True) 