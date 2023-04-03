#!/usr/bin/env python
# coding: utf-8

# # Fitting py3DCORE_h4c

import heliosat as heliosat
import logging as logging
import datetime as datetime
import numpy as np
import os as os
import py3dcore_h4c as py3dcore_h4c

from heliosat.util import sanitize_dt

logging.basicConfig(level=logging.INFO)
logging.getLogger("heliosat.spice").setLevel("WARNING")
logging.getLogger("heliosat.spacecraft").setLevel("WARNING")


if __name__ == "__main__":
    t_launch = datetime.datetime(2022, 6, 2, 7, 30 tzinfo=datetime.timezone.utc)

    t_s = datetime.datetime(2022, 6, 2, 10, tzinfo=datetime.timezone.utc)
    t_e = datetime.datetime(2022, 6, 2, 15, tzinfo=datetime.timezone.utc)

    t_fit = [datetime.datetime(2022, 6, 2, 12, tzinfo=datetime.timezone.utc),
        datetime.datetime(2022, 6, 2, 12, 30, tzinfo=datetime.timezone.utc),
        datetime.datetime(2022, 6, 2, 13, 30, tzinfo=datetime.timezone.utc),
        datetime.datetime(2022, 6, 2, 14, tzinfo=datetime.timezone.utc)]


# Restraining the initial values for the ensemble members leads to more efficient fitting.
# 
#     Model Parameters
#     ================
#         For this specific model there are a total of 14 initial parameters which are as follows:
#         0: t_i          time offset
#         1: lon          longitude
#         2: lat          latitude
#         3: inc          inclination
# 
#         4: dia          cross section diameter at 1 AU
#         5: delta        cross section aspect ratio
# 
#         6: r0           initial cme radius
#         7: v0           initial cme velocity
#         8: T            T factor (related to the twist)
# 
#         9: n_a          expansion rate
#         10: n_b         magnetic field decay rate
# 
#         11: b           magnetic field strength at center at 1AU
#         12: bg_d        solar wind background drag coefficient
#         13: bg_v        solar wind background speed
# 
#         There are 4 state parameters which are as follows:
#         0: v_t          current velocity
#         1: rho_0        torus major radius
#         2: rho_1        torus minor radius
#         3: b_t          magnetic field strength at center

    model_kwargs = {
        "ensemble_size": int(2**16), 
        "iparams": {
           "cme_longitude": {
               "maximum": -60,
               "minimum": -140
           },
           "cme_latitude": {
               "maximum": 50,
               "minimum": -60
           },
           "cme_inclination": {
               "maximum": 50,
               "minimum": 0
           }, 
           "cme_launch_radius": {
               "maximum": 15,
               "minimum": 3
           },
           "cme_launch_velocity": {
               "maximum": 1200,
               "minimum": 300
           },
           "t_factor": {
               "maximum": 250,
               "minimum": -250
           },
           "background_velocity": {
               "maximum": 1200,
               "minimum": 300
           } 
            
        }
    }

    output = 'psp02062022_heeq_512_5/'

    fitter = py3dcore_h4c.ABC_SMC()
    fitter.initialize(t_launch, py3dcore_h4c.ToroidalModel, model_kwargs)
    fitter.add_observer("PSP", t_fit, t_s, t_e, custom_data='psp_2022jun.p')

    fitter.run(ensemble_size=512, reference_frame="HEEQ", jobs=5, workers=5, sampling_freq=3600, output=output, 
               use_multiprocessing=True, custom_data=True)
