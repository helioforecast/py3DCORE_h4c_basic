#!/usr/bin/env python
# coding: utf-8

# # Fitting py3DCORE_h4c

import heliosat as heliosat
import logging as logging
import datetime as datetime
import numpy as np
import os as os
import py3dcore_h4c as py3dcore_h4c
import shutil as shutil

from heliosat.util import sanitize_dt

logging.basicConfig(level=logging.INFO)
logging.getLogger("heliosat.spice").setLevel("WARNING")
logging.getLogger("heliosat.spacecraft").setLevel("WARNING")


if __name__ == "__main__":
    t_launch = datetime.datetime(2022, 9, 5, 18, 45, tzinfo=datetime.timezone.utc) # launch time assumed at CME impact at PSP at 14.72 Rs

    t_s_solo = datetime.datetime(2022, 9, 7, 8, tzinfo=datetime.timezone.utc) 
    t_e_solo = datetime.datetime(2022, 9, 8, 5, tzinfo=datetime.timezone.utc)

    t_solo = [
        datetime.datetime(2022, 9, 7, 9, tzinfo=datetime.timezone.utc),
        datetime.datetime(2022, 9, 7, 15, tzinfo=datetime.timezone.utc),
        datetime.datetime(2022, 9, 7, 21, tzinfo=datetime.timezone.utc),
        datetime.datetime(2022, 9, 8, 3, tzinfo=datetime.timezone.utc)#,
#        datetime.datetime(2022, 9, 7, 18, tzinfo=datetime.timezone.utc)
    ]

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
        "ensemble_size": int(2**16), #2**17
        "iparams": {
           "cme_longitude": {
               "maximum": 180,
               "minimum": -180
           },
           "cme_latitude": {
               "maximum": 90,
               "minimum": -90
           },
           "cme_inclination": {
               "maximum": 90,
               "minimum": 0
           }, 
           "cme_launch_velocity": {
               "maximum": 1700,
               "minimum": 500
           },
           "cme_launch_radius": {
               "maximum": 20,
               "minimum": 10
           },
           #"t_factor": {
           #    "maximum": 250,
           #    "minimum": 0
           #},
            "background_velocity": {
               "maximum": 600,
               "minimum": 200
           } 
        }
    }

    output = 'solo06092022_heeq_512_4FP_allP_2/'

    fitter = py3dcore_h4c.ABC_SMC()
    fitter.initialize(t_launch, py3dcore_h4c.ToroidalModel, model_kwargs)
    fitter.add_observer("SOLO", t_solo, t_s_solo, t_e_solo)

    fitter.run(ensemble_size=512, reference_frame="HEEQ", jobs=64, workers=64, sampling_freq=3600, output=output, eps_quantile=0.25, use_multiprocessing=True, custom_data='solo_2022sep.p')
