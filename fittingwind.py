import heliosat
import logging
import datetime
import numpy as np
import os
import pickle
import py3dcore_h4c
import matplotlib.pyplot as plt



from heliosat.util import sanitize_dt

#logging.basicConfig(level=logging.DEBUG)
logging.getLogger("heliosat.spice").setLevel("WARNING")
logging.getLogger("heliosat.spacecraft").setLevel("WARNING")

t_launch = datetime.datetime(2022, 1, 30, 1,tzinfo=datetime.timezone.utc)

t_s_wind = datetime.datetime(2022, 2, 2, 18, tzinfo=datetime.timezone.utc)
t_e_wind = datetime.datetime(2022, 2, 3, 18,tzinfo=datetime.timezone.utc)

t_wind = [
    datetime.datetime(2022, 2, 3,2, tzinfo=datetime.timezone.utc),
    datetime.datetime(2022, 2, 3,4, tzinfo=datetime.timezone.utc),
    datetime.datetime(2022, 2, 3,6, tzinfo=datetime.timezone.utc),
    datetime.datetime(2022, 2, 3,8, tzinfo=datetime.timezone.utc)
]

model_kwargs = {
    "ensemble_size": int(2**16),
    "iparams": {
        "cme_longitude": {
            "maximum": 5,
            "minimum": -35
        },
        "cme_latitude": {
            "maximum": 15,
            "minimum": -15
        },
        "cme_inclination": {
            "maximum": 330,
            "minimum": 260
        }, 
        "cme_aspect_ratio": {
            "maximum": 2.1,
            "minimum": 1.9
        },
        "cme_launch_velocity": {
            "maximum": 900,
            "minimum": 500
        },
        "cme_launch_radius": {
            "default_value": 8.3
        }
    }
}

fitter = py3dcore_h4c.ABC_SMC()
fitter.initialize(t_launch, py3dcore_h4c.ToroidalModel, model_kwargs)
print('initialized')
fitter.add_observer("WIND", t_wind, t_s_wind, t_e_wind)
print('observer added')
fitter.run(9, 512, reference_frame="HEEQ", jobs=8, workers=8, sampling_freq=3600, output="out_wind_heeq/")