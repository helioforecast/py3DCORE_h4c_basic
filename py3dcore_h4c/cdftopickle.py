import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.dates import  DateFormatter
import datetime
from datetime import timedelta
import time
import pickle
import seaborn as sns

import astropy
import astropy.constants as const
import astropy.units as u
#import astroquery  
import sunpy
from sunpy.coordinates import frames, get_horizons_coord
from sunpy.time import parse_time
import astrospice

#import heliopy.data.spice as spicedata
#import heliopy.spice as spice


import logging

import sys
import os
#import urllib
#import json
#import importlib
import pandas as pd
#import copy
#import openpyxl
#import h5py
import cdflib


import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

#import heliosat

#on mac
#sys.path.append('/Users/chris/python/heliocats')

#from heliocats import plot as hp
#importlib.reload(hp) #reload again while debugging

#from heliocats import data as hd
#importlib.reload(hd) #reload again while debugging

#from heliocats import cats as hc
#importlib.reload(hc) #reload again while debugging

#from heliocats import stats as hs
#importlib.reload(hs) #reload again while debugging

#where the in situ data files are located is read 
#from config.py 

#import config
#importlib.reload(config)
#from config import data_path
#from config import data_path_ML

from astropy.constants import au
AU=au.value/1e3 #in km

from astropy.constants import R_sun
RS=R_sun.value*1e-3 #in km


def cdftopickle(magpath, swapath, sc):
    
    if sc == 'solo':
        fullname = 'solar orbiter'
    
    ll_path = swapath
    
    files = os.listdir(ll_path)
    files.sort()
    llfiles = [os.path.join(ll_path, f) for f in files]
    
    timep=np.zeros(0,dtype=[('time',object)])
    den=np.zeros(0)
    temp=np.zeros(0)
    vr=np.zeros(0)
    vt=np.zeros(0)
    vn=np.zeros(0)
    
    for i in np.arange(0,len(llfiles)):
        p1 = cdflib.CDF(llfiles[i])
        
        den1=p1.varget('N')
        speed1=p1.varget('V_RTN')
        temp1=p1.varget('T')

        vr1=speed1[:,0]
        vt1=speed1[:,1]
        vn1=speed1[:,2]

        vr=np.append(vr1,vr)
        vt=np.append(vt1,vt)
        vn=np.append(vn1,vn)


        temp=np.append(temp1,temp)
        den=np.append(den1,den)


        time1=p1.varget('EPOCH')
        t1=parse_time(cdflib.cdfastropy.convert_to_astropy(time1, format=None)).datetime
        timep=np.append(timep,t1)
        
        temp=temp*(1.602176634*1e-19)/(1.38064852*1e-23) # from ev to K 

    ll_path = magpath

    files = os.listdir(ll_path)
    files.sort()
    llfiles = [os.path.join(ll_path, f) for f in files]
    
    br1=np.zeros(0)
    bt1=np.zeros(0)
    bn1=np.zeros(0)
    time1=np.zeros(0,dtype=[('time',object)])
    
    for i in np.arange(0,len(llfiles)):
        m1 = cdflib.CDF(llfiles[i])
        
        b=m1.varget('B_RTN')
        br=b[:,0]
        bt=b[:,1]
        bn=b[:,2]

        br1=np.append(br1,br)
        bt1=np.append(bt1,bt)
        bn1=np.append(bn1,bn)

        time=m1.varget('EPOCH')

        t1=parse_time(cdflib.cdfastropy.convert_to_astropy(time, format=None)).datetime
        time1=np.append(time1,t1)
        
    starttime = time1[0].replace(hour = 0, minute = 0, second=0, microsecond=0)
    endtime = time1[-1].replace(hour = 0, minute = 0, second=0, microsecond=0)
    
    time_int = []
    while starttime < endtime:
            time_int.append(starttime)
            starttime += timedelta(minutes=1)


    time_int_mat=mdates.date2num(time_int)
    time1_mat=mdates.date2num(time1)
    timep_mat=mdates.date2num(timep)  
    
    solo_ll=np.zeros(np.size(time_int),dtype=[('time',object),('bx', float),('by', float),\
            ('bz', float),('bt', float),('r', float),('lat', float),('lon', float),\
            ('x', float),('y', float),('z', float),('vx', float),\
            ('vy', float),('vz', float),('vt', float),('tp', float),('np', float) ] )   

    solo_ll = solo_ll.view(np.recarray)  


    solo_ll.time=time_int
    solo_ll.bx=np.interp(time_int_mat, time1_mat,br1)
    solo_ll.by=np.interp(time_int_mat, time1_mat,bt1)
    solo_ll.bz=np.interp(time_int_mat, time1_mat,bn1)
    solo_ll.bt=np.sqrt(solo_ll.bx**2+solo_ll.by**2+solo_ll.bz**2)


    solo_ll.np=np.interp(time_int_mat, timep_mat,den)
    solo_ll.tp=np.interp(time_int_mat, timep_mat,temp) 
    solo_ll.vx=np.interp(time_int_mat, timep_mat,vr)
    solo_ll.vy=np.interp(time_int_mat, timep_mat,vt)
    solo_ll.vz=np.interp(time_int_mat, timep_mat,vn)
    solo_ll.vt=np.sqrt(solo_ll.vx**2+solo_ll.vy**2+solo_ll.vz**2)


    #spacecraft position with astrospice
    kernels = astrospice.registry.get_kernels(fullname, 'predict')
    solo_kernel = kernels[0]

    solo_coords = astrospice.generate_coords(fullname, time_int)
    solo_coords_heeq = solo_coords.transform_to(sunpy.coordinates.HeliographicStonyhurst())

    solo_ll.lon=solo_coords_heeq.lon.value
    solo_ll.lat=solo_coords_heeq.lat.value
    solo_ll.r=solo_coords_heeq.radius.to(u.au).value
    
    solo_ll = sphere2cart(solo_ll)
    
    return solo_ll
    
    
def sphere2cart(ll):
        
    ll.x = ll.r * np.sin(ll.lat) * np.cos(ll.lon)
    ll.y = ll.r * np.sin(ll.lat) * np.sin(ll.lon)
    ll.z = ll.r * np.cos(ll.lat)
    
    return ll
