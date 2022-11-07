import pandas as pd
import numpy as np

class Event:

    def __init__(self, begin, cloudbegin, end, param=None):
        self.begin = begin
        self.end = end
        self.duration = self.end-self.begin
        self.cloud = cloudbegin
        
def make_cat(wini):
    '''
    read catalog and return eventlist
    '''
    evtList = []
    
    for i in wini.index:
        begin = wini['icme_start_time'][i]
        end = wini['mo_end_time'][i]
        cloud = wini['mo_start_time'][i]
        
        evtList.append(Event(pd.to_datetime(begin), pd.to_datetime(cloud),pd.to_datetime(end)))
                                                                                                       
    return evtList

def get_cat():
    url='https://helioforecast.space/static/sync/icmecat/HELIO4CAST_ICMECAT_v21.csv'
    ic=pd.read_csv(url)

    # Spacecraft
    isc = ic.loc[:,'sc_insitu'] 

    # Shock arrival or density enhancement time
    icme_start_time = ic.loc[:,'icme_start_time']
    #icme_start_time_num = date2num(np.array(icme_start_time))

    # Start time of the magnetic obstacle (mo)
    mo_start_time = ic.loc[:,'mo_start_time']
    #mo_start_time_num = date2num(np.array(mo_start_time))

    # End time of the magnetic obstacle (mo)
    mo_end_time = ic.loc[:,'mo_end_time']
    #mo_end_time_num = date2num(np.array(mo_end_time))

    #get indices for each target
    wini=ic[ic.sc_insitu=='Wind']
    stai=ic[ic.sc_insitu=='STEREO-A']
    stbi=ic[ic.sc_insitu=='STEREO-B']
    pspi=ic[ic.sc_insitu=='PSP']
    soloi=ic[ic.sc_insitu=='SolarOrbiter']
    bepii=ic[ic.sc_insitu=='BepiColombo']
    ulyi=ic[ic.sc_insitu=='Ulysses']
    messi=ic[ic.sc_insitu=='Messenger']
    vexi=ic[ic.sc_insitu=='VEX']
    
    return wini,stai,stbi,pspi,soloi,bepii,ulyi,messi,vexi