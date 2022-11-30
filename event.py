import pandas as pds
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
    ic=pds.read_csv(url)

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
    wini=np.where(isc=='Wind')[0]
    stai=np.where(isc=='STEREO-A')[0]
    stbi=np.where(isc=='STEREO-B')[0]
    pspi=np.where(isc=='PSP')[0]
    soloi=np.where(isc=='SolarOrbiter')[0]
    bepii=np.where(isc=='BepiColombo')[0]
    ulyi=np.where(isc=='Ulysses')[0]
    messi=np.where(isc=='Messenger')[0]
    vexi=np.where(isc=='VEX')[0]
    
    wincat = get_evtlist(icme_start_time, mo_start_time, mo_end_time, wini)
    print('got wincat: '+str(len(wincat)))
    stacat = get_evtlist(icme_start_time, mo_start_time, mo_end_time, stai)
    print('got stacat: '+str(len(stacat)))
    stbcat = get_evtlist(icme_start_time, mo_start_time, mo_end_time, stbi)
    print('got stbcat: '+str(len(stbcat)))
    pspcat = get_evtlist(icme_start_time, mo_start_time, mo_end_time, pspi)
    print('got pspcat: '+str(len(pspcat)))
    solocat = get_evtlist(icme_start_time, mo_start_time, mo_end_time, soloi)
    print('got solocat: '+str(len(solocat)))
    bepicat = get_evtlist(icme_start_time, mo_start_time, mo_end_time, bepii)
    print('got bepicat: '+str(len(bepicat)))
    ulycat = get_evtlist(icme_start_time, mo_start_time, mo_end_time, ulyi)
    print('got ulycat: '+str(len(ulycat)))
    messcat = get_evtlist(icme_start_time, mo_start_time, mo_end_time, messi)
    print('got messcat: '+str(len(messcat)))
    vexcat = get_evtlist(icme_start_time, mo_start_time, mo_end_time, vexi)
    print('got vexcat: '+str(len(vexcat)))
    
    return wincat,stacat,stbcat,pspcat,solocat,bepicat,ulycat,messcat,vexcat

def get_evtlist(begin, mobegin, end, ind, dateFormat="%Y/%m/%d %H:%M",sep=','):
    '''
    get indices of events by different spacecraft
    '''    
    evtList = []
    begin = pds.to_datetime(begin[ind], format=dateFormat)
    mobegin = pds.to_datetime(mobegin[ind], format=dateFormat)
    end = pds.to_datetime(end[ind], format=dateFormat)
    for i in ind:
        evtList.append(Event(begin[i], mobegin[i], end[i]))
    return evtList

def findevent(cat, year=None,month=None,day=None,hour=None):

    event = cat
    
    if year is not None:
        event = [e for e in event if e.begin.year == year]
    if month is not None:
        event = [e for e in event if e.begin.month == month]
    if day is not None:
        event = [e for e in event if e.begin.day == day]
    if hour is not None:
        event = [e for e in event if e.begin.hour == hour]
    
    if event == []:
        print('No event found, search in different time interval!')
        
    else:
        print('Number of events in time interval: ' + str(len(event)))
    
    return event