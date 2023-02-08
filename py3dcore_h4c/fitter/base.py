# -*- coding: utf-8 -*-

import datetime as datetime
import heliosat as heliosat
import numpy as np
import os as os
import pickle as pickle

from ..model import SimulationBlackBox
from .sumstat import sumstat
from ..util import mag_fft
from heliosat.util import sanitize_dt
from heliosat.transform import transform_reference_frame
from typing import Any, List, Optional, Sequence, Type, Union

from py3dcore_h4c.cdftopickle import cdftopickle

import logging

logger = logging.getLogger(__name__)

def generate_ensemble(path: str, dt: Sequence[datetime.datetime], reference_frame: str = "HCI", reference_frame_to: str = "HCI", perc: float = 0.95, max_index=None, custom_data= False) -> np.ndarray:
    
    """
    Generates an ensemble from a Fitter object.
    
    Arguments:
        path                where to load from
        dt                  time axis used for fitting
        reference_frame     reference frame used for fitter object
        reference_frame_to  reference frame for output data
        perc                percentage of quantile to be used
        max_index           how much of ensemble is kept
        custom_data         path to custom data
    Returns:
        ensemble_data 
    """
    
    observers = BaseFitter(path).observers
    ensemble_data = []
    

    for (observer, _, _, _, _) in observers:
        ftobj = BaseFitter(path) # load Fitter from path
        
        if custom_data == False:
            observer_obj = getattr(heliosat, observer)() # get observer obj
            logger.info("Using HelioSat to retrieve observer data")
        else:
            observer_obj = custom_observer(custom_data)
            logger.info("Using custom datafile: %s", custom_data)
            
            
            
        # simulate flux ropes using iparams from loaded fitter
        ensemble = np.squeeze(np.array(ftobj.model_obj.simulator(dt, observer_obj.trajectory(dt, reference_frame=reference_frame))[0]))
        
        # how much to keep of the generated ensemble
        if max_index is None:
            max_index =  ensemble.shape[1]

        ensemble = ensemble[:, :max_index, :]

        # transform frame
        if reference_frame != reference_frame_to:
            for k in range(0, ensemble.shape[1]):
                ensemble[:, k, :] = transform_reference_frame(dt, ensemble[:, k, :], reference_frame, reference_frame_to)

        ensemble[np.where(ensemble == 0)] = np.nan

        # generate quantiles
        b_m = np.nanmean(ensemble, axis=1)

        b_s2p = np.nanquantile(ensemble, 0.5 + perc / 2, axis=1)
        b_s2n = np.nanquantile(ensemble, 0.5 - perc / 2, axis=1)

        b_t = np.sqrt(np.sum(ensemble**2, axis=2))
        b_tm = np.nanmean(b_t, axis=1)

        b_ts2p = np.nanquantile(b_t, 0.5 + perc / 2, axis=1)
        b_ts2n = np.nanquantile(b_t, 0.5 - perc / 2, axis=1)

        ensemble_data.append([None, None, (b_s2p, b_s2n), (b_ts2p, b_ts2n)])

    return ensemble_data


class BaseFitter(object):
    """
    Class(object) used for fitting.

    Arguments:
        path   Optional     where to save
    Returns:
        None

    Functions:
        add_observer
        initialize
        save
        load
    """
    
    dt_0: datetime.datetime
    locked: bool
    model: Type[SimulationBlackBox]
    model_obj: SimulationBlackBox
    model_kwargs: Optional[dict]
    observers: list

    def __init__(self, path: Optional[str] = None) -> None:
        if path:
            self.locked = True
            self._load(path)
        else:
            self.locked = False

    def add_observer(self, observer: str, dt: Union[str, datetime.datetime, Sequence[str], Sequence[datetime.datetime]], dt_s: Union[str, datetime.datetime], dt_e: Union[str, datetime.datetime], dt_shift: datetime.timedelta = None) -> None:        
        
        """
        Appends an observer.
    
        Arguments:
            observer          name of the observer
            dt                datetime points to be used for fitting
            dt_s              reference point prior to the fluxrope
            dt_e              reference point after the fluxrope
            dt_shift          datetime can be shifted
    
        Returns:
            None
        """
        
        self.observers.append([observer, sanitize_dt(dt), sanitize_dt(dt_s), sanitize_dt(dt_e), dt_shift]) # append observer with sanitized datetime (HelioSat sanitize deals with different timezones)

    def initialize(self, dt_0: Union[str, datetime.datetime], model: Type[SimulationBlackBox], model_kwargs: dict = {}) -> None:
        
        """
        Initializes the Fitter.
        Sets the following properties for self:
            dt_0              sanitized launch time
            observers         empty list
            reference_frame   reference frame to work in
    
        Arguments:
            dt_0              launch time
            model             fluxrope model
            model_kwargs      dictionary containing all kwargs for the model
    
        Returns:
            None
        """
        
        if self.locked:
            raise RuntimeError("is locked")
        
        self.dt_0 = sanitize_dt(dt_0)
        self.model_kwargs = model_kwargs
        self.observers = []
        self.model =  model

    def save(self, path: str, **kwargs: Any) -> None:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        data = {attr: getattr(self, attr) for attr in self.__dict__ if not callable(attr) and not attr.startswith("_")}

        for k, v in kwargs.items():
            data[k] = v

        with open(path, "wb") as fh:
            pickle.dump(data, fh)

    def _load(self, path: str) -> None:
        with open(path, "rb") as fh:
            data = pickle.load(fh)

        for k, v in data.items():
            setattr(self, k, v)

    def _run(self, *args: Any, **kwargs: Any) -> None:
        pass


class FittingData(object):
    
    """
    Class(object) to handle the data used for fitting.
    Sets the following properties for self:
        length                  length of list of observers
        observers               list of observers
        reference_frame         reference frame to work in
    
    Arguments:
        observers               list of observers
        reference_frame         reference frame to work in
        
    Returns:
        None
        
    Functions:
        add_noise
        generate_noise
        generate_data
        sumstat
    """
    
    data_dt: List[np.ndarray]
    data_b: List[np.ndarray]
    data_o: List[np.ndarray]
    data_m: List[np.ndarray]
    data_l: List[int]

    psd_dt: List[np.ndarray]
    psd_fft: List[np.ndarray]    

    length: int
    noise_model: str
    observers: list
    reference_frame: str
    sampling_freq: int

    def __init__(self, observers: list, reference_frame: str) -> None:
        self.observers = observers
        self.reference_frame = reference_frame
        self.length = len(self.observers)

    def add_noise(self, profiles: np.ndarray) -> np.ndarray:
        if self.noise_model == "psd":
            _offset = 0
            for o in range(self.length):
                dt = self.psd_dt[o]
                fft = self.psd_fft[o]
                dtl = self.data_l[o]

                sampling_fac = np.sqrt(self.sampling_freq)

                ensemble_size = len(profiles[0])

                null_flt = (profiles[1 + _offset:_offset + (dtl + 2) - 1, :, 0] != 0)

                # generate noise for each component
                for c in range(3):
                    noise = np.real(np.fft.ifft(np.fft.fft(np.random.normal(0, 1, size=(ensemble_size, len(fft))).astype(np.float32)) * fft) / sampling_fac).T
                    profiles[1 + _offset:_offset + (dtl + 2) - 1, :, c][null_flt] += noise[dt][null_flt]

                _offset += dtl + 2
        else:
            raise NotImplementedError

        return profiles

    def generate_noise(self, noise_model: str = "psd", sampling_freq: int = 300, **kwargs: Any) -> None:
        

        """
        Generates noise according to the noise model.
        Sets the following properties for self:
            psd_dt                altered time axis for power spectrum
            psd_fft               power spectrum
            sampling_freq         sampling frequency
            noise_model           model used to calculate noise

        Arguments:
            noise_model    "psd"     model to use for generating noise (e.g. power spectrum distribution)
            sampling_freq  300       sampling frequency of data

        Returns:
            None
        """
            
        self.psd_dt = []
        self.psd_fft = []
        self.sampling_freq = sampling_freq

        self.noise_model = noise_model

        if noise_model == "psd":
        # get data for each observer
            for o in range(self.length):
                observer, dt, dt_s, dt_e, _ = self.observers[o]

                observer_obj = getattr(heliosat, observer)()

                _, data = observer_obj.get([dt_s, dt_e], "mag", reference_frame=self.reference_frame, sampling_freq=sampling_freq, use_cache=True, as_endpoints=True)
                
                data[np.isnan(data)] = 0 #set all nan values to 0

                fF, fS = mag_fft(dt, data, sampling_freq=sampling_freq) # computes the mean power spectrum distribution

                kdt = (len(fS) - 1) / (dt[-1].timestamp() - dt[0].timestamp()) 
                fT = np.array([int((_.timestamp() - dt[0].timestamp()) * kdt) for _ in dt]) 

                self.psd_dt.append(fT) # appends the altered time axis
                self.psd_fft.append(fS)
                # appends the power spectrum 
        else:
            raise NotImplementedError

    def generate_data(self, time_offset: Union[int, Sequence], **kwargs: Any) -> None:
        
        """
        Generates data for each observer at the given times. 
        Sets the following properties for self:
            data_dt      all needed timesteps [dt_s, dt, dt_e]
            data_b       magnetic field data for data_dt
            data_o       trajectory of observers
            data_m       mask for data_b with 1 for each point except first and last
            data_l       length of data

        Arguments:
            time_offset  shift timeseries for observer   
            **kwargs     Any

        Returns:
            None
        """
        
        self.data_dt = []
        self.data_b = []
        self.data_o = []
        self.data_m = []
        self.data_l = []
        
        # Each observer is treated separately

        for o in range(self.length):
            
            # The values of the observer are unpacked
            
            observer, dt, dt_s, dt_e, dt_shift = self.observers[o]
            
            # The reference points are corrected by time_offset

            if hasattr(time_offset, "__len__"):
                dt_s -= datetime.timedelta(hours=time_offset[o])  # type: ignore
                dt_e += datetime.timedelta(hours=time_offset[o])  # type: ignore
            else:
                dt_s -= datetime.timedelta(hours=time_offset)  # type: ignore
                dt_e += datetime.timedelta(hours=time_offset)  # type: ignore
            # The observer object is created
                        
            observer_obj = getattr(heliosat, observer)()
            
            # The according magnetic field data 
            # for the fitting points is obtained
            
            _, data = observer_obj.get(dt, "mag", reference_frame=self.reference_frame, use_cache=True, **kwargs)
            
            dt_all = [dt_s] + dt + [dt_e] # all time points
            trajectory = observer_obj.trajectory(dt_all, reference_frame=self.reference_frame) # returns the spacecraft trajectory
            # an array containing the data plus one additional 
            # zero each at the beginning and the end is created
            
            b_all = np.zeros((len(data) + 2, 3))
            b_all[1:-1] = data
            
            # the mask is created, a list containing 1] for each 
            # data point and 0 for the first and last entry
            mask = [1] * len(b_all)
            mask[0] = 0
            mask[-1] = 0

            if dt_shift:
                self.data_dt.extend([_ + dt_shift for _ in dt_all])
            else:
                self.data_dt.extend(dt_all)
            self.data_b.extend(b_all)
            self.data_o.extend(trajectory)
            self.data_m.extend(mask)
            self.data_l.append(len(data))

    def sumstat(self, values: np.ndarray, stype: str = "norm_rmse", use_mask: bool = True) -> np.ndarray:   
        
        """
        Returns the summary statistic comparing given values to the data object.

        Arguments:
            values                   fitted values to compare with the data  
            stype      "norm_rmse"   method to use for the summary statistic
            use_mask   True          mask the data

        Returns:
            sumstat                  Summary statistic for each observer
        """
    
        if use_mask:
            return sumstat(values, self.data_b, stype, mask=self.data_m, data_l=self.data_l, length=self.length)
        else:
            return sumstat(values, self.data_b, stype, data_l=self.data_l, length=self.length)

            

class custom_observer(object):
    
    """Handles custom data and sets the following attributes for self:
            data         full custom dataset

        Arguments:
            data_path    where to find the data
            kwargs       any

        Returns:
            None
        """
    
    def __init__(self, data_path:str, **kwargs: Any) -> None:
        
        try:
            file = pickle.load(open('py3dcore_h4c/custom_data/'+ data_path, 'rb'))
            self.data = file
            self.sphere2cart()
        except:
            logger.info("Did not find %s, creating pickle file from cdf", data_path)
            #try:
            createpicklefiles(self,data_path)
            file = pickle.load(open('py3dcore_h4c/custom_data/'+ data_path, 'rb'))
            self.data = file
            #except:
             #   raise NameError('Datatype not implemented or cdf files not found!')
        
        
    def sphere2cart(self):
        self.data['x'] = self.data['r'] * np.cos(np.deg2rad(self.data['lon'])) * np.cos(np.deg2rad(self.data['lat']))
        #print(self.data['x'])
        self.data['y'] = self.data['r'] * np.sin(np.deg2rad(self.data['lon'] )) * np.cos( np.deg2rad(self.data['lat'] ))
        self.data['z'] = self.data['r'] * np.sin(np.deg2rad( self.data['lat'] ))
    
        
    def get(self, dtp: Union[str, datetime.datetime, Sequence[str], Sequence[datetime.datetime]], data_key: str, **kwargs: Any) -> np.ndarray:
        
        sampling_freq = kwargs.pop("sampling_freq", 60)
        
        if kwargs.pop("as_endpoints", False):
            _ = np.linspace(dtp[0].timestamp(), dtp[-1].timestamp(), int((dtp[-1].timestamp() - dtp[0].timestamp()) // sampling_freq))  
            dtp = [datetime.datetime.fromtimestamp(_, datetime.timezone.utc) for _ in _]
            
        dat = []
        tt = [x.replace(tzinfo=None,second=0, microsecond=0) for x in dtp]
        
        ii = [np.where(self.data['time']==x)[0][0] for x in tt if np.where(self.data['time']==x)[0].size > 0]
        
        for t in ii:
            res = [self.data[com][t] for com in ['bx','by','bz']]
            dat.append((res))
            
        return np.array(dtp), np.array(dat)

    
    def trajectory(self, dtp: Union[str, datetime.datetime, Sequence[str], Sequence[datetime.datetime]], **kwargs: Any) -> np.ndarray:
        
        tra = []
        tt = [x.replace(tzinfo=None,second=0, microsecond=0) for x in dtp]

        ii = [np.where(self.data['time']==x)[0][0] for x in tt if np.where(self.data['time']==x)[0].size > 0]
        
        for t in ii:
            res = [self.data[com][t] for com in ['x','y','z']]
            tra.append(res)
            
        return np.array(tra)
        
def createpicklefiles(self, data_path):
    name = data_path.split('.')[0]
    sc = name.split('_')[0]
    ev = name.split('_')[1]

    magpath = 'py3dcore_h4c/custom_data/' + sc +'_mag_'+ ev
    swapath = 'py3dcore_h4c/custom_data/' + sc +'_swa_'+ ev

    ll = cdftopickle(magpath, swapath, sc)

    filename= sc +'_'+ ev + '.p'

    pickle.dump(ll, open('py3dcore_h4c/custom_data/' + filename, "wb"))
    logger.info("Created pickle file from cdf: %s", filename)
                
        
class CustomData(FittingData):
    """
    Class(object) to handle custom data used for fitting.
    Sets the following properties for self:
        length                  length of list of observers
        observers               list of observers
        reference_frame         reference frame to work in
    
    Arguments:
        observers               list of observers
        reference_frame         reference frame to work in        
        data_path               where to find the dataset
        
    Returns:
        None
        
    Functions:
        add_noise
        generate_noise
        generate_data
        sumstat
    """
    
    data_dt: List[np.ndarray]
    data_b: List[np.ndarray]
    data_o: List[np.ndarray]
    data_m: List[np.ndarray]
    data_l: List[int]

    psd_dt: List[np.ndarray]
    psd_fft: List[np.ndarray]    

    length: int
    noise_model: str
    observers: list
    reference_frame: str
    sampling_freq: int

    def __init__(self, observers: list, reference_frame: str, data_path: str) -> None:
        
        FittingData.__init__(self, observers, reference_frame)        
        self.data_path = data_path
    
    def generate_data(self, time_offset: Union[int, Sequence], **kwargs: Any) -> None:
        
        """
        Generates data for each observer at the given times. 
        Sets the following properties for self:
            data_dt      all needed timesteps [dt_s, dt, dt_e]
            data_b       magnetic field data for data_dt
            data_o       trajectory of observers
            data_m       mask for data_b with 1 for each point except first and last
            data_l       length of data

        Arguments:
            time_offset  shift timeseries for observer   
            **kwargs     Any

        Returns:
            None
        """
        
        self.data_dt = []
        self.data_b = []
        self.data_o = []
        self.data_m = []
        self.data_l = []
        
        # Each observer is treated separately

        for o in range(self.length):
            
            # The values of the observer are unpacked
            
            observer, dt, dt_s, dt_e, dt_shift = self.observers[o]
            
            # The reference points are corrected by time_offset

            if hasattr(time_offset, "__len__"):
                dt_s -= datetime.timedelta(hours=time_offset[o])  # type: ignore
                dt_e += datetime.timedelta(hours=time_offset[o])  # type: ignore
            else:
                dt_s -= datetime.timedelta(hours=time_offset)  # type: ignore
                dt_e += datetime.timedelta(hours=time_offset)  # type: ignore
            # The observer object is created
                        
            observer_obj = custom_observer(self.data_path)
            
            # The according magnetic field data 
            # for the fitting points is obtained
            
            _, data = observer_obj.get(dt, "mag", reference_frame=self.reference_frame, use_cache=True, **kwargs)
            
            dt_all = [dt_s] + dt + [dt_e] # all time points
            trajectory = observer_obj.trajectory(dt_all, reference_frame=self.reference_frame) # returns the spacecraft trajectory
            # an array containing the data plus one additional 
            # zero each at the beginning and the end is created
            
            b_all = np.zeros((len(data) + 2, 3))
            b_all[1:-1] = data
            
            # the mask is created, a list containing 1] for each 
            # data point and 0 for the first and last entry
            mask = [1] * len(b_all)
            mask[0] = 0
            mask[-1] = 0

            if dt_shift:
                self.data_dt.extend([_ + dt_shift for _ in dt_all])
            else:
                self.data_dt.extend(dt_all)
            self.data_b.extend(b_all)
            self.data_o.extend(trajectory)
            self.data_m.extend(mask)
            self.data_l.append(len(data))

    def generate_noise(self, noise_model: str = "psd", sampling_freq: int = 300, **kwargs: Any) -> None:
        

        """
        Generates noise according to the noise model.
        Sets the following properties for self:
            psd_dt                altered time axis for power spectrum
            psd_fft               power spectrum
            sampling_freq         sampling frequency
            noise_model           model used to calculate noise

        Arguments:
            noise_model    "psd"     model to use for generating noise (e.g. power spectrum distribution)
            sampling_freq  300       sampling frequency of data

        Returns:
            None
        """
            
        self.psd_dt = []
        self.psd_fft = []
        self.sampling_freq = sampling_freq

        self.noise_model = noise_model

        if noise_model == "psd":
        # get data for each observer
            for o in range(self.length):
                observer, dt, dt_s, dt_e, _ = self.observers[o]

                 # The observer object is created
                        
                observer_obj = custom_observer(self.data_path)
            
                # The according magnetic field data 
                # for the fitting points is obtained
            
                _, data = observer_obj.get([dt_s, dt_e], "mag", reference_frame=self.reference_frame, sampling_freq=sampling_freq, use_cache=True, as_endpoints=True)
                
                data[np.isnan(data)] = 0 #set all nan values to 0

                fF, fS = mag_fft(dt, data, sampling_freq=sampling_freq) # computes the mean power spectrum distribution

                kdt = (len(fS) - 1) / (dt[-1].timestamp() - dt[0].timestamp()) 
                fT = np.array([int((_.timestamp() - dt[0].timestamp()) * kdt) for _ in dt]) 

                self.psd_dt.append(fT) # appends the altered time axis
                self.psd_fft.append(fS)
                # appends the power spectrum 
        else:
            raise NotImplementedError
        
        
        
        
        
 #       (dt, "mag", reference_frame=self.reference_frame, use_cache=True, **kwargs)