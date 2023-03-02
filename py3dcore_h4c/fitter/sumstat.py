# -*- coding: utf-8 -*-

"""sumstat.py

Implements functions for ABC-SMC summary statistics
"""

import numpy as np
import numba as numba

from typing import Any, Optional


def sumstat(values: np.ndarray, reference: np.ndarray, stype: str = "norm_rmse", **kwargs: Any) -> np.ndarray:
    
    """
    Returns the summary statistic comparing given values to reference.
    
    Arguments:
        values                   fitted values
        reference                data
        stype      "norm_rmse"   method to use
        **kwargs   Any
        
    Returns:
        rmse_all.T
    """
    
    if stype == "norm_rmse":
        data_l = np.array(kwargs.pop("data_l")) # length of data        
        length = kwargs.pop("length") # length of observer list
        mask = kwargs.pop("mask", None)

        varr = np.array(values) # array of values

        rmse_all = np.zeros((length, varr.shape[1])) #stores the rmse for all observers in one array. Gets initialized with zeros

        _dl = 0 
        
        # iterate through all observers
        
        for i in range(length):
            slc = slice(_dl, _dl + data_l[i] + 2) #create slice object
            # for _dl = 0: slice(0,length of data for obs1+2)
            
            values_i = varr[slc] # value array of slice
            # values only for current observer
            
            reference_i = np.array(reference)[slc]
            # reference only for current observer

            normfac = np.mean(np.sqrt(np.sum(reference_i**2, axis=1)))
            #normfactor is created for current reference 

            if mask is not None:
                mask_i = np.array(mask)[slc] 
                # mask for current observer is used
            else:
                mask_i = None
                
            # rmse is calculated for the current observer 
            # and added to rmse_all    
            rmse_all[i] = rmse(values_i, reference_i, mask=mask_i) / normfac

            _dl += data_l[i] + 2 # dl is changed to obtain the correct data for each observer

        return rmse_all.T
    else:
        raise NotImplementedError

@numba.njit
def rmse(values: np.ndarray, reference: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    rmse = np.zeros(len(values[0]))
    """
    Returns the rmse of values to reference.

    Arguments:
        values       fitted values
        reference    data
        mask

    Returns:
        rmse         root mean squared error
    """     

    if mask is not None:
        for i in range(len(reference)):
            # compute the rmse for each value
            _error_rmse(values[i], reference[i], mask[i], rmse)
            
        #computes the mean for the full array
        rmse = np.sqrt(rmse / len(values))
    
        mask_arr = np.copy(rmse)
        
        #check if magnetic field at reference points is 0
        for i in range(len(reference)):
            _error_mask(values[i], mask[i], mask_arr)

        return mask_arr
    else:
        for i in range(len(reference)):
            # compute the rmse for each value
            _error_rmse(values[i], reference[i], 1, rmse)
            
        #computes the mean for the full array
        rmse = np.sqrt(rmse / len(values))

        return rmse


@numba.njit
def _error_mask(values_t: np.ndarray, mask: np.ndarray, rmse: np.ndarray) -> None:
    """
    Sets the rmse to infinity if reference points have nonzero magnetic field.
    """
    for i in numba.prange(len(values_t)):
        _v = np.sum(values_t[i]**2)
        if (_v > 0 and mask == 0) or (_v == 0 and mask != 0):
            rmse[i] = np.inf


@numba.njit
def _error_rmse(values_t: np.ndarray, ref_t: np.ndarray, mask: np.ndarray, rmse: np.ndarray) -> None:
    
    """
    Returns the rmse of values to reference.
    """
    
    for i in numba.prange(len(values_t)):
        if mask == 1:
            rmse[i] += np.sum((values_t[i] - ref_t)**2)
