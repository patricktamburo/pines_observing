import pandas as pd
import os
import pdb
import numpy as np
import math

def get_seeing(daostarfinder_fwhm):
    '''Purpose: Reads the log and takes the median of the previous 5 seeing measurements to update daostarfinder_fwhm.'''
    number_to_average = 5 #Number of seeing measurements to use when calculating median
    path_to_log = '20200304_log.txt' #Will have to update with actual log path

    if os.path.exists(path_to_log): #Check if the log exists
        log_data = pd.read_csv(path_to_log) #Read in log data as pandas dataframe
        if len(log_data) > number_to_average: #Check if there are more than number_to_average lines
            x_seeing = np.array([float(i) for i in log_data[' X seeing']]) #Grabs X and Y seeing values as numpy float arrays.
            y_seeing = np.array([float(i) for i in log_data[' Y seeing']]) #Since they're saved as strings, have to convert them to floats!
            current_line = len(log_data)-1 #Take number_to_average measurements before current_line. 
            recent_x = x_seeing[current_line-number_to_average:current_line] #Grab the last number_to_average x seeing measurements. 
            recent_y = y_seeing[current_line-number_to_average:current_line] #Do the same for y. 
            recent_combined = []
            recent_combined.extend(recent_x) #Read x and y seeing measurements into one array
            recent_combined.extend(recent_y)
            recent_combined = np.array(recent_combined) #Convert it from a Python list to a numpy array (so we can do np.nanmedian on it).
            median_seeing = np.nanmedian(recent_combined) #Take the median of the last number_to_average x and y seeing measurements. This ignores nans!
            if math.isnan(median_seeing): #Check if the calculated median seeing is a nan, which should only happen if all the recent_combined measurements just happen to be nans (i.e., shouldn't happen.)
                return(daostarfinder_fwhm) #If so, just return the original value.
            else:
                daostarfinder_fwhm = median_seeing * fwhm_to_sigma / plate_scale
                return(daostarfinder_fwhm)
        else:
            return(daostarfinder_fwhm) #If < num_to_average lines, return the current value unchanged.
    else:
        return(daostarfinder_fwhm) #If no log file, return the current value unchanged. 

plate_scale = 0.579 #Mimir's plate scale ["/pix]
fwhm_to_sigma = 2*np.sqrt(2*np.log(2)) #FWHM to sigma conversion factor (is this 2.35 or 2.6? Wiki has two values.)
current_seeing = 1.5
daostarfinder_fwhm = current_seeing*fwhm_to_sigma/plate_scale #Set the daostarfinder_fwhm at the start of the night based on seeing
print('Original daostarfinder_fwhm is ',daostarfinder_fwhm)
daostarfinder_fwhm = get_seeing(daostarfinder_fwhm) #Update the current value using the log. 
print('Updated daostarfinder_fwhm is ',daostarfinder_fwhm)
