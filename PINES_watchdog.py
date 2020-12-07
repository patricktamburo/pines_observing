import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import sys, os.path, time, logging
from astropy.io import fits
from watchdog.observers.polling import PollingObserver
from watchdog.events import PatternMatchingEventHandler
import numpy as np
import pickle
from astropy.io import fits
import pdb
from pathlib import Path
from scipy import signal
import time
import datetime
from image_shifting_functions import *
from logging_functions import *
import pandas as pd

class MyEventHandler(PatternMatchingEventHandler):
    def __init__(self, directory,  *args, **kwargs):
        super(MyEventHandler, self).__init__(*args, **kwargs)
        if os.path.isfile(directory+'analyzed_files.p'):
            self.already_created = pickle.load(open(directory+'analyzed_files.p','rb'))
        else:
            self.already_created = []

    #Can also have on_moved(self,event) and on_deleted(self,event)...
    def on_created(self, event):
        global daostarfinder_fwhm
        #Activated when a new yyyymmdd.###.fits file is written.
        directory = os.path.dirname(event.src_path)+'/'
        filename = event.src_path.split('/')[-1]
        file_size = os.stat(event.src_path).st_size
        while file_size < 4213440:
           #Let the whole file read out...
            time.sleep(0.1)
            file_size = os.stat(event.src_path).st_size
        
        super(MyEventHandler, self).on_created(event)

        #Check if this file has already been analyzed...sometimes Watchdog/Saturn bugs out, and suddenly sees all of the 
        #previous files from tonight in random order. 
        if event.src_path not in self.already_created:
            self.already_created.append(event.src_path)
            logging.info("%s created." % event.src_path)
            #Calculate shift from master image (for guiding purposes). 

            #Search for a matching maser_dark file.
            exptime = fits.open(directory+filename)[0].header['EXPTIME']
            dark_path = calibration_path+'Darks/master_dark_'+str(exptime)+'.fits'
            if os.path.isfile(dark_path):
                dark = fits.open(dark_path)[0].data
            else:
                print('No master_dark file found matching the exposure time of '+str(exptime)+' seconds in '+calibration_path+'.')
                print('You need to make one!')
            if exptime >= 1:
                (x_shift,y_shift,x_seeing,y_seeing) = image_shift_calculator(lines, master_coordinates, dark, flat, bpm, daostarfinder_fwhm, directory=directory,filename=filename)
            else:
                print('EXPTIME < 1, not measuring shifts.')
                x_shift = 0
                y_shift = 0
                x_seeing = 0
                y_seeing = 0
            #print('Logging.')
            PINES_logger(x_shift,y_shift,x_seeing,y_seeing,master_coordinates,lines,directory=directory,filename=filename, mode='created')
            
            #To speed things up, PINES_guide.tcl will read average x/y shifts every three images, without having to wait for watchdog to analyze things. 
            log_filename = directory+directory.split('/')[-2]+'_log.txt'
            image_shift_filename = directory+'image_shift.txt'
            #Read in the night's log.
            df = pd.read_csv(log_filename, names=['Filename', 'Date', 'Target', 'Filt.', 'Exptime', 'Airmass', 'X shift', 'Y shift', 'X seeing', 'Y seeing'], comment='#', header=None)

            #TODO: This may break if watchdog is running while taking darks/flats, test this behavior.
            #If there are less than three measurements, just append 0s to image_shift.txt. 
            if len(df['X shift']) < 3:
                with open(image_shift_filename, 'a') as f:
                    f.write('\n')
                    f.write('0 0')
            else:
                #Otherwise, write the median of the last three shifts in x/y to image_shift.txt.
                last_three_x = np.array([float(i) for i in df['X shift'][-3:]])
                last_three_y = np.array([float(i) for i in df['Y shift'][-3:]])
                
                med_x_shift = np.nanmedian(last_three_x) #Use nanmedian in case any of the last three measurements were nans.
                med_y_shift = np.nanmedian(last_three_y)
                
                print('')
                print('Last 3 x shifts: {}, median = {:3.1f} pix ({:3.1f}")'.format(last_three_x, med_x_shift, med_x_shift*0.579))
                print('Last 3 y shifts: {}, median = {:3.1f} pix ({:3.1f}")'.format(last_three_y, med_y_shift, med_y_shift*0.579))
                
                #Make sure the median is not a nan, and that the shift is not too large. Shouldn't be > 100 pixels if PINES_peakup worked properly.
                if np.isnan(med_x_shift) or np.isnan(med_y_shift) or (abs(med_x_shift) > 100) or (abs(med_y_shift) > 100):
                    with open(image_shift_filename, 'a') as f:
                        print('Got 3 nans or median shift > 100 pixels, returning 0 shifts.')
                        f.write('\n')
                        f.write('0 0')
                else:
                    with open(image_shift_filename, 'a') as f:
                        f.write('\n')
                        f.write('{} {}'.format(np.round(med_x_shift*0.579,1), np.round(med_y_shift*0.579,1)))
                            
            #Update source detection fwhm value as night progresses.
            if len(df['X seeing']) >= 3:
                last_three_seeing = np.array([float(i) for i in df['X seeing'][-3:]])
                avg_seeing = np.nanmean(last_three_seeing)
                
                print('Last 3 seeing FWHMs: {}, average = {:1.1f}"'.format(last_three_seeing, avg_seeing))

                if (not np.isnan(avg_seeing)) and (avg_seeing != 0):
                    #print('Updating source detection seeing to {}".'.format(np.round(avg_seeing,1)))
                    daostarfinder_fwhm = avg_seeing*2.355/0.579

            print('')
            pickle.dump(self.already_created,open(directory+'analyzed_files.p','wb'))

        
    def on_modified(self, event):
        #Activated when a new test.fits file is written. 
        directory = os.path.dirname(event.src_path)+'/'
        filename = event.src_path.split('/')[-1]
        file_size = os.stat(event.src_path).st_size
        if (file_size != 0) and (filename == 'test.fits'):
            #Search for a matching master_dark file. 
            exptime = fits.open(directory+filename)[0].header['EXPTIME']
            dark_path = calibration_path+'Darks/master_dark_'+str(exptime)+'.fits'
            if os.path.isfile(dark_path):
                dark = fits.open(dark_path)[0].data
            else:
                print('No master_dark file found matching the exposure time of '+str(exptime)+' seconds in '+calibration_path+'.')
                print('You need to make one!')
            super(MyEventHandler, self).on_modified(event)
            logging.info("%s modified." % event.src_path)
            #Calculate shift between test and master image. 
            if exptime >= 1:
                (x_shift,y_shift,x_seeing,y_seeing) = image_shift_calculator(lines, master_coordinates, dark, flat, bpm, daostarfinder_fwhm, directory=directory,filename=filename)
                print('')
            else:
                print('EXPTIME < 1, not measuring shifts.')


#By default, point to today's date directory.
ut_date = datetime.datetime.utcnow()
if ut_date.month < 10:
    month_string = '0'+str(ut_date.month)
else:
    month_string = str(ut_date.month)

if ut_date.day < 10:
    day_string = '0'+str(ut_date.day)
else:
    day_string = str(ut_date.day)

date_string = str(ut_date.year)+month_string+day_string

#Read in list of master images.
file = open('input_file.txt', 'r')
global lines, master_coordinates
lines = file.readlines()
file.close()
master_coordinates = []
target_names = []

#Loop over master files and get the telescope coordinates. These will be checked against when a new test.fits image loaded in to figure out
#   which object you're looking at!
for i in range(len(lines)):
    try:
        master_image_path = Path(lines[i].split(', ')[0])
        image_header = fits.open(master_image_path)[0].header
        target_names.append(lines[i].split(', ')[-1].split('\n')[0])
        #Grab image coordinates of master images. When a new image is read in, its coordinates will be compared against this list. The coordinates
        #that most closely match the check image's coordinates is chosen as appropriate the master image. 
        #Convert coordinates to decimal
        ra  = [float(i) for i in image_header['TELRA'].split(':')]
        dec = [float(i) for i in image_header['TELDEC'].split(':')]
        master_coordinates.append((15*ra[0]+15*ra[1]/60+15*ra[2]/3600, dec[0]+dec[1]/60+dec[2]/3600))
    except:
        print(lines[i],' no longer on disk.')
        print('')
print('')
print('~/Desktop/PINES_scripts/master_images/ has masters for ',len(master_coordinates),' targets.')

#Read in basic calibration data, for image shifting/guiding. 
global dark, flat, bpm, calibration_path
calibration_path = '/Users/obs72/Desktop/PINES_scripts/Calibrations/'
flat_path = calibration_path+'Flats/master_flat_J.fits'
flat = fits.open(flat_path)[0].data[0:1024,:]
###bpm_path = calibration_path+'Bad_pixel_masks/bpm.p'
###bpm = (1-pickle.load(open(bpm_path,'rb'))).astype('bool')
bpm_path = calibration_path+'Bad_pixel_masks/bpm.fits'
bpm = fits.open(bpm_path)[0].data

def PINES_watchdog(date=date_string,seeing=2.3):
    '''PURPOSE:
            Monitors a data directory for new .fits images. 
        INPUTS:
            date: a string of the target directory's UT date in yyyymmdd format. By default, set to today's UT date. 
            seeing: an estimate of the seeing, in arcseconds, used to set the fwhm value used in the source detection. 
                This value will (eventually) be updated throughout the night as seeing measurements are recorded in the log. 
                By default, set to 2". 
    '''

    #Set the fwhm for finding sources in the images with DAOstarfinder. As a rule, a good guess is 
    #2.355 (or 2.6? It's the fwhm to sigma factor, Wikipedia lists two of them) * average seeing (as reported in the log) / 0.579 (Mimir's plate scale). 2.6 is the conversion of fwhm to sigma. 
    global daostarfinder_fwhm
    daostarfinder_fwhm = seeing*2.355/0.579 
    print('')
    file_path = '/mimir/data/obs72/' + date + '/'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    watched_dir = os.path.split(file_path)[0]+'/'
    print('Monitoring {watched_dir}'.format(watched_dir=watched_dir),'for new *.fits files.')
    print('')
    patterns = [file_path+'*.fits']
    event_handler = MyEventHandler(directory=file_path,patterns=patterns)
    observer = PollingObserver()
    observer.schedule(event_handler, watched_dir, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

