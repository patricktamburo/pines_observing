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

class MyEventHandler(PatternMatchingEventHandler):
    def __init__(self, directory,  *args, **kwargs):
        super(MyEventHandler, self).__init__(*args, **kwargs)
        if os.path.isfile(directory+'analyzed_files.p'):
            self.already_created = pickle.load(open(directory+'analyzed_files.p','rb'))
        else:
            self.already_created = []

        #pdb.set_trace()
    #Can also have on_moved(self,event) and on_deleted(self,event)...
    def on_created(self, event):
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
            logging.info("File %s was just created." % event.src_path)
            #Calculate shift from master image (for guiding purposes). 

            #Search for a matching master_dark file. 
            exptime = fits.open(directory+filename)[0].header['EXPTIME']
            dark_path = calibration_path+'Darks/master_dark_'+str(exptime)+'.fits'
            if os.path.isfile(dark_path):
                dark = fits.open(dark_path)[0].data
            else:
                print('No master_dark file found matching the exposure time of '+str(exptime)+' seconds in '+calibration_path+'.')
                print('You need to make one!')
            #This will crash if you take a calibration image, not science!
            if exptime >= 1:
                (x_shift,y_shift,x_seeing,y_seeing) = image_shift_calculator(lines, master_coordinates, dark, flat, bpm, daostarfinder_fwhm, directory=directory,filename=filename)
            else:
                print('EXPTIME < 1, not measuring shifts.')
                x_shift = 0
                y_shift = 0
                x_seeing = 0
                y_seeing = 0
            print('Logging.')
            PINES_logger(x_shift,y_shift,x_seeing,y_seeing,master_coordinates,lines,directory=directory,filename=filename)
            print('')
            ### print('Hit the on_created loop')
            pickle.dump(self.already_created,open(directory+'analyzed_files.p','wb'))
        else:
            print('Skipped the on_created loop')

        
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
            logging.info("File %s was just modified." % event.src_path)
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
bpm_path = calibration_path+'Bad_pixel_masks/bpm.p'
bpm = (1-pickle.load(open(bpm_path,'rb'))).astype('bool')

def PINES_watchdog(date=date_string,seeing=2):
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
    ### event_handler = MyEventHandler(patterns=patterns)
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

