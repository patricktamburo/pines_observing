import matplotlib
matplotlib.use('MacOSX')
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
import datetime
import pdb
import sys
import argparse
import os
from dateutil import tz

'''Author: Patrick Tamburo, Boston University, v1 Jun. 4 2020 
   Purpose: Does a quick reduction and image display of PINES images. 
   Calling sequence: If you want to look at test.fits from tonight's data directory, simply type python3 PINES_quicklook.py from the command line.
       To specify other dates or files, do python3 PINES_quicklook.py --utdate YYYYMMDD --filename 'filename.fits'

'''
def PINES_quicklook(ut_date, file_name):
    calibration_path = '/Users/obs72/Desktop/PINES_scripts/Calibrations/'
    file_path = '/mimir/data/obs72/'+ut_date+'/'+file_name
    if os.path.exists(file_path):
        header = fits.open(file_path)[0].header
        band = header['FILTNME2']
        exptime = str(header['EXPTIME'])
        flat_path = calibration_path+'Flats/master_flat_'+band+'.fits'
        if os.path.exists(flat_path):
            flat = fits.open(flat_path)[0].data
        else:
            print('ERROR: No ',band,'-band flat exists in ',calibration_path,'Flats/...make one.')
            return
        
        dark_path = calibration_path+'Darks/master_dark_'+exptime+'.fits'
        if os.path.exists(dark_path):
            dark = fits.open(dark_path)[0].data
        else:
            print('ERROR: No ',exptime,'-s dark exists in ',calibration_path,'Darks/...make one.')
            return

        ut_str = header['DATE-OBS'].split('T')[0] + ' ' + header['DATE-OBS'].split('T')[1].split('.')[0]
        from_zone = tz.gettz('UTC')
        to_zone = tz.gettz('America/Phoenix')
        utc = datetime.datetime.strptime(ut_str, '%Y-%m-%d %H:%M:%S')
        utc = utc.replace(tzinfo=from_zone)
        local = utc.astimezone(to_zone)
        local_str = local.strftime('%Y-%m-%d %H:%M:%S')
        
        raw_image = fits.open(file_path)[0].data[0:1024,:]
        reduced_image = (raw_image - dark) / flat
        avg, med, std = sigma_clipped_stats(reduced_image)

        

        fig, ax = plt.subplots(figsize=(9,9))        
        ax.set_aspect('equal')
        ax.imshow(reduced_image, origin='lower', vmin=med, vmax=med+5*std)
        ax.set_title('File: '+file_path+'\n UT: '+ut_str+', Local: '+local_str+'\n Band: '+band+', Exptime: '+exptime+' s')
        plt.show()
    else:
        print('ERROR: file ',file_path,' does not exist.')
        return 





if __name__ == '__main__':
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--utdate')
    parser.add_argument('--filename')
    args = parser.parse_args()

    #If no argument is passed for utdate, default to today's ut date.
    if args.utdate is None:
        args.utdate = date_string

    #If no argument is passed for filename, default to test.fits.
    if args.filename is None:
        args.filename = 'test.fits'

    args.utdate = str(args.utdate)
    args.filename = str(args.filename)
    
    PINES_quicklook(args.utdate, args.filename)
