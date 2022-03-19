import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.ion()
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel
from astropy.visualization import ImageNormalize, ZScaleInterval
import datetime
import pdb
import sys
import argparse
import os
from dateutil import tz
from glob import glob

'''Author: Patrick Tamburo, Boston University, v1 Jun. 4 2020 
   Purpose: Does a quick reduction and image display of PINES images. 
   Calling sequence: If you want to look at test.fits from tonight's data directory, simply type python3 PINES_quicklook.py from the command line.
       To specify other dates or files, do python3 PINES_quicklook.py --utdate YYYYMMDD --filename 'filename.fits'

'''
def PINES_quicklook(image_name='test.fits', interp=True):
    calibration_path = '/Users/obs72/Desktop/PINES_scripts/Calibrations/'
    if image_name == 'test.fits':
       file_path = '/Users/obs72/Desktop/PINES_scripts/test_image/test.fits'
    else:
       date_string = image_name.split('.')[0]
       file_path = '/mimir/data/obs72/'+date_string+'/'+image_name
       
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
            
        #Select the master dark on-disk with the closest exposure time to exptime. 
        dark_top_level_path = calibration_path+'Darks/'
        dark_files = np.array(glob(dark_top_level_path+'*.fits'))
        dark_exptimes = np.array([float(i.split('master_dark_')[1].split('.fits')[0]) for i in dark_files])
        dark_files = dark_files[np.argsort(dark_exptimes)]
        dark_exptimes = dark_exptimes[np.argsort(dark_exptimes)]
        dark_ind = np.where(abs(dark_exptimes-float(exptime)) == np.min(abs(dark_exptimes-float(exptime))))[0][0]
        dark_path = dark_files[dark_ind]
        dark = fits.open(dark_path)[0].data

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

        if interp:
           bpm = fits.open('/Users/obs72/Desktop/PINES_scripts/Calibrations/Bad_pixel_masks/bpm.fits')[0].data
           reduced_image[bpm == 1] = np.nan
           reduced_image = interpolate_replace_nans(reduced_image, kernel=Gaussian2DKernel(0.5))

        fig, ax = plt.subplots(figsize=(9,8))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        ax.set_aspect('equal')
        norm = ImageNormalize(reduced_image, interval=ZScaleInterval())
        im = ax.imshow(reduced_image, origin='lower', norm=norm)
        fig.colorbar(im, cax=cax, orientation='vertical', label='ADU')
        ax.set_title(file_path.split('/')[-1], fontsize=16)
        plt.tight_layout()
        breakpoint()
        plt.close()
    else:
        print('ERROR: file ',file_path,' does not exist.')
        return 

if __name__ == '__main__':
   file_name = '20220315.100.fits'
   PINES_quicklook(file_name)
