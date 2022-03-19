from glob import glob
import numpy as np
from natsort import natsorted
import datetime
import time
import matplotlib.pyplot as plt
plt.ion()
from astropy.io import fits
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel
from astropy.visualization import ImageNormalize, ZScaleInterval
import os

sleep_time = 1
interp = True
ut_date = datetime.datetime.utcnow()
month_string = str(ut_date.month).zfill(2)
day_string = str(ut_date.day).zfill(2)
date_string = str(ut_date.year)+month_string+day_string

data_directory = '/mimir/data/obs72/'+date_string
print('Watching {} for new images'.format(data_directory))

fits_files = np.array(natsorted(glob(data_directory+'/*.fits')))
bpm = fits.open('/Users/obs72/Desktop/PINES_scripts/Calibrations/Bad_pixel_masks/bpm.fits')[0].data

#Get darks on disk.
calibration_path = '/Users/obs72/Desktop/PINES_scripts/Calibrations/'
dark_top_level_path = calibration_path+'Darks/'
dark_files = np.array(glob(dark_top_level_path+'*.fits'))
dark_exptimes = np.array([float(i.split('master_dark_')[1].split('.fits')[0]) for i in dark_files])
dark_files = dark_files[np.argsort(dark_exptimes)]
dark_exptimes = dark_exptimes[np.argsort(dark_exptimes)]

test_path = '/Users/obs72/Desktop/PINES_scripts/test_image/test.fits'
test_mtime = os.stat(test_path).st_mtime

scale=0.9
plt.figure(figsize=(9*scale,8*scale))
fig = plt.gcf()
fig.canvas.set_window_title('PINES data monitor')
plt.title('PINES data monitor\nWaiting for new images', fontsize=16)
plt.pause(0.2)
while True:
    loop_files = np.array(natsorted(glob(data_directory+'/'+date_string+'*.fits')))
    loop_mtime = os.stat(test_path).st_mtime
    
    if loop_mtime != test_mtime:
        plt.clf()
        print('test.fits modified!')
        time.sleep(2) #Allow file to read out
        new_im = fits.open(test_path)[0].data[0:1024,:]
        header = fits.open(test_path)[0].header
        date = header['DATE'].split('T')[1]
    
        #Get the right flat.
        band = header['FILTNME2']
        exptime = str(header['EXPTIME'])
        flat_path = calibration_path+'Flats/master_flat_'+band+'.fits'
        if os.path.exists(flat_path):
            flat = fits.open(flat_path)[0].data
        else:
            print('ERROR: No ',band,'-band flat exists in ',calibration_path,'Flats/...make one.')
        
        #Select the master dark on-disk with the closest exposure time to exptime. d
        dark_ind = np.where(abs(dark_exptimes-float(exptime)) == np.min(abs(dark_exptimes-float(exptime))))[0][0]
        dark_path = dark_files[dark_ind]
        dark = fits.open(dark_path)[0].data

        reduced_image = (new_im - dark) / flat
    
        if interp:
            reduced_image[bpm == 1] = np.nan
            reduced_image = interpolate_replace_nans(reduced_image, kernel=Gaussian2DKernel(0.5))

        norm = ImageNormalize(reduced_image, interval=ZScaleInterval())
        plot_data = plt.imshow(reduced_image, origin='lower', norm=norm)
        cb = plt.colorbar(plot_data)
        cb.set_label('ADU', fontsize=16)
        plt.title('test.fits, UT {}'.format(date), fontsize=16)
        plt.tight_layout()
        plt.pause(header['EXPTIME']+3)

        
        test_mtime = loop_mtime
    
    if len(loop_files) != len(fits_files):
        plt.clf()
        print('{} created!'.format(loop_files[-1]))
        time.sleep(2) #Allow file to read out
        new_file = loop_files[-1]
        new_im = fits.open(new_file)[0].data[0:1024,:]
        header = fits.open(new_file)[0].header
        date = header['DATE'].split('T')[1]
        
        #Get the right flat.
        band = header['FILTNME2']
        exptime = str(header['EXPTIME'])
        flat_path = calibration_path+'Flats/master_flat_'+band+'.fits'
        if os.path.exists(flat_path):
            flat = fits.open(flat_path)[0].data
        else:
            print('ERROR: No ',band,'-band flat exists in ',calibration_path,'Flats/...make one.')
        
        #Select the master dark on-disk with the closest exposure time to exptime. d
        dark_ind = np.where(abs(dark_exptimes-float(exptime)) == np.min(abs(dark_exptimes-float(exptime))))[0][0]
        dark_path = dark_files[dark_ind]
        dark = fits.open(dark_path)[0].data
        
        reduced_image = (new_im - dark) / flat
        
        if interp:
            reduced_image[bpm == 1] = np.nan
            reduced_image = interpolate_replace_nans(reduced_image, kernel=Gaussian2DKernel(0.5))
        
        norm = ImageNormalize(reduced_image, interval=ZScaleInterval())
        plot_data = plt.imshow(reduced_image, origin='lower', norm=norm)
        cb = plt.colorbar(plot_data)
        cb.set_label('ADU', fontsize=16)
        plt.title(header['FILENAME']+', UT {}'.format(date), fontsize=16)
        plt.tight_layout()
        plt.pause(header['EXPTIME']+3)
        fits_files = loop_files
    time.sleep(sleep_time)
    #print('watching...')
