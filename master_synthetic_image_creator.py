import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import pdb
import scipy.optimize as opt
from scipy import signal
from astropy.modeling import models, fitting
from astropy.io import fits
from photutils import DAOStarFinder, aperture_photometry, CircularAperture, Background2D, MedianBackground
from astropy.stats import sigma_clipped_stats, SigmaClip
import pickle 
from pathlib import Path
import os 
import time
import datetime
import shutil
import sys
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans

def master_synthetic_image_creator(target, seeing=2.5):

    def mimir_source_finder(image_path,sigma_above_bg,fwhm):
        """Find sources in Mimir images."""
        
        np.seterr(all='ignore') #Ignore invalids (i.e. divide by zeros)

        #Find stars in the master image.
        avg, med, stddev = sigma_clipped_stats(image,sigma=3.0,maxiters=3) #Previously maxiters = 5!   
        daofind = DAOStarFinder(fwhm=fwhm, threshold=sigma_above_bg*stddev,sky=med,ratio=0.8)
        new_sources = daofind(image)
        x_centroids = new_sources['xcentroid']
        y_centroids = new_sources['ycentroid']
        sharpness = new_sources['sharpness']
        fluxes = new_sources['flux']
        peaks = new_sources['peak']

        #Cut sources that are found within 20 pix of the edges.
        use_x = np.where((x_centroids > 20) & (x_centroids < 1004))[0]
        x_centroids = x_centroids[use_x]
        y_centroids = y_centroids[use_x]
        sharpness = sharpness[use_x]
        fluxes = fluxes[use_x]
        peaks = peaks[use_x]
        use_y = np.where((y_centroids  > 20) & (y_centroids  < 1004))[0]
        x_centroids  = x_centroids [use_y]
        y_centroids  = y_centroids [use_y]
        sharpness = sharpness[use_y]
        fluxes = fluxes[use_y]
        peaks = peaks[use_y]

        #Also cut using sharpness, this seems to eliminate a lot of false detections.
        use_sharp = np.where(sharpness > 0.5)[0]
        x_centroids  = x_centroids [use_sharp]
        y_centroids  = y_centroids [use_sharp]
        sharpness = sharpness[use_sharp]
        fluxes = fluxes[use_sharp]
        peaks = peaks[use_sharp]

        #Cut sources in the lower left, if bars are present.
        #use_ll =  np.where((x_centroids > 512) | (y_centroids > 512))
        #x_centroids  = x_centroids [use_ll]
        #y_centroids  = y_centroids [use_ll]
        #sharpness = sharpness[use_ll]
        #fluxes = fluxes[use_ll]
        #peaks = peaks[use_ll]
        
        #Cut targets whose y centroids are near y = 512. These are usually bad.
        use_512 = np.where(np.logical_or((y_centroids < 510),(y_centroids > 514)))[0]
        x_centroids  = x_centroids [use_512]
        y_centroids  = y_centroids [use_512]
        sharpness = sharpness[use_512]
        fluxes = fluxes[use_512]
        peaks = peaks[use_512]

        #Cut sources with negative/saturated peaks
        use_peaks = np.where((peaks > 30) & (peaks < 7000))[0]
        x_centroids  = x_centroids [use_peaks]
        y_centroids  = y_centroids [use_peaks]
        sharpness = sharpness[use_peaks]
        fluxes = fluxes[use_peaks]
        peaks = peaks[use_peaks]

        #Do quick photometry on the remaining sources. 
        positions = [(x_centroids[i], y_centroids[i]) for i in range(len(x_centroids))]
        apertures = CircularAperture(positions, r=4)
        phot_table = aperture_photometry(image-med, apertures)

        #Cut based on brightness.
        phot_table.sort('aperture_sum')
        cutoff = 1*std*np.pi*4**2
        bad_source_locs = np.where(phot_table['aperture_sum'] < cutoff)
        phot_table.remove_rows(bad_source_locs)
        
        if len(phot_table) > 100:
            x_centroids = phot_table['xcenter'].value[-101:-1]
            y_centroids = phot_table['ycenter'].value[-101:-1]
        else:
            x_centroids = phot_table['xcenter'].value
            y_centroids = phot_table['ycenter'].value

        return(x_centroids,y_centroids)

    def synthetic_image_maker(x_centroids,y_centroids,fwhm):
        #Construct synthetic images from centroid/flux data.
        synthetic_image = np.zeros((1024,1024))
        sigma = fwhm/2.355
        for i in range(len(x_centroids)):
            #Cut out little boxes around each source and add in Gaussian representations. This saves time. 
            int_centroid_x = int(np.round(x_centroids[i]))
            int_centroid_y = int(np.round(y_centroids[i]))
            y_cut, x_cut = np.mgrid[int_centroid_y-10:int_centroid_y+10,int_centroid_x-10:int_centroid_x+10]
            dist = np.sqrt((x_cut-x_centroids[i])**2+(y_cut-y_centroids[i])**2)
            synthetic_image[y_cut,x_cut] += np.exp(-((dist)**2/(2*sigma**2)+((dist)**2/(2*sigma**2))))
        return(synthetic_image)
    
    plt.ion()

    target = target.replace(' ','')
    seeing = float(seeing)
    daostarfinder_fwhm = seeing*2.355/0.579

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

    #Copy the test.fits file to the master_images directory in PINES scripts.
    #test_path =  '/mimir/data/obs72/'+date_string+'/test.fits'
    test_path = '/Users/obs72/Desktop/PINES_scripts/test_image/test.fits'
    target_path = '/Users/obs72/Desktop/PINES_scripts/master_images/'+target+'_master.fits'
    shutil.copyfile(test_path, target_path)

    
    file_path = '/Users/obs72/Desktop/PINES_scripts/master_images/'+target+'_master.fits'
    calibration_path = '/Users/obs72/Desktop/PINES_scripts/Calibrations/'

    #Open the image and calibration files. 
    header = fits.open(file_path)[0].header
    exptime = header['EXPTIME']
    filter = header['FILTNME2']
    image = fits.open(file_path)[0].data[0:1024,:].astype('float')
    dark = fits.open(calibration_path+'Darks/master_dark_'+str(exptime)+'.fits')[0].data
    flat = fits.open(calibration_path+'Flats/master_flat_'+filter+'.fits')[0].data
    bpm_path = calibration_path+'Bad_pixel_masks/bpm.fits'
    bpm = fits.open(bpm_path)[0].data

    #Reduce image.
    image = (image-dark)/flat

    #Interpolate over bad pixels
    image[np.where(bpm)] = np.nan
    kernel = Gaussian2DKernel(x_stddev=1)
    image = interpolate_replace_nans(image, kernel)

    #Do a simple 2d background model. 
    box_size=32
    sigma_clip = SigmaClip(sigma=3.)
    bkg_estimator = MedianBackground()
    bkg = Background2D(image, (box_size, box_size), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    image = image - bkg.background

    avg,med,std = sigma_clipped_stats(image)

    #Save reduced image to test_image for inspection
    hdu_reduced = fits.PrimaryHDU(image)
    hdu_reduced.writeto('/Users/obs72/Desktop/PINES_scripts/test_image/master_reduced.fits')
    
    #Find sources in the image. 
    (x_centroids,y_centroids) = mimir_source_finder(image,sigma_above_bg=5,fwhm=daostarfinder_fwhm)

    #Plot the field with detected sources. 
    plt.figure(figsize=(9,9))
    plt.imshow(image,origin='lower',vmin=med,vmax=med+4*std)
    plt.plot(x_centroids,y_centroids,'rx')
    for i in range(len(x_centroids)):
        plt.text(x_centroids[i]+8,y_centroids[i]+8,str(i),color='r', fontsize=14)
    plt.title('Inspect to make sure stars were found!\nO for magnification tool, R to reset view')
    plt.tight_layout()
    plt.show()

    print('')
    print('')
    print('')

    #Prompt the user to remove any false detections.
    ids = input('Enter ids of sources to be removed separated by commas (i.e., 4,18,22). If none to remove, hit enter. To break, ctrl + D. ')
    if ids != '':
        ids_to_eliminate = [int(i) for i in ids.split(',')]
        ids = [int(i) for i in np.linspace(0,len(x_centroids)-1,len(x_centroids))]
        ids_to_keep = []
        for i in range(len(ids)):
            if ids[i] not in ids_to_eliminate:
                ids_to_keep.append(ids[i])
    else:
        ids_to_keep = [int(i) for i in np.linspace(0,len(x_centroids)-1,len(x_centroids))]
    plt.clf()
    plt.imshow(image,origin='lower',vmin=med,vmax=med+5*std)
    plt.plot(x_centroids[ids_to_keep],y_centroids[ids_to_keep],'rx')
    for i in range(len(x_centroids[ids_to_keep])):
        plt.text(x_centroids[ids_to_keep][i]+8,y_centroids[ids_to_keep][i]+8,str(i),color='r')

    #Create the synthetic image using the accepted sources. 
    synthetic_image = synthetic_image_maker(x_centroids[ids_to_keep],y_centroids[ids_to_keep],8)
    plt.figure(figsize=(9,7))
    plt.imshow(synthetic_image,origin='lower')
    plt.title('Synthetic image')
    plt.show()

    print('')
    print('')
    print('')
    #Now write to a master synthetic image.fits file.
    hdu = fits.PrimaryHDU(synthetic_image, header=header)
    if os.path.exists('master_images/'+target+'_master_synthetic.fits'):
        ans = input('Master image already exists for target, type y to overwrite. ')
        if ans == 'y':
            os.remove('master_images/'+target+'_master_synthetic.fits')
            hdu.writeto('master_images/'+target+'_master_synthetic.fits')
            print('Writing master synthetic image to master_images/'+target+'_master_synthetic.fits')
        else:
            print('New master synthetic image not saved.')
    else:
        hdu.writeto('master_images/'+target+'_master_synthetic.fits')
        print('Writing master synthetic image to master_images/'+target+'_master_synthetic.fits')

        #Open list of master images and append new one.
        master_image_list = '/Users/obs72/Desktop/PINES_scripts/input_file.txt'
        file_object = open(master_image_list, 'a')
        append_str = '/Users/obs72/Desktop/PINES_scripts/master_images/'+target+'_master.fits, 2MASS '+target.split('2MASS')[1]
        file_object.write('\n')
        file_object.write(append_str)
        file_object.close()

if __name__ == '__main__':
    # Map command line arguments to function arguments.
    master_synthetic_image_creator(*sys.argv[1:])
