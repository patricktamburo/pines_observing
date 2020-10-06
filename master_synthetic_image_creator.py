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
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats
import pickle 
from pathlib import Path
import os 
import time
import datetime
import shutil
import sys

def master_synthetic_image_creator(target,daostarfinder_fwhm = 8):

    def mimir_source_finder(image_path,sigma_above_bg,fwhm):
        #Find sources in Mimir images. 

        np.seterr(all='ignore') #Ignore invalids (i.e. divide by zeros)


        #Find stars in the master image.
        avg, med, stddev = sigma_clipped_stats(image,sigma=3.0,maxiters=3) #Previously maxiters = 5!   
        daofind = DAOStarFinder(fwhm=fwhm, threshold=sigma_above_bg*stddev,sky=med,ratio=0.8)
        new_sources = daofind(image,mask=bpm)
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

        #Also cut for sharpnesses > 0.4, this seems to eliminate a lot of false detections.
        use_sharp = np.where(sharpness > 0.5)[0]
        x_centroids  = x_centroids [use_sharp]
        y_centroids  = y_centroids [use_sharp]
        sharpness = sharpness[use_sharp]
        fluxes = fluxes[use_sharp]
        peaks = peaks[use_sharp]

        #Finally, cut targets whose y centroids are near y = 512. These are usually bad.
        use_512 = np.where(np.logical_or((y_centroids < 510),(y_centroids > 514)))[0]
        x_centroids  = x_centroids [use_512]
        y_centroids  = y_centroids [use_512]
        sharpness = sharpness[use_512]
        fluxes = fluxes[use_512]
        peaks = peaks[use_512]

        if len(peaks) > 15: #Take the 15 brightest.
            brightest = np.argsort(peaks)[::-1][0:15]
            x_centroids = x_centroids[brightest]
            y_centroids = y_centroids[brightest]
            sharpness = sharpness[brightest]
            fluxes = fluxes[brightest]
            peaks = peaks[brightest]
        return(x_centroids,y_centroids,fluxes,sharpness)

    def synthetic_image_maker(x_centroids,y_centroids,fluxes,fwhm):
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

    ### target = '2MASSJ12345678+1234567' #Testing.
    target = target.replace(' ','')
    daostarfinder_fwhm = float(daostarfinder_fwhm)
    ### daostarfinder_fwhm = 8

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
    ### date_string = '20200506' #Testing

    #Copy the test.fits file to the master_images directory in PINES scripts.
    test_path =  '/mimir/data/obs72/'+date_string+'/test.fits'
    target_path = '/Users/obs72/Desktop/PINES_scripts/master_images/'+target+'_master.fits'
    shutil.copyfile(test_path, target_path)

    file_path = '/Users/obs72/Desktop/PINES_scripts/master_images/'+target+'_master.fits'
    calibration_path = '/Users/obs72/Desktop/PINES_scripts/Calibrations/'
    header = fits.open(file_path)[0].header
    exptime = header['EXPTIME']
    filter = header['FILTNME2']
    image = fits.open(file_path)[0].data[0:1024,:]
    dark = fits.open(calibration_path+'Darks/master_dark_'+str(exptime)+'.fits')[0].data
    flat = fits.open(calibration_path+'Flats/master_flat_'+filter+'.fits')[0].data
    bpm_path = calibration_path+'Bad_pixel_masks/bpm.p'
    bpm = (1-pickle.load(open(bpm_path,'rb'))).astype('bool')


    image = (image-dark)/flat
    avg,med,std = sigma_clipped_stats(image)

    (x_centroids,y_centroids,fluxes,sharpness) = mimir_source_finder(image,sigma_above_bg=3,fwhm=daostarfinder_fwhm)
    plt.ion()
    plt.figure(figsize=(9,7))
    plt.imshow(image,origin='lower',vmin=med,vmax=med+5*std)
    plt.plot(x_centroids,y_centroids,'rx')
    for i in range(len(x_centroids)):
        plt.text(x_centroids[i]+8,y_centroids[i]+8,str(i),color='r')
    plt.title('Inspect to make sure stars were found!\nO for magnification tool, R to reset view')
    plt.show()

    print('')
    print('')
    print('')

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

    synthetic_image = synthetic_image_maker(x_centroids[ids_to_keep],y_centroids[ids_to_keep],fluxes,8)
    plt.figure(figsize=(9,7))
    plt.imshow(synthetic_image,origin='lower')
    plt.title('Synthetic image')
    plt.show()

    print('')
    print('')
    print('')
    #Now write to a master synthetic image.fits file.
    hdu = fits.PrimaryHDU(synthetic_image)
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
