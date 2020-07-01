import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import pdb
import scipy.optimize as opt
from scipy import signal
from astropy.modeling import models, fitting
from astropy.io import fits
from photutils import DAOStarFinder
from astropy.stats import sigma_clipped_stats, sigma_clip
import pickle 
from pathlib import Path
import os
import time
from numpy.lib.stride_tricks import as_strided


def mimir_source_finder(image_path,calibration_path,dark,flat,bpm,sigma_above_bg,fwhm):

    def strided_rescale(g, bin_fac):
        #Function to lower the resolution of images. 
        strided = as_strided(g,
            shape=(g.shape[0]//bin_fac, g.shape[1]//bin_fac, bin_fac, bin_fac),
            strides=((g.strides[0]*bin_fac, g.strides[1]*bin_fac)+g.strides))
        return strided.mean(axis=-1).mean(axis=-1)

    #Find sources in Mimir images. 

    np.seterr(all='ignore') #Ignore invalids (i.e. divide by zeros)

    #Read in image/calibration/bad pixel data.     
    image = fits.open(image_path)[0].data[0:1024,:]

    #Reduce the image. 
    image = (image - dark) / flat

    #Lower the image resolution, to save time doing source detection. 
    binning_factor = 2
    lowres_image = strided_rescale(image,binning_factor)
    lowres_bpm = strided_rescale(bpm,binning_factor).astype('bool')

    #Find stars in the master image. #THIS IS WHERE MOST TIME IS LOST!
    #~4 seconds to run daofind on the image. 
    #Old way: Do the source detection on the full-res image. 
    # t1 = time.time()
    # avg, med, stddev = sigma_clipped_stats(image,sigma=3.0,maxiters=3) #Previously maxiters = 5!   
    # daofind = DAOStarFinder(fwhm=fwhm, threshold=sigma_above_bg*stddev,sky=med,ratio=0.9)
    # new_sources = daofind(image,mask=bpm)
    # t2 = time.time()
    # print('Full res time: ',np.round(t2-t1,1))

    #New way: do source detection on half-res image. 
    #t1 = time.time()
    avg, med, stddev = sigma_clipped_stats(lowres_image,sigma=3.0,maxiters=3) #Previously maxiters = 5!   
    daofind = DAOStarFinder(fwhm=fwhm/binning_factor, threshold=sigma_above_bg*stddev,sky=med,ratio=0.9)
    new_sources = daofind(lowres_image,mask=lowres_bpm)
    #t2 = time.time()
    #print('Half res time: ',np.round(t2-t1,1))

    x_centroids = new_sources['xcentroid']
    y_centroids = new_sources['ycentroid']
    sharpness = new_sources['sharpness']
    fluxes = new_sources['flux']
    peaks = new_sources['peak']

    #Cut sources that are found within 20 pix of the edges.
    use_x = np.where((x_centroids > 20/binning_factor) & (x_centroids < 1004/binning_factor))[0]
    x_centroids = x_centroids[use_x]
    y_centroids = y_centroids[use_x]
    sharpness = sharpness[use_x]
    fluxes = fluxes[use_x]
    peaks = peaks[use_x]

    use_y = np.where((y_centroids  > 20/binning_factor) & (y_centroids  < 1004/binning_factor))[0]
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
   
    #Finally, cut targets whose y centroids are near y = 512 (in full-res images). These are usually bad.
    use_512 = np.where(np.logical_or((y_centroids < 510/binning_factor),(y_centroids > 514/binning_factor)))[0]
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
    
    return(binning_factor*x_centroids,binning_factor*y_centroids,fluxes,sharpness,image)

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

def corr_shift_determination(corr):
    #Measure shift between the check and master images by fitting a 2D gaussian to corr. This gives sub-pixel accuracy. 
    y_max, x_max = np.unravel_index(np.argmax(corr), corr.shape) #Find the pixel with highest correlation, then use this as estimate for gaussian fit.
    y, x = np.mgrid[y_max-10:y_max+10, x_max-10:x_max+10]
    try:
        corr_cut =  corr[y,x]
        gaussian_init = models.Gaussian2D(np.max(corr_cut),x_max,y_max,8/2.355,8/2.355,0)
        fit_gauss = fitting.LevMarLSQFitter()
        gaussian = fit_gauss(gaussian_init, x, y,corr_cut)
        fit_x = gaussian.x_mean.value
        fit_y = gaussian.y_mean.value
        x_shift = fit_x - 512
        y_shift = 512 - fit_y 
        return(x_shift,y_shift)
    except:
        print('Problem with corr indexing, returning 0 shifts.')
        return(0,0)
    

def tie_sigma(model):
    return model.x_stddev_1
        
def guide_star_seeing(subframe):
    # subframe = subframe - np.median(subframe)
    subframe = subframe - np.percentile(subframe,5)
    sub_frame_l = int(np.shape(subframe)[0])
    y, x = np.mgrid[:sub_frame_l, :sub_frame_l]

    # gaussian_init = models.Gaussian2D(subframe[int(sub_frame_l/2),int(sub_frame_l/2)],int(sub_frame_l/2),int(sub_frame_l/2),8/2.355,8/2.355,0)
    # fit_gauss = fitting.LevMarLSQFitter()
    # gaussian = fit_gauss(gaussian_init, x, y, subframe)
    # fwhm_x = 2.355*gaussian.x_stddev.value
    # fwhm_y = 2.355*gaussian.y_stddev.value

    # Fit with constant, bounds, tied x and y sigmas and outlier rejection:
    gaussian_init = models.Const2D(0.0) + models.Gaussian2D(subframe[int(sub_frame_l/2),int(sub_frame_l/2)],int(sub_frame_l/2),int(sub_frame_l/2),8/2.355,8/2.355,0)
    gaussian_init.x_stddev_1.min = 1.0/2.355
    gaussian_init.x_stddev_1.max = 20.0/2.355
    gaussian_init.y_stddev_1.min = 1.0/2.355
    gaussian_init.y_stddev_1.max = 20.0/2.355
    gaussian_init.y_stddev_1.tied = tie_sigma
    gaussian_init.theta_1.fixed = True
    fit_gauss = fitting.FittingWithOutlierRemoval(fitting.LevMarLSQFitter(),sigma_clip,niter=3,sigma=3.0)
    # gaussian, mask = fit_gauss(gaussian_init, x, y, subframe)
    gain = 8.21 #e per ADU
    read_noise = 2.43 #ADU
    weights = gain / np.sqrt(np.absolute(subframe)*gain + (read_noise*gain)**2) #1/sigma for each pixel
    gaussian, mask = fit_gauss(gaussian_init, x, y, subframe, weights)
    fwhm_x = 2.355*gaussian.x_stddev_1.value
    fwhm_y = 2.355*gaussian.y_stddev_1.value

    x_seeing = fwhm_x * 0.579
    y_seeing = fwhm_y * 0.579
    return(x_seeing,y_seeing)

def image_shift_calculator(lines, master_coordinates, dark, flat, bpm, daostarfinder_fwhm,directory = '',filename = '', 
                           calibration_path='/Users/obs72/Desktop/PINES_scripts/Calibrations/',input_file = 'input_file.txt'):
    diagnostic_plot = 0 #Set to true to plot master and check images with sources. 

    #Get the check image and its coordinates
    check_path = Path(directory+filename)
    check_hdul = fits.open(check_path)
    check_header = check_hdul[0].header
    check_exptime = check_header['EXPTIME']
    ra  = [float(i) for i in check_header['TELRA'].split(':')]
    dec = [float(i) for i in check_header['TELDEC'].split(':')]
    check_ra = 15*ra[0]+15*ra[1]/60+15*ra[2]/3600 
    check_dec = dec[0]+dec[1]/60+dec[2]/3600
    master_ra = np.array([master_coordinates[i][0] for i in range(len(master_coordinates))])
    master_dec = np.array([master_coordinates[i][1] for i in range(len(master_coordinates))])

    #Get distances between the check image coordinates and all the master image coordinates
    distances = np.sqrt((master_ra-check_ra)**2+(master_dec-check_dec)**2)
    min_distance_loc = np.where(distances == min(distances))[0][0]
    if distances[min_distance_loc] > 1:
        print('Warning: Current pointing off by > 1 degree of closest master image. Are you pointed in the right place?')

    master_path = Path(lines[min_distance_loc].split(', ')[0])
    target_name = lines[min_distance_loc].split(', ')[1].split('\n')[0]
    trimmed_target_name = target_name.replace(' ','')
    print('Closest target = ',target_name)

    #Get the appropriate master_dark for the master image. 
    exptime = fits.open(master_path)[0].header['EXPTIME']
    master_dark_path = calibration_path+'Darks/master_dark_'+str(exptime)+'.fits'
    if os.path.isfile(master_dark_path):
        master_dark = fits.open(master_dark_path)[0].data
    else:
        print('No master_dark file found matching the exposure time of '+exptime+' seconds in '+calibration_path+'.')
        print('You need to make one!')

    #Get sources in the check images.
    (check_x_centroids, check_y_centroids, check_fluxes , check_sharpness,check_image)  = mimir_source_finder(check_path, calibration_path, dark, flat, bpm,sigma_above_bg=3,fwhm=daostarfinder_fwhm)

    print(len(check_x_centroids),' sources found!')
    #If there were no sources found in the test image, don't do anything; likely clouds.
    if len(check_x_centroids) == 0:
        print('No sources found in latest test, take another!')
        print('')
        x_shift = 'nan'
        y_shift = 'nan'
        x_seeing = 'nan'
        y_seeing = 'nan'
        f = open(directory+'image_shift.txt','w')
        f.write(str(0)+' '+str(0)+' '+trimmed_target_name+' '+str(-1))
        f.close()
        #continue
            
    #Load the master synthetic image
    master_synthetic_image = fits.open('master_images/'+target_name.replace(' ','')+'_master_synthetic.fits')[0].data

    #Create a check synthetic image using the measured positions from mimir_source_finder()
    check_synthetic_image = synthetic_image_maker(check_x_centroids,check_y_centroids,check_fluxes,fwhm=8)
    
    #Measure the shift between the synthetic images.
    corr = signal.fftconvolve(master_synthetic_image,check_synthetic_image[::-1,::-1], mode='same')
    (x_shift,y_shift) = corr_shift_determination(corr)
    x_shift = x_shift - 0.5 #THESE CORRECTIONS WORK FOR A BINNING FACTOR OF 2. NOT SURE WHAT THEY ARE FOR ANY OTHER BINNING FACTOR.
    y_shift = y_shift + 0.5

    if diagnostic_plot:
        fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(8,8))
        ax[0,0].imshow(master_synthetic_image,origin='lower',cmap='Greys')
        ax[0,0].set_title('Master image and sources')
        plt.savefig('master.png',dpi=300)

        avg, med, std = sigma_clipped_stats(check_image,mask=bpm,sigma=3,maxiters=3)
        ax[0,1].imshow(check_image,origin='lower',vmin=med,vmax=med+6*std,cmap='Greys')
        ax[0,1].plot(check_x_centroids,check_y_centroids,'bo',alpha=0.7,ms=8,mfc='None')
        ax[0,1].set_title('Check image and sources')

        ax[1,1].imshow(corr,origin='lower')
        ax[1,1].set_title('Correlation')
        plt.savefig('shifting_diagnostic_plot.png',dpi=300)
        plt.clf()
        plt.close()

    #Get a suitable guide star from the check image to measure its seeing. 
    # guide_star_cut = np.where((check_x_centroids > 200) & (check_x_centroids < 824) & (check_y_centroids > 200) & (check_y_centroids < 824))[0]
    # if len(guide_star_cut) != 0:
    #     brightest_guide_star_ind = np.where(check_fluxes[guide_star_cut] == max(check_fluxes[guide_star_cut]))[0]
    #     guide_star_x_int = int(check_x_centroids[guide_star_cut][brightest_guide_star_ind])
    #     guide_star_y_int = int(check_y_centroids[guide_star_cut][brightest_guide_star_ind])
    #     guide_star_subframe = check_image[guide_star_y_int-15:guide_star_y_int+15,guide_star_x_int-15:guide_star_x_int+15]
    #     (x_seeing,y_seeing) = guide_star_seeing(guide_star_subframe)
    # else:
    #     x_seeing = 'nan'
    #     y_seeing = 'nan'

    #Median of FWHM of guide stars to get seeing 
    # guide_star_cut = np.where((check_x_centroids > 200) & (check_x_centroids < 824) & (check_y_centroids > 200) & (check_y_centroids < 824))[0]
    guide_star_cut = np.where((check_x_centroids > 50) & (check_x_centroids < 975) & (check_y_centroids > 50) & (check_y_centroids < 975))[0]
    if len(guide_star_cut) != 0:
        x_seeing_array = np.array([])
        y_seeing_array = np.array([])
        for guide_star_ind in guide_star_cut:
            guide_star_x_int = int(check_x_centroids[guide_star_ind])
            guide_star_y_int = int(check_y_centroids[guide_star_ind])
            guide_star_subframe = check_image[guide_star_y_int-15:guide_star_y_int+15,guide_star_x_int-15:guide_star_x_int+15]
            (x_seeing,y_seeing) = guide_star_seeing(guide_star_subframe)
            x_seeing_array = np.append(x_seeing_array,x_seeing)
            y_seeing_array = np.append(y_seeing_array,y_seeing)
        print(len(guide_star_cut),"sources used for seeing calc:",np.round(x_seeing_array,2))
        x_seeing = np.nanmedian(x_seeing_array)
        y_seeing = np.nanmedian(y_seeing_array)
    else:
        print("No sources for seeing calc, returning NaNs.")
        x_seeing = 'nan'
        y_seeing = 'nan'

        
    #Make sure large shifts aren't reported.
    if (abs(x_shift) > 200) or (abs(y_shift) > 200):
        print('Image shift larger than 200 pixels measured in at least one dimension. Returning zeros, inspect manually!')
        f = open(directory+'image_shift.txt','w')
        x_shift = 'nan'
        y_shift = 'nan'
        f.write(str(0)+' '+str(0)+' '+trimmed_target_name+' '+str(-1))
        f.close()
    else:
        x_shift = np.round(x_shift,1)
        y_shift = np.round(y_shift,1)
        ra_shift = np.round(x_shift*0.579,1)
        dec_shift = np.round(y_shift*0.579,1)
        if type(x_seeing) != str:
            x_seeing = np.round(x_seeing,1)
            y_seeing = np.round(y_seeing,1)
            max_seeing = np.round(np.max([x_seeing,y_seeing]),1)
        else:
            max_seeing = -1
        pix_shift_string = '('+str(x_shift)+','+str(y_shift)+')'
        pix_shift_string.replace(' ','')
        ang_shift_string = '('+str(ra_shift)+'",'+str(dec_shift)+'")'
        ang_shift_string.replace(' ','')
        seeing_string = '('+str(x_seeing)+'",'+str(y_seeing)+'")'
        print('Measured (X shift, Y shift):    ',pix_shift_string)
        print('Measured (RA shift, Dec shift): ',ang_shift_string)
        print('Measured (X seeing, Y seeing):  ',seeing_string)
        f = open(directory+'image_shift.txt','w')
        f.write(str(ra_shift)+' '+str(dec_shift)+' '+trimmed_target_name+' '+str(max_seeing))
        f.close()
        #break #No need to keep looping, you already found the match!

    #if match == False:
    #    print('No matching master image found! Do you have a master for this field?')
    #else: 
    return(x_shift,y_shift,x_seeing,y_seeing)
