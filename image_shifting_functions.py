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
from photutils import DAOStarFinder, Background2D, MedianBackground
from astropy.stats import sigma_clipped_stats, sigma_clip, SigmaClip
import pickle 
from pathlib import Path
import os
import time
from numpy.lib.stride_tricks import as_strided
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans


def auto_correlation_seeing(image_path, calibration_path, dark, flat, bpm, cutout_w=15):
    
    im = fits.open(image_path)[0].data[0:1024,:]
    
    #Reduce the image.
    im = (im - dark) / flat
    
    #Interpolate over bad pixels
    im[np.where(bpm)] = np.nan
    kernel = Gaussian2DKernel(x_stddev=1)
    im = interpolate_replace_nans(im, kernel)
    
    plate_scale = 0.579 #arcsec/pix
    sigma_to_fwhm = 2.355
    
    #Set a row to NaNs, which will dominate the autocorrelation of Mimir images.
    im[513] = np.nan
    
    #Interpolate nans in the image, repeating until no nans remain.
    while sum(sum(np.isnan(im))) > 0:
        im = interpolate_replace_nans(im, kernel=Gaussian2DKernel(0.5))

    #Cut 80 pixels near top/bottom edges, which can dominate the fft if they have a "ski jump" feature.
    im = im[80:944, :]
    y_dim, x_dim = im.shape
    
    #Subtract off a simple estimate of the image background.
    im -= sigma_clipped_stats(im)[1]

    #Do auto correlation
    fft = signal.fftconvolve(im, im[::-1, ::-1], mode='same')
    
    #Do a cutout around the center of the fft.
    cutout = fft[int(y_dim/2)-cutout_w:int(y_dim/2)+cutout_w,int(x_dim/2)-cutout_w:int(x_dim/2)+cutout_w]
    
    #Set the midplane of the cutout to nans and interpolate.
    cutout[cutout_w] = np.nan
    
    while sum(sum(np.isnan(cutout))) > 0:
        cutout = interpolate_replace_nans(cutout, Gaussian2DKernel(0.25))
    
    #Subtract off "background"
    cutout -= np.nanmedian(cutout)
    
    #Fit a 2D Gaussian to the cutout
    #Assume a seeing of 2".7, the average value measured for PINES.
    g_init = models.Gaussian2D(amplitude=np.nanmax(cutout), x_mean=cutout_w, y_mean=cutout_w, x_stddev=2.7/plate_scale/sigma_to_fwhm*np.sqrt(2), y_stddev=2.7/plate_scale/sigma_to_fwhm*np.sqrt(2))
    g_init.x_mean.fixed = True
    g_init.y_mean.fixed = True
    #Set limits on the fitted gaussians between 1".6 and 7".0
    #Factor of sqrt(2) corrects for autocorrelation of 2 gaussians.
    g_init.x_stddev.min = 1.6/plate_scale/sigma_to_fwhm*np.sqrt(2)
    g_init.y_stddev.min = 1.6/plate_scale/sigma_to_fwhm*np.sqrt(2)
    g_init.x_stddev.max = 7/plate_scale/sigma_to_fwhm*np.sqrt(2)
    g_init.y_stddev.max = 7/plate_scale/sigma_to_fwhm*np.sqrt(2)
    
    fit_g = fitting.LevMarLSQFitter()
    y, x = np.mgrid[:int(2*cutout_w), :int(2*cutout_w)]
    g = fit_g(g_init, x, y, cutout)
    
    #Convert to fwhm in arcsec.
    seeing_fwhm_as = g.y_stddev.value/np.sqrt(2)*sigma_to_fwhm*plate_scale
    
    return seeing_fwhm_as

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

    #Estimate the seeing using a 2d cross correlation
    #seeing = auto_correlation_seeing(image)
    #fwhm = seeing / 0.579 #units of pixels
    
    #Lower the image resolution, to save time doing source detection. 
    binning_factor = 2
    lowres_image = strided_rescale(image,binning_factor)
    ###lowres_bpm = strided_rescale(bpm,binning_factor).astype('bool')

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
    print(('Using a FWHM of {x:.1f}" for finding sources...').format(x=fwhm * 0.579))
    avg, med, stddev = sigma_clipped_stats(lowres_image,sigma=3.0,maxiters=3) #Previously maxiters = 5!
    daofind = DAOStarFinder(fwhm=fwhm/binning_factor, threshold=sigma_above_bg*stddev,sky=med,ratio=0.9)
    new_sources = daofind(lowres_image)

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

    #Also cut for sharpnesses > 0.5, this seems to eliminate a lot of false detections.
    use_sharp = np.where(sharpness > 0.5)[0]
    x_centroids  = x_centroids [use_sharp]
    y_centroids  = y_centroids [use_sharp]
    sharpness = sharpness[use_sharp]
    fluxes = fluxes[use_sharp]
    peaks = peaks[use_sharp]
    
    
    #Cut sources in the lower left, if bars are present.
    #use_ll =  np.where((x_centroids > 512/binning_factor) | (y_centroids > 512/binning_factor))
    #x_centroids  = x_centroids [use_ll]
    #y_centroids  = y_centroids [use_ll]
    #sharpness = sharpness[use_ll]
    #fluxes = fluxes[use_ll]
    #peaks = peaks[use_ll]
    
    #Finally, cut targets whose y centroids are near y = 512 (in full-res images). These are usually bad.
    use_512 = np.where(np.logical_or((y_centroids < 510/binning_factor),(y_centroids > 514/binning_factor)))[0]
    x_centroids  = x_centroids [use_512]
    y_centroids  = y_centroids [use_512]
    sharpness = sharpness[use_512]
    fluxes = fluxes[use_512]
    peaks = peaks[use_512]

    #Cut sources with negative/highly saturated peaks
    use_peaks = np.where((peaks > 30) & (peaks < 7000))[0]
    x_centroids  = x_centroids [use_peaks]
    y_centroids  = y_centroids [use_peaks]
    sharpness = sharpness[use_peaks]
    fluxes = fluxes[use_peaks]
    peaks = peaks[use_peaks]


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
    #weights = gain / np.sqrt(np.absolute(subframe)*gain + (read_noise*gain)**2) #1/sigma for each pixel
    weights = 1.0 / np.sqrt(np.absolute(subframe)/gain + read_noise**2) #1/sigma for each pixel
    gaussian, mask = fit_gauss(gaussian_init, x, y, subframe, weights)
    fwhm_x = 2.355*gaussian.x_stddev_1.value
    fwhm_y = 2.355*gaussian.y_stddev_1.value

    x_seeing = fwhm_x * 0.579
    y_seeing = fwhm_y * 0.579
    return(x_seeing,y_seeing)

def image_shift_calculator(lines, master_coordinates, dark, flat, bpm, daostarfinder_fwhm,directory = '',filename = '', 
                           calibration_path='/Users/obs72/Desktop/PINES_scripts/Calibrations/',input_file = 'input_file.txt', mode='created'):

    diagnostic_plot = 0 #Set to true to plot master and check images with sources.

    #Get the check image and its coordinates
    check_path = Path(directory+filename)
    check_hdul = fits.open(check_path)
    check_header = check_hdul[0].header
    check_exptime = check_header['EXPTIME']
    #ra  = [float(i) for i in check_header['TELRA'].split(':')]
    #dec = [float(i) for i in check_header['TELDEC'].split(':')]
    #The following change sets the RA and Dec to a default value if it isn't in the header.
    #This is useful if you take darks with LOIS not connected to MOVE (tele=none on configure)
    ra  = [float(i) for i in check_header.get('TELRA',default='00:00:00').split(':')]
    dec = [float(i) for i in check_header.get('TELDEC',default='00:00:00').split(':')]
    check_ra = 15*ra[0]+15*ra[1]/60+15*ra[2]/3600 
    check_dec = dec[0]+dec[1]/60+dec[2]/3600
    master_ra = np.array([master_coordinates[i][0] for i in range(len(master_coordinates))])
    master_dec = np.array([master_coordinates[i][1] for i in range(len(master_coordinates))])

    #Get distances between the check image coordinates and all the master image coordinates
    distances = np.sqrt((master_ra-check_ra)**2+(master_dec-check_dec)**2)
    min_distance_loc = np.where(distances == min(distances))[0][0]
    if distances[min_distance_loc] > 1:
        print('Current pointing off by > 1 degree of closest master image. Not measuring shifts/seeing.')
        x_shift = 'nan'
        y_shift = 'nan'
        x_seeing = 'nan'
        y_seeing = 'nan'
        return x_shift,y_shift,x_seeing,y_seeing
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

    #IF this is a science image, use reference stars to measure FWHM seeing.
    if mode == 'created':
        #Generate a seeing estimate using autocorrelation.
        seeing = auto_correlation_seeing(check_path, calibration_path, dark, flat, bpm)
        x_seeing = seeing
        y_seeing = seeing
        #print('Autocorr seeing: {:1.1f}'.format(seeing))
        
        #If that gives unrealistic values, check with the original approach: fit 2D gaussians to sources.
        if (seeing < 1.6) or (seeing > 7) or (np.isnan(seeing)):
            print('auto_correlation_seeing() failed, reverting to guide_star_seeing()')
            guide_star_cut = np.where((check_x_centroids > 50) & (check_x_centroids < 975) & (check_y_centroids > 50) & (check_y_centroids < 975))[0]
            if len(guide_star_cut) != 0:
                x_seeing_array = np.array([])
                y_seeing_array = np.array([])
                #Loop over the potential guide star positions, make cutouts around those positions, and measure the seeing in the subframes.
                for guide_star_ind in guide_star_cut:
                    guide_star_x_int = int(check_x_centroids[guide_star_ind])
                    guide_star_y_int = int(check_y_centroids[guide_star_ind])
                    guide_star_subframe = check_image[guide_star_y_int-15:guide_star_y_int+15,guide_star_x_int-15:guide_star_x_int+15]
                    (x_seeing,y_seeing) = guide_star_seeing(guide_star_subframe)
            
                    #Only use sources with believable seeing measurements.
                    if x_seeing > 0.8 and x_seeing < 6.5:
                        x_seeing_array = np.append(x_seeing_array,x_seeing)
                        y_seeing_array = np.append(y_seeing_array,y_seeing)
        
                print(len(x_seeing_array),"sources used for seeing calc:",np.round(x_seeing_array,2))
                x_seeing = np.nanmedian(x_seeing_array)
                y_seeing = np.nanmedian(y_seeing_array)
            else:
                print("No sources for seeing calc, returning NaNs.")
                x_seeing = 'nan'
                y_seeing = 'nan'

    #If it's a test image, skip the seeing measurement to save time.
    elif mode == 'modified':
        print('Test image, skipping seeing measurment.')
        x_seeing = 'nan'
        y_seeing = 'nan'
        
    #Make sure large shifts aren't reported.
    if (abs(x_shift) > 500) or (abs(y_shift) > 500):
        print('Image shift larger than 500 pixels measured in at least one dimension. Returning zeros, inspect manually!')
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
        #Write measured shifts in arcseconds to file.
        f.write(str(ra_shift)+' '+str(dec_shift)+' '+trimmed_target_name+' '+str(max_seeing))
        f.close()

    return(x_shift,y_shift,x_seeing,y_seeing)
