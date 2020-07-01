import pdb
from astropy.io import fits
from pathlib import Path
import os
import numpy as np

def PINES_logger(x_shift, y_shift, x_seeing, y_seeing, master_coordinates, lines, directory='',filename=''):
    #Read in header of fits file, extract info, and append it to a nights PINES_log.txt file.
    image_path = Path(directory+filename)
    image_hdul = fits.open(image_path)
    image_header = image_hdul[0].header

    

    filename = image_header['FILENAME']
    exptime = image_header['EXPTIME']
    date = image_header['DATE']
    filter_name = image_header['FILTNME2']
    obs_type = image_header['OBSTYPE'].strip()
    airmass = image_header['AIRMASS']
    if obs_type == 'OBJECT':
        if image_header['FILTNME2'].strip() == 'Dark':
            target_name = 'Dark'
        elif exptime < 1:
            target_name = 'Flat'
        else:
            ra  = [float(i) for i in image_header['TELRA'].split(':')]
            dec = [float(i) for i in image_header['TELDEC'].split(':')]
            check_ra = 15*ra[0]+15*ra[1]/60+15*ra[2]/3600 
            check_dec = dec[0]+dec[1]/60+dec[2]/3600

            master_ra = np.array([master_coordinates[i][0] for i in range(len(master_coordinates))])
            master_dec = np.array([master_coordinates[i][1] for i in range(len(master_coordinates))])

            #Get distances between the check image coordinates and all the master image coordinates
            distances = np.sqrt((master_ra-check_ra)**2+(master_dec-check_dec)**2)
            min_distance_loc = np.where(distances == min(distances))[0][0]
            master_path = Path(lines[min_distance_loc].split(', ')[0])
            target_name = lines[min_distance_loc].split(', ')[1].split('\n')[0]
    else:
        targ_name = ' '
    if x_shift != 'nan':
        #log_text = filename+', '+date+', '+target_name+', '+filter_name+', '+str(exptime)+', '+str(airmass)+'\n'
        
        if x_seeing != 'nan':
            log_text = ' {:<19}, {:<20}, {:<30}, {:<6}, {:<8}, {:<8}, {:<8}, {:<8}, {:<9}, {:<9}\n'.format(filename, date, target_name, 
                                                                                                       filter_name,str(exptime),
                                                                                                       str(airmass),str(np.round(x_shift,1)),
                                                                                                       str(np.round(y_shift,1)),
                                                                                                       str(np.round(x_seeing,1)),
                                                                                                       str(np.round(y_seeing,1)))
        else:
            log_text = ' {:<19}, {:<20}, {:<30}, {:<6}, {:<8}, {:<8}, {:<8}, {:<8}, {:<9}, {:<9}\n'.format(filename, date, target_name, 
                                                                                                       filter_name,str(exptime),
                                                                                                       str(airmass),str(np.round(x_shift,1)),
                                                                                                       str(np.round(y_shift,1)),
                                                                                                       x_seeing,y_seeing)
    else:
        if x_seeing != 'nan':
            log_text = ' {:<19}, {:<20}, {:<30}, {:<6}, {:<8}, {:<8}, {:<8}, {:<8}, {:<9}, {:<9}\n'.format(filename, date, target_name, filter_name,
                                                                                    str(exptime),str(airmass),x_shift,y_shift,
                                                                                                      str(np.round(x_seeing,1)), str(np.round(y_seeing,1)))

        else:
            log_text = ' {:<19}, {:<20}, {:<30}, {:<6}, {:<8}, {:<8}, {:<8}, {:<8}, {:<9}, {:<9}\n'.format(filename, date, target_name, 
                                                                                                       filter_name,str(exptime),
                                                                                                       str(airmass),x_shift,y_shift,
                                                                                                       x_seeing,y_seeing)
                                                                                                           
    log_filename = directory.split('/')[-2]+'_log.txt'
    with open(directory+log_filename, 'a') as myfile:
        if os.stat(directory+log_filename).st_size == 0:
            header_text = '#{:<19}, {:<20}, {:<30}, {:<6}, {:<8}, {:<8}, {:<8}, {:<8}, {:<8}, {:<8}\n'.format('Filename', 'Date', 'Target', 'Filt.','Exptime','Airmass','X shift', 'Y shift', 'X seeing','Y seeing')
            myfile.write(header_text)
        myfile.write(log_text)  
    myfile.close()
