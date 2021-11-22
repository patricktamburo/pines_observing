import pysftp
import getpass
from datetime import date, datetime, timedelta
from natsort import natsorted 
from glob import glob 
import numpy as np 
from astropy.io import fits 
from pathlib import Path

def pines_logging(filename, date, target_name, filter_name, exptime, airmass, x_shift, y_shift, x_seeing, y_seeing, post_processing_flag,   shift_quality_flag):
    """Exports a line of log text in the same style as the telescope observing logs. 

    :param filename: string of raw filename (e.g., '20210214.203.fits')
    :type filename: str
    :param date: date of image (YYYYMMDD)
    :type date: str
    :param target_name: taget's long name
    :type target_name: str
    :param filter_name: filter name (e.g., 'J' or 'H')
    :type filter_name: str
    :param exptime: exposure time
    :type exptime: str
    :param airmass: airmass
    :type airmass: str
    :param x_shift: x shift in pixels
    :type x_shift: str
    :param y_shift: y shift in pixels
    :type y_shift: str
    :param x_seeing: x seeing in arcsec
    :type x_seeing: str
    :param y_seeing: y seeing in arcsec
    :type y_seeing: str
    :param post_processing_flag: 0 (not yet post-processed) or 1 (post-processed)
    :type post_processing_flag: int
    :param shift_quality_flag: 0 (good shifts) or 1 (bad shifts)
    :type shift_quality_flag: int
    """
    log_text = ' {:<19}, {:<20}, {:<30}, {:<6}, {:<8}, {:<8}, {:<8}, {:<8}, {:<9}, {:<9}, {:<20}, {:<20}\n'.format(filename, date, target_name,
                                                                                                                       filter_name, str(
                                                                                                                           exptime),
                                                                                                                       str(airmass), str(
                                                                                                                           x_shift),
                                                                                                                       str(
                                                                                                                           y_shift),
                                                                                                                       str(
                                                                                                                           x_seeing),
                                                                                                                       str(
                                                                                                                           y_seeing),
                                                                                                                       str(
                                                                                                                           post_processing_flag),
                                                                                                                       str(shift_quality_flag))


    return log_text

#By default, point to today's date directory.
ut_date = datetime.utcnow()
if ut_date.month < 10:
    month_string = '0'+str(ut_date.month)
else:
    month_string = str(ut_date.month)

if ut_date.day < 10:
    day_string = '0'+str(ut_date.day)
else:
    day_string = str(ut_date.day)

date_string = str(ut_date.year)+month_string+day_string
day = int(date_string[6:])

i = 3
while i != 0:
    print('')
    username = input('Enter pines.bu.edu username: ')
    password = getpass.getpass('Enter password for {}@pines.bu.edu: '.format(username))
    print('')
    try:
        sftp = pysftp.Connection('pines.bu.edu', username=username, password=password)
        print('Login successful!\n')
        break
    except:
        i -= 1
        if i == 1:
            verb = 'try'  # lol
        else:
            verb = 'tries'
        if i != 0:
            print('Login failed! {} {} remaining.'.format(i, verb))
        else:
            print('ERROR: login failed after {} tries.'.format(i))

minus_run = datetime.strptime(date_string, '%Y%m%d') - timedelta(days=(day+1))
plus_run = datetime.strptime(date_string, '%Y%m%d') + timedelta(days=29) #Can always add 29 days and get to the next month. 
possible_run_ids = [datetime.strftime(minus_run, '%Y%m'), date_string[0:6], datetime.strftime(plus_run, '%Y%m')]

while True:
    run_id = input("Enter the run identifier (YYYYMM). Given today's UT date, it should be {}, {}, or {}: ".format(possible_run_ids[0], possible_run_ids[1], possible_run_ids[2]))
    if len(run_id) != 6:
        print('Wrong run_id format, try again.')
    elif (int(run_id[4:]) > 12) or int(run_id[4:]) < 1:
        print('Month must be between 01 and 12, try again.')
    elif (int(run_id[0:4]) < 2015):
        print('Year must be 2015 or later, try again.')
    elif run_id not in possible_run_ids:
        print('Error, expected a run id in {}, {}, or {}, try again.'.format(possible_run_ids[0], possible_run_ids[1], possible_run_ids[2]))
    else:
        break

#Move the sftp into the upload directory
sftp.chdir('/data/raw/mimir/{}/'.format(run_id))
try:
    sftp.chdir(date_string)
except:
    sftp.mkdir(date_string)
    sftp.chdir(date_string)

local_path = '/mimir/data/obs72/{}/'.format(date_string) 
#local_path = '/Users/tamburo/Downloads/20211112/' #For testing
remote_path = '/volume1/data/raw/mimir/{}/{}/'.format(run_id, date_string)

local_files = np.array(natsorted(glob(local_path+'*')))
for i in range(len(local_files)):
    local_file_path = local_files[i]
    local_file_name = local_file_path.split('/')[-1]
    remote_file_path = remote_path+local_file_name
    print('{}, {} of {}.'.format(local_file_name, i+1, len(local_files)))

    #Upload if the file does not exist already. 
    if not sftp.exists(local_file_name):
        sftp.put(local_file_path, local_file_name)
    else:
        #If it's a fits image, make sure the current version on the server is the correct file size. 
        if ('fits'in local_file_name) and (sftp.stat(local_file_name).st_size != 4213440):
            sftp.put(local_file_path, local_file_name)

    #Make a copy of the log file in /data/logs/, so master_log_creator.py can see it. 
    if '_log.txt' in local_file_name:

        #Check that there is a log line for every fits image in the local directory. 
        with open(local_file_path, 'r') as f:
            lines = f.readlines()
        log_filenames = [x.split(',')[0].replace(' ','') for x in lines[1:]]
        n_log_lines = len(log_filenames)

        local_fits_files = [x.split('/')[-1] for x in local_files if 'fits' in x]
        n_local_fits_files = len(local_fits_files)
        #If there are discrepancies, fix them. 
        if n_log_lines < n_local_fits_files:
            missing_files = [x for x in local_fits_files if x not in log_filenames]
            
            for x in missing_files:
                log_insert_ind = int(x.split('.')[1])
                header = fits.open(Path(local_files[0]).parent/x)[0].header
                filename = x
                target = '2MASS J'+header['OBJECT'].split('J')[1]
                date = header['DATE']
                filter = header['FILTNME2']
                exptime = str(header['EXPTIME'])
                airmass = str(header['AIRMASS'])
                x_shift = str(0.0)
                y_shift = str(0.0)
                x_seeing = str(2.5)
                y_seeing = str(2.5)
                pp_flag = str(0)
                sq_flag = str(0)
                missing_line = pines_logging(filename, date, target, filter, exptime, airmass, x_shift, y_shift, x_seeing, y_seeing, pp_flag, sq_flag)
                lines.insert(log_insert_ind, missing_line)
                
            #Write out the corrected log. 
            with open(local_file_path, 'w') as f:
                f.writelines(lines)
            #Reupload the correct log
            sftp.put(local_file_path, local_file_name)

        log_copy_path = '/data/logs/'+local_file_name
        sftp.put(local_file_path, log_copy_path)
    
#Now update the master log with the new log information. 
stdout = sftp.execute('python /volume1/code/PINES_server_scripts/master_log_creator.py')
print(stdout[0])

breakpoint()
