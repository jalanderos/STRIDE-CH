"""
"""

import os
import glob
import gzip
import shutil
import sunpy.map
import numpy as np
from PIL import Image
from scipy import ndimage
from datetime import datetime, timedelta

import astropy.units as u
from astropy.io import fits

from sunpy.coordinates import frames
from sunpy.net import Fido, attrs as a
from sunpy.coordinates import propagate_with_solar_surface
from sunpy.coordinates.sun import carrington_rotation_number


# Module variables
DICT_DATE_STR_FORMAT = '%Y_%m_%d__%H_%M'
HE_OBS_DATE_STR_FORMAT = '%Y-%m-%dT%H:%M:%S'
NUM_DISPLAY_DATES = 4


# Extraction Functions
def extract_he(date_range, all_He_dir, select_He_dir):
    """Extract He I FITS files to a dictionary keyed by date strings"""
    fits_path_list = get_fits_path_list(
        date_range, all_He_dir, select_He_dir
    )
        
    he_fits_dict = {}
    
    for fits_path in fits_path_list:
        he_fits, date_str, num_data_arrays = get_fits_content(fits_path)
        
        if not he_fits:
            continue
                
        # Extract and flip arrays upside down to visualize as images
        # Handle FITS files with a single data array
        if not num_data_arrays:
            raw_eqw = np.flipud(he_fits[0].data)
            
            title = 'Untitled Array'
            
            he_fits_dict[date_str] = (raw_eqw, title)
        else:
            raw_eqw = np.flipud(he_fits[0].data[0])
            continuum = np.flipud(he_fits[0].data[1])
            
            # List to hold data array titles
            title_list = []
            
            title_list.append(he_fits[0].header['IMTYPE1'])
            title_list.append(he_fits[0].header['IMTYPE2'])
            
            if num_data_arrays < 3:
                he_fits_dict[date_str] = (raw_eqw, continuum, title_list)
            else:
                cloud = np.flipud(he_fits[0].data[2])
                title_list.append(he_fits[0].header['IMTYPE3'])
                
                he_fits_dict[date_str] = (raw_eqw, continuum, cloud, title_list)
                
        he_fits.close()
        
    return he_fits_dict


def extract_comparison_fits(date_range, all_fits_dir, select_fits_dir):
    """Extract comparison FITS files to a dictionary keyed by date strings.
    UNUSED.
    """
    fits_path_list = get_fits_path_list(
        date_range, all_fits_dir, select_fits_dir
    )
    fits_dict = {}
    
    for fits_path in fits_path_list:
        hdu_list, date_str, num_data_arrays = get_fits_content(fits_path)
        
        if not hdu_list:
            continue
           
        # Extract and flip arrays upside down to visualize as images
        # Handle FITS files with a single data array
        if not num_data_arrays:
            img = np.flipud(hdu_list[-1].data)
        else:
            img = np.flipud(hdu_list[-1].data[0])
        
        hdu_list.close()
                
        fits_dict[date_str] = img

    return fits_dict


def extract_gong(gong_dir):
    """Extract GONG observatory magnetograms from FIT files
    to a dictionary keyed by Carrington Rotation strings.
    """
    glob_pattern = gong_dir + '*.fits'
        
    fits_path_list = glob.glob(glob_pattern)
        
    gong_dict = {}
    
    for fits_path in fits_path_list:
        gong_fits = fits.open(fits_path)
        
        gong_fits_header_keys = list(gong_fits[0].header.keys())
        
        # Pass to next FITS file if header information is missing
        if 'CAR_ROT' not in gong_fits_header_keys:
            continue
           
        # Carrington Rotation
        cr_num = gong_fits[0].header["CAR_ROT"]
        
        # Extract and flip arrays upside down to visualize as images
        magnetogram = np.flipud(gong_fits[0].data)
        
        gong_fits.close()
                
        gong_dict[cr_num] = magnetogram

    return gong_dict


def extract_comparison_ims(data_dir):
    """Extract all EUV or WSA coronal holes plot images to a
    dictionary keyed by date strings for comparison with He I observations.
    """
    glob_pattern = data_dir + '*.png'
    
    img_path_list = glob.glob(glob_pattern)
    
    img_dict = {}
    
    for img_path in img_path_list:
        img_file = img_path.split('/')[-1]
        date_str = img_file.split('.')[0]
        
        comp_img = Image.open(img_path)
        
        img_dict[date_str] = comp_img
        
    return img_dict


def extract_nso_eqw(nso_single_dir):
    """Extract NSO pre-processed Equivalent Width arrays from 
    He I FITS files to a dictionary keyed by date strings.
    """
    glob_pattern = nso_single_dir + '*e31hr*.fts'
    
    fits_path_list = glob.glob(glob_pattern)

    nso_eqw_dict = {}
    
    for fits_path in fits_path_list:
        nso_fits = fits.open(fits_path)
        
        nso_fits_header_keys = list(nso_fits[0].header.keys())
        
        # Pass to next FITS file if header information is missing
        if 'DATE' not in nso_fits_header_keys:
            continue

        # Extract observation datetime
        nso_datetime = datetime.strptime(
            nso_fits[0].header['DATE'], '%Y%m%d'
        )
        date_str = datetime.strftime(nso_datetime, DICT_DATE_STR_FORMAT)
           
        # Extract and flip arrays upside down to visualize as images
        nso_eqw = np.flipud(nso_fits[0].data[0])
        
        nso_fits.close()
                
        nso_eqw_dict[date_str] = nso_eqw
        
    return nso_eqw_dict


def extract_nso_ch_maps(nso_merged_dir):
    """Extract NSO synoptic Carrington maps of estimated coronal holes
    from FIT files to a dictionary keyed by Carrington Rotation strings.
    """
    glob_pattern = nso_merged_dir + '*o31hr*.fts'
    
    fits_path_list = glob.glob(glob_pattern)
    
    ch_map_dict = {}
    
    for fits_path in fits_path_list:
        nso_ch_fits = fits.open(fits_path)
        
        nso_ch_fits_header_keys = list(nso_ch_fits[0].header.keys())
        
        # Carrington Rotation Numbers
        cr_keys = [header_key
                   for header_key in nso_ch_fits_header_keys
                   if header_key[:4] == 'CARR']
        
        cr_keys.sort()
        first_cr = nso_ch_fits[0].header[cr_keys[0]]
        final_cr = nso_ch_fits[0].header[cr_keys[-1]]
           
        cr_key = f'{first_cr}__{final_cr}'
        
        # Extract and flip arrays upside down to visualize as images
        ch_map = np.flipud(nso_ch_fits[0].data[0])
        
        nso_ch_fits.close()
        
        ch_map_dict[cr_key] = ch_map
        
    return ch_map_dict


def download_euv(download_date_list, euv_date_list, download_dir, hr_window):
    """Download an EUV FITS file for each datetime in download list in
    a surrounding hour window.
    """
    dates_to_download = []
    fetch_results = []
    downloaded_dates = []

    for date_str in download_date_list:
        center_date = datetime.strptime(date_str, DICT_DATE_STR_FORMAT)
        
        min_date = center_date - timedelta(hours=hr_window)
        max_date = center_date + timedelta(hours=hr_window)
        
        # Skip download if nearest available EUV datetime lies within
        # hour window
        euv_date_str = get_nearest_date_str(
            date_str_list=euv_date_list, selected_date_str=date_str
        )
        if euv_date_str:
            euv_datetime = datetime.strptime(
                euv_date_str, DICT_DATE_STR_FORMAT
            )
            
            euv_available_for_eqw = ((min_date <= euv_datetime)
                                    and (euv_datetime <= max_date))
            if euv_available_for_eqw:
                continue
        
        dates_to_download.append(date_str)
        
        result = Fido.search(
            a.Time(min_date, max_date),
            a.Instrument.aia, a.Wavelength(193*u.angstrom),
            a.Sample(30*u.minute), 
        )
        center_result = result[:, len(result)//2]
        fetch_results.append(center_result)
        
    num_dates = len(dates_to_download)
    if dates_to_download:
        print(f'{num_dates} Datetimes for which to Download EUV:')
        display_dates(dates_to_download)
        print()

    for i, date, fetch_result in zip(
        range(num_dates), dates_to_download, fetch_results):
        print(f'Fetching data for {date}... {i + 1}/{num_dates}')

        downloaded_files = Fido.fetch(
            fetch_result, path=download_dir + '{file}'
        )
        
        if downloaded_files.errors:
            print(f'Error downloading EUV for {date}. '
                  + 'Please reattempt.')
        else:
            downloaded_dates.append(date)

    if downloaded_dates:
        print('Downloaded EUV Observation Datetimes:')
        display_dates(downloaded_dates)
    else:
        print('No EUV files were downloaded.')


# Extraction Processing Functions
def get_fits_path_list(date_range, all_dir, select_dir):
    """Retrieve FITS path list in the specified date range
    """
    if date_range:
        data_dir = all_dir
    else:
        data_dir = select_dir
    
    glob_pattern = data_dir + '*.fts'
        
    fits_path_list = glob.glob(glob_pattern)
    
    if date_range:
        min_date = datetime.strptime(date_range[0], DICT_DATE_STR_FORMAT)
        max_date = datetime.strptime(date_range[1], DICT_DATE_STR_FORMAT)
        
        in_range_fits_path_list = []
        
        for fits_path in fits_path_list:
            fits_file = fits_path.split('/')[-1]
            date_str = fits_file.split('.')[0]
            
            fits_datetime = datetime.strptime(date_str, DICT_DATE_STR_FORMAT)
            
            if (fits_datetime < min_date) or (fits_datetime > max_date):
                continue
            
            fits_file = f'{date_str}.fts'
            fits_path = os.path.join(data_dir, fits_file)
            
            in_range_fits_path_list.append(fits_path)
            
        fits_path_list = in_range_fits_path_list
        
    return fits_path_list


def get_fits_content(fits_path):
    """Extract content from a FITS file
    """
    try:
        hdu_list = fits.open(fits_path)
    except Exception as e:
        print(f'Error occured opening {fits_path}.')
        print(e)
        return None, None, None        
    
    # Take header from final HDU
    header = hdu_list[-1].header
        
    # Pass to next FITS file if header information is missing
    if 'DATE-OBS' not in header.keys():
        print(f'Observation date keyword missing in header of {fits_path}.')
        return None, None, None

    # Extract observation datetime
    obs_datetime = datetime.fromisoformat(header['DATE-OBS'])
    date_str = datetime.strftime(obs_datetime, DICT_DATE_STR_FORMAT)

    num_data_arrays = hdu_list[0].header.get('NAXIS3')
    
    return hdu_list, date_str, num_data_arrays


def get_fits_date_list(date_range, all_dir, select_dir):
    """Retrieve list of available date strings of FITS files in the
    specified date range.
    """
    fits_path_list = get_fits_path_list(date_range, all_dir, select_dir)
    fits_file_list = [fits_path.split('/')[-1]
                      for fits_path in fits_path_list]
    date_str_list = [fits_file.split('.')[0] for fits_file in fits_file_list]
    date_str_list.sort()
    
    return date_str_list


# Renaming Data Functions
def rename_dir(data_dir, remove_gzip=False):
    """Rename all He FITS files to include observation date in title
    """
    # Copy gzip files to FITS files and delete gzip files
    glob_gzip_pattern = data_dir + '*.fts.gz'
    gzip_path_list = glob.glob(glob_gzip_pattern)
    
    for gzip_path in gzip_path_list:
        
        with gzip.open(gzip_path, 'rb') as f_in:
            fits_path = gzip_path[:-3]
            
            with open(fits_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            
        if remove_gzip:
            os.remove(gzip_path)
    
    # Rename FITS files
    fits_path_list = glob.glob(data_dir + '*.fts')
    fits_path_list.extend(glob.glob(data_dir + '*.fits'))
    
    for fits_path in fits_path_list:
        hdu_list, date_str = get_fits_content(fits_path)[:2]
        
        hdu_list.close()
            
        os.rename(fits_path, data_dir + date_str + '.fts')
        
        
def rename_all_gong(gong_dir):
    """Rename all GONG magnetogram FITS files to include observation date in title"""
    glob_pattern = gong_dir + '*.fits'
    
    fits_path_list = glob.glob(glob_pattern)
    
    for fits_path in fits_path_list:
        gong_fits = fits.open(fits_path)
        
        gong_fits_header_keys = list(gong_fits[0].header.keys())
                
        # Pass to next FITS file if header information is missing
        if 'CAR_ROT' not in gong_fits_header_keys:
            continue
        
        # Carrington Rotation
        CR_str = f'CR{gong_fits[0].header["CAR_ROT"]}'
        
        gong_fits.close()
            
        os.rename(fits_path, gong_dir + CR_str + '.fits')


# Coordinate Map Functions
def get_solis_sunpy_map(fits_file):
    """Retrieve a Sunpy map with a Helioprojective Cartesian
    coordinate system and the first data array in a SOLIS VSM FITS file.
    
    Args
        fits_file: path to FITS file
    Returns
        Sunpy map object.
    """
    with fits.open(fits_file) as hdu_list:
        header = hdu_list[-1].header
        num_data_arrays = header.get('NAXIS3')
        
        if not num_data_arrays:
            data = hdu_list[-1].data
        else:
            data = hdu_list[-1].data[0]

    # Take absolute value of coordinate change per pixel such that
    # Solar-X is positive
    header['CDELT1'] = abs(header['CDELT1'])

    # Units must be arcsec for further processing if
    # Helioprojective Cartesian is the primary World Coordinate System
    if (header['WCSNAME'] == 'Helioprojective-cartesian'
        and header['CUNIT1'] != 'arcsec'):
        return sunpy.map.Map(data, header)
        
    # Specify Earth-based observer for solar radius, distance to Sun,
    # and Heliographic coordinates to avoid warning messages due to
    # missing keywords.
    earth_hp_coords = frames.Helioprojective(
        header['CRVAL1']*u.arcsec, header['CRVAL2']*u.arcsec,
        observer='earth', obstime=header['DATE-OBS'],
    )
    earth_header = sunpy.map.make_fitswcs_header(data, earth_hp_coords)
    for earth_coord_key in ['RSUN_REF', 'DSUN_OBS', 'HGLN_OBS', 'HGLT_OBS']:
        header[earth_coord_key] = earth_header[earth_coord_key]

    # Change primary World Coordinate System from Heliocentric Cartesian
    # to Helioprojective Cartesian for Sunpy to create map
    if header['WCSNAME'] == 'Heliocentric-cartesian (approximate)':
        
        # Units must be solar radii for Heliocentric Cartesian as 
        # primary World Coordinate System
        if header['CUNIT1'] != 'solRad':
            print((f'Failed to convert {fits_file} into a Sunpy map.')
                + ('Coordinates were Heliocentric but did not use solar ')
                + ('radii units.'))
            return None
        
        header['WCSNAME'] = 'Helioprojective-cartesian'
        header['CTYPE1'] = 'HPLN-TAN'
        header['CTYPE2'] = 'HPLT-TAN'
        header['CUNIT1'] = 'arcsec'
        header['CUNIT2'] = 'arcsec'
        
        # Remove error causing keywords indicate presence of
        # coordinate transformation
        header.pop('PC1_1')
        header.pop('PC2_2')
        
        # Z-axis is neglected
        z = 0*u.m
        
        # Center pixel coordinates
        hc_coords = frames.Heliocentric(
            header['CRVAL1']*u.solRad, header['CRVAL2']*u.solRad, z,
            observer='earth', obstime=header['DATE-OBS']
        )
        hp_coords = hc_coords.transform_to(earth_hp_coords)
        header['CRVAL1'] = hp_coords.Tx.value
        header['CRVAL2'] = hp_coords.Ty.value
        
        # Angular change per pixel
        hc_delta_coords = frames.Heliocentric(
            header['CDELT1']*u.solRad, header['CDELT2']*u.solRad, z,
            observer='earth', obstime=header['DATE-OBS']
        )
        hp_delta_coords = hc_delta_coords.transform_to(earth_hp_coords)
        header['CDELT1'] = hp_delta_coords.Tx.value
        header['CDELT2'] = hp_delta_coords.Ty.value

    return sunpy.map.Map(data, header)


def get_reprojected_map(input_map, target_map):
    """Reproject an input map with differential rotation to the datetime of a
    target map. Meta data warnings appear which may possibly be attenuated by
    specifying header keywords from Earth sky coordinates, not sure yet.
    
    Args
        input_map: Sunpy map to reporoject
        target_map: Sunpy map with observation datetime to reproject to
    Returns
        Reprojected input map to the datetime of the target map.
    """
    with propagate_with_solar_surface():
        reprojected_map = input_map.reproject_to(target_map.wcs)
    return reprojected_map
 
 
def get_smoothed_map(sunpy_map, smooth_size_percent):
    """Retrieve uniformly smoothed Sunpy map.
    
    Args
        sunpy_map: Sunpy map instance
        smooth_size_percent: float to specify uniform smoothing kernel size
            as a percentage of image size
    Returns
        Sunpy map with smoothed data
    """
    data = sunpy_map.data
    smooth_size = smooth_size_percent/100 *data.shape[0]
    smoothed_data = ndimage.uniform_filter(
        data, smooth_size
    )
    # Remove background after to avoid empty output 
    # from removal then smoothing
    smoothed_data = np.where(
        data == 0, np.nan, smoothed_data
    )
    return sunpy.map.Map(smoothed_data, sunpy_map.meta)


# Display Data Functions
def get_nearest_date_str(date_str_list, selected_date_str):
    """Retrieve date string in list that is nearest a selected date string.
    
    Args
        date_str_list: list of date strings
        selected_date_str: desired date string
    Returns
        Nearest date string from list.
    """
    if not date_str_list:
        return None
    
    datetimes = [datetime.strptime(date_str, DICT_DATE_STR_FORMAT)
                 for date_str in date_str_list]
    selected_datetime = datetime.strptime(selected_date_str, DICT_DATE_STR_FORMAT)
    nearest_datetime = min(
        datetimes, key=lambda datetime: abs(datetime - selected_datetime)
    )
    return datetime.strftime(nearest_datetime, DICT_DATE_STR_FORMAT)


def get_latest_date_str(date_str_list, selected_date_str, hr_window=3):
    """Retrieve date string in list that is nearest a selected date string
    within an hour window or the nearest in the past. Avoids jumping ahead
    in time with images in movies.
    
    Args
        date_str_list: list of date strings
        selected_date_str: desired date string
    Returns
        Latest date string from list.
    """
    if not date_str_list:
        return None

    datetimes = np.array([datetime.strptime(date_str, DICT_DATE_STR_FORMAT)
                        for date_str in date_str_list])
    selected_datetime = datetime.strptime(selected_date_str, DICT_DATE_STR_FORMAT)
    nearest_datetime = min(
        datetimes, key=lambda datetime: abs(datetime - selected_datetime)
    )

    # Select nearest datetime in the past if outside hour window
    datetime_diff = nearest_datetime - selected_datetime
    if datetime_diff > timedelta(hours=hr_window):        
        datetimes = datetimes[[datetime < selected_datetime for datetime in datetimes]]
        nearest_datetime = max(
            datetimes, key=lambda datetime: datetime - selected_datetime
        )

    return datetime.strftime(nearest_datetime, DICT_DATE_STR_FORMAT)


def display_dates(date_list):
    """
    """
    count = 0
    for date_str in date_list:
        print(f'{date_str} \t', end='')
        
        count += 1
        if np.mod(count,NUM_DISPLAY_DATES) == 0:
            print()


def display_crs(cr_list):
    """
    """
    count = 0
    for cr_num in cr_list:
        print(f'{cr_num} \t', end='')
        
        count += 1
        if np.mod(count,NUM_DISPLAY_DATES) == 0:
            print()


def display_date_to_cr(date_str):
    """
    """
    # Extract observation datetime
    selected_datetime = datetime.strptime(date_str, DICT_DATE_STR_FORMAT)

    print(f'CR: {int(carrington_rotation_number(selected_datetime))}')


# UNUSED EQW Extraction
def extract_eqw_v0_1(date_range, all_He_dir, select_He_dir):
    """Extract pre-processed Equivalent Width arrays from He I FITS files
    to a dictionary keyed by date strings.
    """
    fits_path_list = get_fits_path_list(
        date_range, all_He_dir, select_He_dir
    )
    eqw_dict = {}
    
    for fits_path in fits_path_list:
        he_fits, date_str, num_data_arrays = get_fits_content(fits_path)
        
        if not he_fits:
            continue
           
        # Extract and flip arrays upside down to visualize as images
        # Handle FITS files with a single data array
        if not num_data_arrays:
            raw_eqw = np.flipud(he_fits[0].data)
        else:
            raw_eqw = np.flipud(he_fits[0].data[0])
        
        he_fits.close()
        
        eqw = pre_process_eqw_v0_1(raw_eqw)[0]
        
        eqw_dict[date_str] = eqw

    return eqw_dict


def extract_eqw_v0_3(date_range, all_He_dir, select_He_dir):
    """Extract pre-processed Equivalent Width arrays from He I FITS files
    to a dictionary keyed by date strings.
    """
    fits_path_list = get_fits_path_list(
        date_range, all_He_dir, select_He_dir
    )
    eqw_dict = {}
    
    for fits_path in fits_path_list:
        he_fits, date_str, num_data_arrays = get_fits_content(fits_path)
        
        if not he_fits:
            continue
           
        # Extract and flip arrays upside down to visualize as images
        # Handle FITS files with a single data array
        if not num_data_arrays:
            raw_eqw = np.flipud(he_fits[0].data)
        else:
            raw_eqw = np.flipud(he_fits[0].data[0])
        
        he_fits.close()
        
        eqw = pre_process_eqw_v0_3(raw_eqw)[0]
        
        eqw_dict[date_str] = eqw

    return eqw_dict


# Trial Download Routine
# from datetime import datetime, timedelta

# DICT_DATE_STR_FORMAT = '%Y_%m_%d__%H_%M'

# # List of dates to download files for
# download_dates = []

# # Loop between start_date and end_date to build list of dates
# date_range = ('2012_04_01__00_00', '2012_04_04__00_00')
# min_date = datetime.strptime(date_range[0], DICT_DATE_STR_FORMAT)
# max_date = datetime.strptime(date_range[1], DICT_DATE_STR_FORMAT)
# current_date = min_date

# while current_date <= max_date:
#     download_dates.append(current_date)
#     current_date += timedelta(days=1)

# download_dates

# # Loop over dates to request and record file names and responses in lists
# all_requested_imgs = []
# all_responses = []

# for download_date in download_dates:
#     requested_imgs, responses = request_from_nso(download_date, output_dir)

#     if requested_imgs is None: 
#         continue
    
#     all_requested_imgs.extend(requested_imgs)
#     all_responses.extend(responses)


# def request_from_nso(download_date, output_path):
#     """Request files from NSO SOLIS VSM website by downloading HTML 
#     content and searching for matching file links using a regular expression.
#     If no files are found, return None.

#     Args:
#        download_date: Date to download images for (datetime.date)
#        output_path: path to downloaded images
#     Returns:
#        List of image file names in the format 'YYYYMMDD_HHMMSS_n7euA_195.jpg'
#        List of HTTP responses (requests.Response) 

#        Returns None in place of both lists if the request times out, an HTTP
#        error occurs, a generic request error occurs, or no regular expression
#        matches are found.
#     """
#     requested_imgs = []
#     responses = []

#     remote_addr = ('https://stereo-ssc.nascom.nasa.gov/browse/' +
#         f'{download_date.strftime("%Y/%m/%d")}/{args.spacecraft}/' +
#         'euvi/195/512/')
