"""Library of functions to prepare observations for coronal hole detection.
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
EARTH_COORD_KEYS = ['RSUN_REF', 'DSUN_OBS', 'HGLN_OBS', 'HGLT_OBS']


# Extraction Functions
def get_image_from_fits(fits_file):
    """Retrieve the first image array from a FITS file, flipped
    upside down for visualization.
    
    Args
        fits_file: path to FITS file
    Returns
        First numpy array in primary HDU.
    """
    with fits.open(fits_file) as hdu_list:
        if len(hdu_list[0].shape) > 2:
            image = np.flipud(hdu_list[0].data[0])
        else:
            image = np.flipud(hdu_list[0].data)
    
    return image


def download_euv(download_date_list, euv_date_list, sat,
                 output_dir, hr_window):
    """Download an EUV FITS file for each datetime in download list in
    a surrounding hour window.
    
    Args
        download_date_list: list of date strings for desired EUV dates
        euv_date_list: list of available EUV data date strings
        sat: satellite name between 'SDO' or 'SOHO'. Defaults to 'SDO'.
        output_dir: path to directory to download data to
        hr_window: int for number of hours around desired download dates
            to search in
    """
    dates_to_download = []
    fetch_results = []
    downloaded_dates = []
    failed_dates = []

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
            
            euv_available_for_he = ((min_date <= euv_datetime)
                                    and (euv_datetime <= max_date))
            if euv_available_for_he:
                continue
        
        dates_to_download.append(date_str)
        
        time_range = a.Time(min_date, max_date)
        cadence = a.Sample(30*u.minute)
        
        if sat == 'SOHO':
            result = Fido.search(
                time_range,
                a.Instrument.eit, a.Wavelength(195*u.angstrom),
                cadence
            )
        else: # sat == SDO
            result = Fido.search(
                time_range,
                a.Instrument.aia, a.Wavelength(193*u.angstrom),
                cadence
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
            fetch_result, path=output_dir + '{file}'
        )
        
        if downloaded_files.errors:
            print(f'Error downloading EUV for {date}. '
                  + 'Please reattempt.')
            failed_dates.append(date)
        elif not downloaded_files:
            print(f'{date} not found.')
            failed_dates.append(date)
        else:
            downloaded_dates.append(date)

    if downloaded_dates:
        print('Downloaded EUV Observation Datetimes:')
        display_dates(downloaded_dates)
        
        print('Failed EUV Observation Datetimes:')
        display_dates(failed_dates)
    else:
        print('No EUV files were downloaded.')


# FITS Extraction
def get_fits_path_list(date_range, all_dir, select_dir):
    """Retrieve FITS path list in the specified date range.
    
    Args
        date_range: tuple of min and max date strings
        all_dir: path to all data directory
        select_dir: path to selected data directory
    Returns
        List of FITS file paths in the specified date range.
    """
    if date_range:
        data_dir = all_dir
    else:
        data_dir = select_dir
    
    glob_pattern = data_dir + '*.fts'
    fits_path_list = glob.glob(glob_pattern)
    
    if not date_range:
        return fits_path_list
    
    min_date = datetime.strptime(date_range[0], DICT_DATE_STR_FORMAT)
    max_date = datetime.strptime(date_range[1], DICT_DATE_STR_FORMAT)

    date_str_list = [fits_path.split('/')[-1].split('.')[0]
                    for fits_path in fits_path_list]
    datetime_list = [datetime.strptime(date_str, DICT_DATE_STR_FORMAT)
                    for date_str in date_str_list]

    fits_path_list = [fits_path for fits_path, datetime
                      in zip(fits_path_list, datetime_list)
                      if (datetime > min_date) and (datetime < max_date)]
    fits_path_list.sort()
        
    return fits_path_list


def get_fits_date_list(date_range, all_dir, select_dir):
    """Retrieve list of available date strings of FITS files in the
    specified date range.
    
    Args
        date_range: tuple of min and max date strings
        all_dir: path to all data directory
        select_dir: path to selected data directory
    Returns
        List of FITS file paths in the specified date range.
    """
    fits_path_list = get_fits_path_list(date_range, all_dir, select_dir)
    fits_file_list = [fits_path.split('/')[-1]
                      for fits_path in fits_path_list]
    date_str_list = [fits_file.split('.')[0] for fits_file in fits_file_list]
    date_str_list.sort()
    
    return date_str_list


def get_fits_path(date_str, date_range, all_dir, select_dir):
    """Retrieve FITS path for the specified date.
    
    Args
        date_str: desired file date string
        date_range: tuple of min and max date strings
        all_dir: path to all data directory
        select_dir: path to selected data directory
    Returns
        FITS file path for the specified date.
    """
    if not date_str:
        raise Exception('Date string was not specified in file path retrieval.')
    
    if date_range:
        data_dir = all_dir
    else:
        data_dir = select_dir
    return f'{data_dir}{date_str}.fts'


def get_fits_content(fits_path):
    """Extract content from a FITS file
    """
    hdu_list = fits.open(fits_path)
    
    # Take header from final HDU
    header = hdu_list[-1].header
    
    date_key = 'DATE-OBS'
    if date_key not in header.keys():
        date_key = 'DATE'

    # Extract observation datetime
    obs_datetime = datetime.fromisoformat(header[date_key])
    date_str = datetime.strftime(obs_datetime, DICT_DATE_STR_FORMAT)

    num_data_arrays = hdu_list[0].header.get('NAXIS3')
    
    return hdu_list, date_str, num_data_arrays


# Renaming Data Functions
def rename_dir(data_dir, remove_gzip=False):
    """Rename all He FITS files to include observation date in title
    """
    # Copy gzip files to FITS files and delete gzip files
    gzip_path_list = glob.glob(data_dir + '*.fts.gz')
    gzip_path_list.extend(glob.glob(data_dir + '*.fits.gz'))
    
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
    
    # Extend with SOHO EIT FITS files
    fits_path_list.extend(glob.glob(data_dir + '*efz*'))
    
    for fits_path in fits_path_list:
        hdu_list, date_str = get_fits_content(fits_path)[:2]
        
        hdu_list.close()
            
        os.rename(fits_path, data_dir + date_str + '.fts')


# Sunpy Map Operations
def get_nso_sunpy_map(fits_file):
    """Retrieve a Sunpy map with a Helioprojective Cartesian
    coordinate system and the first data array in a NSO FITS file
    from the KPVT or VSM instruments.
    
    Args
        fits_file: path to FITS file
    Returns
        Sunpy map object.
    """
    # Retrieve FITS header and data ------------------------------------------
    with fits.open(fits_file) as hdu_list:
        header = hdu_list[-1].header
        num_data_arrays = header.get('NAXIS3')
        
        if not num_data_arrays:
            data = hdu_list[-1].data
        else:
            data = hdu_list[-1].data[0]
    
    # Clean header and data --------------------------------------------------
    
    # Remove error causing keywords
    # PC indicates presence of coordinate transformation
    # BLANK only applies to integer data
    # COMMENT and HISTORY may contain non-ascii content and is not needed
    for key in ['PC1_1', 'PC1_2', 'BLANK', 'COMMENT', 'HISTORY']:
        if key in header.keys():
            header.pop(key)
    
    # Convert pixels with common value to first background pixel to zero value
    background_val = data[0,0]
    data = np.where(data == background_val, 0, data)
    
    if not header.get('DATE-OBS'):
        header['DATE-OBS'] = header['DATE']
    
    # Clean World Coordinate System ------------------------------------------
    
    # Check for incorrect or missing WCS and modify
    wcs_name = header.get('WCSNAME')
    
    # Perform requisite checks on VSM maps with Heliocentric Cartesian
    # coordinate system for coordinate system change to Helioprojective
    
    # Must have arcsec units. Warning messages will appear but the map will
    # be produced successfully.
    if (wcs_name == 'Helioprojective-cartesian'
        and header['CUNIT1'] != 'arcsec'):
        return sunpy.map.Map(data, header)
    
    # Must have zero centered coordinates
    if (wcs_name == 'Heliocentric-cartesian (approximate)'
        and (header['CRVAL1'] != 0 or header['CRVAL2'] != 0)):
        print((f'Failed to convert {fits_file} into a Sunpy map.')
              + ('Coordinates were Heliocentric but were not ')
              + ('zero centered.'))
        return None
        
    # Specify Earth-based observer for solar radius, distance to Sun,
    # and Heliographic coordinates to avoid warning messages due to
    # missing keywords
    earth_hp_coords = frames.Helioprojective(
        0*u.arcsec, 0*u.arcsec,
        observer='earth', obstime=header['DATE-OBS'],
    )
    earth_header = sunpy.map.make_fitswcs_header(data, earth_hp_coords)
    for earth_coord_key in EARTH_COORD_KEYS:
        header[earth_coord_key] = earth_header[earth_coord_key]
        
    # Add Earth-based Carrington longitude
    header['CRLN_OBS'] = sunpy.coordinates.sun.L0(header['DATE-OBS']).value
    
    # Create header keywords for missing WCS
    # Assume helioprojective scale, shared scale in the Tx and Ty directions,
    # and arcsec units
    if not wcs_name:
        header['WCSNAME'] = 'Helioprojective-cartesian'
        header['CTYPE1'] = 'HPLN-TAN'
        header['CTYPE2'] = 'HPLT-TAN'
        header['CDELT1'] = header['SCALE']
        header['CDELT2'] = header['SCALE']
        header['CUNIT1'] = 'arcsec'
        header['CUNIT2'] = 'arcsec'
        return sunpy.map.Map(data, header)
    
    # Apply absolute value of coordinate change per pixel such that
    # Solar-X is positive
    header['CDELT1'] = abs(header['CDELT1'])
    
    # Change primary World Coordinate System from Heliocentric Cartesian
    # to Helioprojective Cartesian for Sunpy to create map
    if header['WCSNAME'] == 'Heliocentric-cartesian (approximate)':
        
        # Cartesian coordinate units
        coord_u1 = u.Unit(header['CUNIT1'])
        coord_u2 = u.Unit(header['CUNIT2'])
        
        # Convert center pixel coordinates from distance to angle
        hc_coords = frames.Heliocentric(
            header['CRVAL1']*coord_u1,
            header['CRVAL2']*coord_u2, z=0*u.m,
            observer='earth', obstime=header['DATE-OBS']
        )
        hp_coords = hc_coords.transform_to(earth_hp_coords)
        header['CRVAL1'] = hp_coords.Tx.value
        header['CRVAL2'] = hp_coords.Ty.value
        
        # Convert change per pixel from distance to angle
        hc_delta_coords = frames.Heliocentric(
            header['CDELT1']*coord_u1,
            header['CDELT2']*coord_u2, z=0*u.m,
            observer='earth', obstime=header['DATE-OBS']
        )
        hp_delta_coords = hc_delta_coords.transform_to(earth_hp_coords)
        header['CDELT1'] = hp_delta_coords.Tx.value
        header['CDELT2'] = hp_delta_coords.Ty.value
        
        # Modify keywords
        header['WCSNAME'] = 'Helioprojective-cartesian'
        header['CTYPE1'] = 'HPLN-TAN'
        header['CTYPE2'] = 'HPLT-TAN'
        header['CUNIT1'] = 'arcsec'
        header['CUNIT2'] = 'arcsec'
        
    return sunpy.map.Map(data, header)


def diff_rotate(input_map, target_map):
    """Reproject an input map with differential rotation to the datetime of a
    target map.
    
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


# Adjacent Observation Date Retrieval
def get_nearest_date_str(date_str_list, selected_date_str):
    """Retrieve date string in list that is nearest a selected date string
    within an hour window.
    
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


# Display Data Functions
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


# Extract arrays from FITS files. To be deleted
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
