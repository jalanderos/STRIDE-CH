"""
Library of functions to prepare observations for coronal hole detection.
"""

import os
import re
import glob
import gzip
import shutil
import requests
import numpy as np
from pathlib import Path
from scipy import ndimage
from skimage import transform
from bs4 import BeautifulSoup
from dataclasses import dataclass
from datetime import datetime, timedelta, date

import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord

import sunpy.map
from sunpy.coordinates import frames
from sunpy.net import Fido, attrs as a
from sunpy.coordinates import propagate_with_solar_surface
from sunpy.coordinates.sun import carrington_rotation_number
from sunpy.map.maputils import (
    all_coordinates_from_map, coordinate_is_on_solar_disk
)

from acwe_lib import acweSaveSeg_v5, acweRestoreScale

from settings import *

# Module variables
HE_OBS_DATE_STR_FORMAT = '%Y-%m-%dT%H:%M:%S'
ACWE_DICT_DATE_STR_FORMAT = '%Y-%m-%dT%H%M%SZ'
ACWE_SPLIT_FILE_NAME_DATE_IDX = 3
NUM_DISPLAY_DATES = 4
EARTH_COORD_KEYS = ['RSUN_REF', 'DSUN_OBS', 'HGLN_OBS', 'HGLT_OBS']

# Data source config via Claude
@dataclass
class SolisDataSource:
    """Configuration for a SOLIS VSM data archive."""
    name:             str    # human-readable label
    base_url:         str    # archive root URL
    dir_prefix:       str    # prefix of daily subdirectory e.g. 'k4v22', 'k4v72'
    file_pattern:     str    # regex matching desired filenames (anchored to basename)
    file_preference:  list[str]  # ordered list of substring preferences when multiple files exist per day
    dest_dir:        str        # local download directory

HE_SOURCE = SolisDataSource(
    name         = 'HeI 1083.0 intensity',
    base_url     = 'https://solis.nso.edu/pubkeep/VSM%20HeI%201083.0,%20level%202%20intensity%20data%20_v22/',
    dir_prefix   = 'k4v22',
    file_pattern = r'k4v22\d{6}t\d{6}_ew\.fts\.gz',
    file_preference = ['_ew.fts.gz'],
    dest_dir        = HE_DIR,
)
MAG_SOURCE = SolisDataSource(
    name         = 'FeI 630.15 longitudinal magnetogram',
    base_url     = 'https://solis.nso.edu/pubkeep/VSM%20FeI%20630.15%20longitudinal,%20level%202%20magnetogram%20data%20_v72/',
    dir_prefix   = 'k4v72',
    file_pattern = r'k4v72\d{6}t\d{6}_mag\d\.fts\.gz',
    file_preference = ['_mag1.fts.gz'],
    dest_dir        = MAG_DIR,
)


# Extraction Functions -------------------------------------------------------
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
        print()
        
        print('Failed EUV Observation Datetimes:')
        display_dates(failed_dates)
    else:
        print('No EUV files were downloaded.')


# Download He I and Magnetogram Functions via Claude -------------------------
# ── Main crawl function ───────────────────────────────────────────────────────
def get_available_dates(
    source:     SolisDataSource,
) -> list[date]:
    """
    Crawl a SOLIS archive and return a sorted list of dates within
    [start_date, end_date] for which at least one matching file exists.
    """
    start_date = datetime.strptime(DATE_RANGE[0], DICT_DATE_STR_FORMAT).date()
    end_date = datetime.strptime(DATE_RANGE[1], DICT_DATE_STR_FORMAT).date()

    available: set[date] = set()

    print(f"Source : {source.name}")
    print(f"Range  : {start_date} → {end_date}\n")

    # ── Level 1: YYYYMM top-level directories ────────────────────────────────
    top_links  = list_hrefs(source.base_url)
    month_dirs = [l for l in top_links if re.fullmatch(r"\d{6}/", l)]

    for month_dir in sorted(month_dirs):
        ym          = month_dir.strip("/")
        year_m      = int(ym[:4])
        month_m     = int(ym[4:])
        month_start = date(year_m, month_m, 1)
        next_month  = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
        month_end   = next_month - timedelta(days=1)

        if month_start > end_date or month_end < start_date:
            continue

        month_url = source.base_url + month_dir
        print(f'Scanning {ym}... ')

        # ── Level 2: daily subdirectories ────────────────────────────────────
        try:
            day_links = list_hrefs(month_url)
        except requests.HTTPError:
            print(f'Error for month_dir: {month_dir}')
            continue

        day_dirs = [
            l for l in day_links
            if re.search(rf"{re.escape(source.dir_prefix)}\d{{6}}/$", l)
        ]

        for day_dir in sorted(day_dirs):
            m = re.search(r"(\d{2})(\d{2})(\d{2})/$", day_dir)
            if not m:
                continue
            yy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
            day_date = date(2000 + yy, mm, dd)

            if not (start_date <= day_date <= end_date):
                continue

            day_url = month_url + day_dir

            # ── Level 3: find and select one matching file per day ────────────
            try:
                file_links = list_hrefs(day_url)
            except requests.HTTPError:
                print(f'Error for day_dir: {day_dir}')
                continue
            except requests.exceptions.ReadTimeout:
                print(f'Repeated timeouts for day_dir: {day_dir}')
                continue

            matching = [
                l for l in file_links
                if re.match(source.file_pattern, l)
            ]

            chosen = select_one_file(matching, source.file_preference)
            if chosen:
                file_date = parse_filename_date(chosen, source.dir_prefix)
                if file_date and start_date <= file_date <= end_date:
                    available.add(file_date)

    return sorted(available)


# ── Main download function ────────────────────────────────────────────────────
def download_dates(
    source:     SolisDataSource,
    dates:      list[date],
    decompress: bool = True,
) -> None:
    """
    Download one file per date from dates list using the source configuration.
    Only crawls months and days present in the input date list.
    """
    if not dates:
        print("No dates provided.")
        return

    Path(source.dest_dir).mkdir(parents=True, exist_ok=True)

    # Group dates by YYYYMM for efficient crawling
    months: dict[str, set[date]] = {}
    for d in dates:
        key = f"{d.year}{d.month:02d}"
        months.setdefault(key, set()).add(d)

    print(f"Source : {source.name}")
    print(f"Dates  : {len(dates)} requested across {len(months)} months\n")

    for ym in sorted(months):
        target_dates = months[ym]
        month_url    = source.base_url + ym + "/"
        print(f"  Month {ym} ({len(target_dates)} dates) …")

        # ── Level 2: daily subdirectories ────────────────────────────────────
        try:
            day_links = list_hrefs(month_url)
        except (requests.HTTPError, requests.exceptions.ReadTimeout):
            print(f"    Could not access {month_url}, skipping")
            continue

        day_dirs = [
            l for l in day_links
            if re.search(rf"{re.escape(source.dir_prefix)}\d{{6}}/$", l)
        ]
        
        for day_dir in sorted(day_dirs):
            match = re.search(r"(\d{2})(\d{2})(\d{2})/$", day_dir)
            if not match:
                print(None)
                continue
            yy, mm, dd = int(match.group(1)), int(match.group(2)), int(match.group(3))
            day_date = date(2000 + yy, mm, dd)
            if day_date not in target_dates:
                continue
            
            day_url = month_url + day_dir
            print(f"    {day_date} …")

            # ── Level 3: find, select, and download one file ──────────────────
            try:
                file_links = list_hrefs(day_url)
            except requests.exceptions.ReadTimeout:
                print(f"      Timeout, skipping {day_dir.strip('/')}")
                continue
            except requests.HTTPError:
                continue

            matching = [
                l for l in file_links
                if re.match(source.file_pattern, l)
            ]

            chosen = select_one_file(matching, source.file_preference)
            if chosen:
                download_file(day_url + chosen, source.dest_dir, decompress)
            else:
                print(f"      No matching file found for {day_date}")


# ── Core utilities ────────────────────────────────────────────────────────────
def list_hrefs(url: str) -> list[str]:
    """Return all href values from an Apache-style index page."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    return [a["href"] for a in soup.find_all("a", href=True)]


def parse_filename_date(filename: str, dir_prefix: str) -> date | None:
    """
    Parse date from filename using dir_prefix to anchor the pattern.
    e.g. k4v22100508t221732_ew.fts.gz → date(2010, 5, 8)
         k4v72100508t184932_mag1.fts.gz → date(2010, 5, 8)
    """
    escaped = re.escape(dir_prefix)
    m = re.search(escaped + r"(\d{2})(\d{2})(\d{2})t\d{6}", filename)
    if not m:
        return None
    yy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return date(2000 + yy, mm, dd)
    except ValueError:
        return None


def select_one_file(
    files:      list[str],
    preference: list[str],
) -> str | None:
    """
    Select one file from a list according to ordered substring preferences.
    Falls back to the first file if no preference matches.
    """
    for pref in preference:
        matches = [f for f in files if pref in f]
        if matches:
            return matches[0]
    return files[0] if files else None


# ── Download utilities ────────────────────────────────────────────────────────
def download_file(
    url:        str,
    dest_dir:   str,
    decompress: bool = True,
) -> None:
    """Download a .fts.gz file and optionally decompress it."""
    filename  = url.split("/")[-1]
    dest_path = Path(dest_dir) / filename

    if decompress:
        final_path = dest_path.with_suffix("")  # strip .gz
        if final_path.exists():
            print(f"      already exists, skipping: {final_path.name}")
            return
    else:
        if dest_path.exists():
            print(f"      already exists, skipping: {dest_path.name}")
            return

    print(f"      downloading {filename} …")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)

    if decompress and dest_path.suffix == ".gz":
        print(f"      decompressing → {dest_path.stem}")
        with gzip.open(dest_path, "rb") as gz_in:
            with open(dest_path.with_suffix(""), "wb") as fts_out:
                shutil.copyfileobj(gz_in, fts_out)
        dest_path.unlink()


# FITS File Loading and Renaming ---------------------------------------------
def get_fits_date_list(he_date_range, data_dir):
    """Retrieve list of available date strings of FITS files in the
    specified date range.
    
    Args
        he_date_range: tuple of start and end date strings for He I data or
            list of He I data dates
        data_dir: path to data directory
    Returns
        List of FITS file paths in the specified date range.
    """
    # Retrieve FITS path list in the specified date range --------------------
    glob_pattern = data_dir + '*.fts'
    fits_path_list = glob.glob(glob_pattern)
    
    date_str_list = [fits_path.split('/')[-1].split('.')[0]
                    for fits_path in fits_path_list]
    datetime_list = [datetime.strptime(date_str, DICT_DATE_STR_FORMAT)
                    for date_str in date_str_list]
    
    if isinstance(he_date_range, tuple):
        # Keep only dates in date range
        start_date = datetime.strptime(he_date_range[0], DICT_DATE_STR_FORMAT)
        end_date = datetime.strptime(he_date_range[1], DICT_DATE_STR_FORMAT)
        
        fits_path_list = [
            fits_path for fits_path, datetime
            in zip(fits_path_list, datetime_list)
            if (datetime > start_date) and (datetime < end_date)
        ]
    else:
        # Keep dates nearest to date range
        near_date_str_list = [
            get_nearest_date_str(date_str_list, selected_date_str)
            for selected_date_str in he_date_range
        ]
        idx_list = [date_str_list.index(near_date_str)
                    for near_date_str in near_date_str_list]
        fits_path_list = [fits_path_list[idx] for idx in idx_list]
    
    # Obtain date strings of FITS paths --------------------------------------
    fits_file_list = [fits_path.split('/')[-1]
                      for fits_path in fits_path_list]
    date_str_list = [fits_file.split('.')[0] for fits_file in fits_file_list]
    date_str_list.sort()
    
    return date_str_list


def get_acwe_date_list(he_date_range):
    """Retrieve list of ACWE available date strings of Numpy zipped files
    in the specified date range.
    
    Args
        he_date_range: tuple of start and end date strings for He I data or
            list of He I data dates
    Returns
        List of FITS file paths in the specified date range.
    """
    # Retrieve FITS path list in the specified date range --------------------
    glob_pattern = ACWE_DIR + '*/*'
    npz_path_list = glob.glob(glob_pattern)

    acwe_format_date_str_list = [
        npz_path.split('/')[-1].split('.')[ACWE_SPLIT_FILE_NAME_DATE_IDX]
        for npz_path in npz_path_list
    ]
    datetime_list = [
        datetime.strptime(date_str, ACWE_DICT_DATE_STR_FORMAT)
        for date_str in acwe_format_date_str_list
    ]
    date_str_list = [
        datetime.strftime(d, DICT_DATE_STR_FORMAT)
        for d in datetime_list
    ]

    if isinstance(he_date_range, tuple):
        # Keep only dates in date range
        start_date = datetime.strptime(he_date_range[0], DICT_DATE_STR_FORMAT)
        end_date = datetime.strptime(he_date_range[1], DICT_DATE_STR_FORMAT)
        
        filtered_date_str_list = [
            date_str for date_str, d
            in zip(date_str_list, datetime_list)
            if (d > start_date) and (d < end_date)
        ]
    else:
        # Keep dates nearest to date range
        filtered_date_str_list = [
            get_nearest_date_str(date_str_list, selected_date_str)
            for selected_date_str in he_date_range
        ]

    filtered_date_str_list.sort()
        
    return filtered_date_str_list


def rename_dir(data_dir, remove_gzip=False):
    """Rename all FITS files to include observation date in title
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


# Sunpy Map Operations -------------------------------------------------------
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


def get_acwe_sunpy_map(acwe_date_str, acwe_date_list):
    """Retrieve a Sunpy map for the ACWE segmentation on the provided
    datetime.
    
    Args
        acwe_date_str: str for ACWE datetime
    Returns
        Sunpy map object.
    """
    acwe_npz_files = sorted(glob.glob(ACWE_DIR + '*/*'))
    acwe_npz_file = acwe_npz_files[acwe_date_list.index(acwe_date_str)]

    # Extract 3D array of shape (slice num, 512, 512)
    fits_header, acwe_header, acwe_slices_data = acweSaveSeg_v5.openSeg(
        acwe_npz_file
    )

    # ACWE confidence map via aggregation of slices taken during the
    # optimization in the segmentation procedure and normalizing
    acwe_confidence_map_data = np.flipud(
        np.sum(acwe_slices_data, axis=0)
        /float(len(acwe_header['BACKGROUND_WEIGHT']))
    )

    # Resize to EUV image scale.
    # To avoid resizing the map, a new header would need to be created as tje
    # reference coordinate acwe_map.reference_pixel cannot be changed with 
    # sunpy.map.PixelPair(ref_pixel, ref_pixel)
    resized_shape = tuple(
        np.array(acwe_confidence_map_data.shape)*acwe_header['RESIZE_PARAM']
    )
    acwe_map_data = transform.resize(
        acwe_confidence_map_data, resized_shape, order=1,
        preserve_range=True, anti_aliasing=True
    )
    
    # Create Sunpy map
    acwe_map = sunpy.map.Map(np.flipud(acwe_map_data), fits_header)
    
    # Remove off disk pixels
    all_hp_coords = sunpy.map.maputils.all_coordinates_from_map(acwe_map)
    on_disk_mask = sunpy.map.maputils.coordinate_is_on_solar_disk(
        all_hp_coords)
    acwe_map = sunpy.map.Map(
        np.where(on_disk_mask, acwe_map.data, np.nan), acwe_map.meta
    )
    
    # Crop EUV map to similar zoom level to other observations 
    acwe_map = acwe_map.submap(
        bottom_left=SkyCoord(
            Tx=-1024*u.arcsec, Ty=-1024*u.arcsec,
            frame=acwe_map.coordinate_frame
        ),
        top_right=SkyCoord(
            Tx=1024*u.arcsec, Ty=1024*u.arcsec,
            frame=acwe_map.coordinate_frame
        )
    )
    acwe_map.plot_settings['norm'] = None
    
    return acwe_map


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


# Data or Data Product Date Handling -----------------------------------------
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

    # Select nearest datetime in the past if outside small hour window
    datetime_diff = nearest_datetime - selected_datetime
    if datetime_diff > timedelta(hours=hr_window):
        datetimes_behind_selected_datetime = [
            datetime < selected_datetime for datetime in datetimes
        ]
        past_datetimes = datetimes[datetimes_behind_selected_datetime]
        if past_datetimes.size == 0:
            raise Exception(
                (f'No datetimes are available behind {selected_date_str} '
                + f'or ahead by <{hr_window} hours. Increase the `hr_window` '
                'argument.')
            )
        
        nearest_datetime = max(
            past_datetimes, key=lambda datetime: datetime - selected_datetime
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
