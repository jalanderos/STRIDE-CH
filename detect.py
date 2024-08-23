"""
Library of functions to detect coronal holes.
"""

import os
import sunpy.map
import numpy as np
import pandas as pd
from scipy import ndimage, stats
import astropy.units as u
from datetime import datetime
from sunpy.coordinates import frames
from astropy.coordinates import SkyCoord
from moviepy.video.io import ImageSequenceClip
from skimage import morphology, filters, exposure

from settings import *
import prepare_data

# Module variables
DICT_DATE_STR_FORMAT = '%Y_%m_%d__%H_%M'
HE_OBS_DATE_STR_FORMAT = '%Y-%m-%dT%H:%M:%S'
SOLAR_AREA = 4*np.pi* (1*u.solRad).to(u.Mm)**2
MIN_PX_SIZE = 5000
MIN_SIZE = 3E9*u.km**2
OUTCOME_KEY_LIST = [
    'area', 'cm_lat', 'cm_lon', 'cm_foreshort',
    'unsigned_flux', 'signed_flux',
    'mag_skew', 'unipolarity', 'grad_median'
]
V1_1_CLASSIFY_FEATURES = [
    'unipolarity', 'grad_median', 'cm_foreshort'
]

# Reprojection map shape scaling factors
# Aim to match Helioprojective scale within 0.01 tolerance
# to preserve resolution by increasing factors to increase resolution
# and reduce distance scale / pixel
CEA_X_SCALE_FACTOR = np.pi/2
CEA_Y_SCALE_FACTOR = 1


# Calculation Functions
def get_hist(array, bins_as_percent=True, n=1000):
    """Get counts of values in an array either as a percentage
    or value.
    
    Args
        array: Numpy array
        bins_as_percent: boolean to specify bin edges as percentage
            from minimum to maximum value or as direct values
        n: int number of bins
    Returns
        Array of count number and associated bin edges
    """
    histogram, bin_edges = np.histogram(array[~np.isnan(array)], bins=n)

    if bins_as_percent:
        array_min = np.nanmin(array)
        array_range = np.nanmax(array) - array_min
        bin_edges = (bin_edges - array_min)/array_range * 100
    
    return histogram, bin_edges


def get_peak_counts_loc(array, bins_as_percent=True):
    """Retrieve the location of peak counts in a histogram either
    as a percentage or value. Ignores the first and final bins.
    
    Args
        array: Numpy array
        bins_as_percent: boolean to specify bin edges as percentage
            from minimum to maximum value or as direct values
    Returns
        Percent or value of array at which the most counts occur.
    """
    hist, edges = get_hist(array, bins_as_percent=bins_as_percent)
    peak_counts_idx = np.argmax(hist[1:-1])
    peak_counts_loc = (edges[peak_counts_idx] + edges[peak_counts_idx + 1])/2
    
    return peak_counts_loc


# Coordinate System Utilities
def get_hp_map_foreshort_factors(hp_map, detected_hg_coords):
    """Retrieve foreshortening factors for correcting disk center quantities
    from a Helioprojective map by division.
    
    Args
        hp_map: Sunpy map object in a Helioprojective coordinate frame
        detected_hg_coords: array of astropy quantities of Heliographic
            lon, lat coordinates
    Returns
        Foreshortening factors to divide disk center quantities by.
    """
    # B-angle to subtract from latitude
    B0 = hp_map.observer_coordinate.lat.to(u.rad).value
    
    # Foreshortening factors per detected pixel
    pixel_lons = detected_hg_coords.lon.to(u.rad).value
    pixel_lats = detected_hg_coords.lat.to(u.rad).value - B0
    foreshort_factors = np.cos(pixel_lons)*np.cos(pixel_lats)
    
    return foreshort_factors


def get_dist_scales(sunpy_map):
    """Retrieve distance scales per pixel in the x and y directions in Mm.
    
    Args
        sunpy_map: Sunpy map object
    Returns
        Astropy quantities of distance per pixel.
    """
    if sunpy_map.coordinate_frame.name == 'helioprojective':
        # Angular scale/px: (HP Tx, Ty arcsec)
        hp_delta_coords = frames.Helioprojective(
            sunpy_map.scale.axis1*u.pix,
            sunpy_map.scale.axis2*u.pix,
            observer='earth', obstime=sunpy_map.date
        )
        
        # Distance scale/px: (HP Tx, Ty arcsec) > (HC Distance in Mm)
        hc_delta_coords = hp_delta_coords.transform_to(
            frames.Heliocentric(observer='earth', obstime=sunpy_map.date)
        )
        x_scale = hc_delta_coords.x.to(u.Mm)
        y_scale = hc_delta_coords.y.to(u.Mm)
        
    elif sunpy_map.coordinate_frame.name == 'heliographic_stonyhurst':
        # Angular scale/px: (HG lon, lat deg)/pix > (HG lon, lat deg)
        lon_scale = sunpy_map.scale.axis1.to(u.deg/u.pix) * u.pix
        lat_scale = sunpy_map.scale.axis2.to(u.deg/u.pix) * u.pix

        # Distance scale/px: (HG lon, lat deg) > (Distance in Mm)
        x_scale = (sunpy_map.rsun_meters * lon_scale.to(u.rad)
                    /u.rad).to(u.Mm)
        y_scale = (sunpy_map.rsun_meters * lat_scale.to(u.rad)
                    /u.rad).to(u.Mm)
    
    return x_scale, y_scale


def get_A_per_square_px(sunpy_map):
    """Retrieve area covered by a normal to line of sight square pixel
    in Mm^2. Helioprojective angular change per pixel is converted to
    distance change per pixel in a Heliocentric frame.
    
    Args
        sunpy_map: Sunpy map object.
    Returns
        Astropy quantity of square area per pixel.
    """
    accepted_coord_frames = ['helioprojective', 'heliographic_stonyhurst']
    if sunpy_map.coordinate_frame.name not in accepted_coord_frames:
        raise Exception(('Coordinate frame not recognized for obtaining '
                        'area per square pixel.'))
        
    x_scale, y_scale = get_dist_scales(sunpy_map)
    A_per_square_px = np.abs(x_scale*y_scale)
    
    return A_per_square_px


def get_detected_hg_coords(ensemble_map, confidence_level):
    """Retrieve array of Heliographic longitude, latitude coordinates
    for each detected pixel at a given confidence level, as well as indices
    of failed coordinate conversion for removal from further data.
    
    Args
        ensemble_map: Sunpy map object of ensemble detection map
        confidence_level: confidence level at which to threshold ensemble maps
            for computing area
    Returns
        Array of astropy quantities of Heliographic lon, lat coordinates.
        Array of failed coordinate conversion indices for further inspection.
    """
    if confidence_level <= 0:
        confidence_level = 1e-3
    
    # Detected pixels at a confidence level
    # Flip upside down to align Sunpy coordinates and Numpy indices
    detected_px_coords = np.where(
        np.flipud(ensemble_map.data) >= confidence_level
    )

    if ensemble_map.coordinate_frame.name == 'helioprojective':
        
        # Convert detected pixels to Helioprojective Tx, Ty
        detected_hp_coords = ensemble_map.pixel_to_world(
            detected_px_coords[1]*u.pix, detected_px_coords[0]*u.pix
        )

        # Convert detected Helioprojective Tx, Ty to Heliographic lon, lat
        raw_detected_hg_coords = detected_hp_coords.transform_to(
            frames.HeliographicStonyhurst(obstime=ensemble_map.date)
        )

        # Remove pixels with failed conversion and longitudes outside (-90,90)
        failed_coord_idxs = np.where(
            np.isnan(raw_detected_hg_coords.lon) 
            | (np.abs(raw_detected_hg_coords.lon.to(u.deg).value) >= 90)
        )
        detected_hg_coords = np.delete(
            raw_detected_hg_coords, failed_coord_idxs
        )
    else:
        
        # Convert detected pixels to Heliographic lon, lat
        detected_hg_coords = ensemble_map.pixel_to_world(
            detected_px_coords[1]*u.pix, detected_px_coords[0]*u.pix
        )
        failed_coord_idxs = np.array([], dtype=np.int8)
    
    return detected_hg_coords, failed_coord_idxs


# Pre Processing Functions
def pre_process_v0_1(raw_he, peak_count_cutoff_percent=0.1):
    """Pre-process equivalent width array by setting background to NaN
    and cutting off at some percentage of histogram peak.
    
    Args
        peak_count_cutoff_percent: Vertical cutoff for histogram
            counts below a percentage
    """
    he_nan = remove_background(raw_he)

    hist, edges = get_hist(he_nan)
    max_count = np.max(hist)

    cutoff = max_count*peak_count_cutoff_percent/100
    
    array_range = np.nanmax(he_nan) - np.nanmin(he_nan)
    val = edges[:-1]/100 * array_range + np.nanmin(he_nan)
    
    cutoff_edges = np.where(hist > cutoff, val, 0)

    he_high_cut = np.where(he_nan > np.max(cutoff_edges), np.NaN, he_nan)

    he_band_cut = np.clip(
        he_high_cut, np.min(cutoff_edges), np.max(cutoff_edges)
    )
    
    return he_band_cut, he_high_cut, he_nan


def pre_process_v0_4(raw_he):
    """Pre-process equivalent width array by applying linear rescaling
    to normalize the contrast and setting background to NaN. Produces a
    less harsh contrast enhancement than histogram equalization.
    
    Args
        raw_he: He I equivalent width Numpy array
    Returns
        Pre-processed He I equivalent width Numpy array
    """
    # Rescale data to hold only intensities between the 2nd and 98th
    # percentiles. 
    p2, p98 = np.percentile(raw_he[~np.isnan(raw_he)], (2, 98))
    he = exposure.rescale_intensity(raw_he, in_range=(p2, p98))
    
    he = remove_background(he)
        
    return he


def remove_background(array):
    """Retrieve an array with the background replaced with nan.
    Assumes a uniform background value.
    
    Args
        array: Numpy array
    Returns
        Array with background set to nan.
    """
    background_val = array[0,0]    
    return np.where(array == background_val, np.nan, array)


def pre_process_v0_5_1(he_map):
    """Pre-process equivalent width array by applying linear rescaling
    to normalize the contrast and setting background to NaN. Produces a
    less harsh contrast enhancement than histogram equalization.
    
    Args
        raw_he: He I equivalent width Numpy array
    Returns
        Pre-processed He I equivalent width Numpy array
    """
    he_map_data = he_map.data
    
    # Rescale data to hold only intensities between the 2nd and 98th
    # percentiles. 
    p2, p98 = np.percentile(he_map_data[~np.isnan(he_map_data)], (2, 98))
    pre_processed_map_data = exposure.rescale_intensity(
        he_map_data, in_range=(p2, p98)
    )
    
    # Remove off limb pixels
    all_hp_coords = sunpy.map.maputils.all_coordinates_from_map(he_map)
    on_disk_mask = sunpy.map.maputils.coordinate_is_on_solar_disk(all_hp_coords)
    
    # TODO: .data redundant?
    pre_processed_map = sunpy.map.Map(
        np.where(on_disk_mask, pre_processed_map_data.data, np.nan), he_map.meta
    )
        
    return pre_processed_map


def pre_process_vY(he_map):
    """Pre-process a He I equivalent width observation by applying
    linear rescaling and reprojecting to a Heliographic Stonyhurst
    frame.
    
    Args
        he_map: Sunpy map object of He I equivalent width
    Returns
        Pre-processed He I equivalent width Sunpy map object.
    """
    # Rescale data to hold only intensities between the 2nd and 98th
    # percentiles.
    he_data = he_map.data
    p2, p98 = np.percentile(he_data[~np.isnan(he_data)], (2, 98))
    pre_processed_data = exposure.rescale_intensity(
        he_data, in_range=(p2, p98)
    )
    pre_processed_map = sunpy.map.Map(pre_processed_data, he_map.meta)
    
    reprojected_map = reproject_to_cea(pre_processed_map)
    
    return reprojected_map


def reproject_to_cea(sunpy_map):
    """Reproject a Sunpy map to a Heliographic Stonyhurst Cylindrical
    Equal Area projection with sine latitude, longitude coordinates.
    Crop map to within 90 degrees of the central meridian.
    
    Args
        sunpy_map: Sunpy map object
    Returns
        Reprojected Sunpy map object.
    """
    # Obtain dimension in image pixel number of the solar radius
    Rs_hp_coord = SkyCoord(
        sunpy_map.rsun_obs, 0*u.arcsec, frame='helioprojective',
        observer='earth', obstime=sunpy_map.date
    )
    Rs_pixel_pair = sunpy_map.world_to_pixel(Rs_hp_coord)
    ref_pixel_pair = sunpy_map.world_to_pixel(sunpy_map.reference_coordinate)
    Rs_dim = int((Rs_pixel_pair.x - ref_pixel_pair.x).value)
    
    # Create coordinate frame header
    new_row_num = int(2*Rs_dim*CEA_Y_SCALE_FACTOR)
    new_col_num = int(4*Rs_dim*CEA_X_SCALE_FACTOR)
    hg_header = sunpy.map.header_helper.make_heliographic_header(
        sunpy_map.date, sunpy_map.observer_coordinate,
        shape=(new_row_num, new_col_num), frame='stonyhurst',
        projection_code='CEA'
    )

    # Specify Earth-based observer for solar radius, distance to Sun,
    # and Heliographic coordinates to avoid warning messages due to
    # missing keywords.
    earth_hp_coords = frames.Helioprojective(
        Tx=0*u.arcsec, Ty=0*u.arcsec,
        observer='earth', obstime=sunpy_map.date,
    )
    earth_header = sunpy.map.make_fitswcs_header((1,1), earth_hp_coords)
    for earth_coord_key in ['RSUN_REF', 'DSUN_OBS', 'HGLN_OBS', 'HGLT_OBS']:
        hg_header[earth_coord_key] = earth_header[earth_coord_key]
    
    reprojected_map = sunpy_map.reproject_to(
        hg_header, algorithm='adaptive'
    )
    
    # Crop map to within 90 degrees of the central meridian
    top_right = SkyCoord(
        lon=90*u.deg, lat=90*u.deg, frame=reprojected_map.coordinate_frame
    )
    bottom_left = SkyCoord(
        lon=-90*u.deg, lat=-90*u.deg, frame=reprojected_map.coordinate_frame
    )
    reprojected_map = reprojected_map.submap(bottom_left, top_right=top_right)

    return reprojected_map


# Segmentation Functions
def get_thresh_bound(array, percent_of_peak):
    """Retrieve the bound for a binary threshold located at a
    percentage of the histogram peak.
    
    Args
        array: Numpy array
        percent_of_peak: float percentage measured from the zero
            value up to or beyond the histogram value
    Returns
        Array value at which to threshold.
    """
    array_min = np.nanmin(array)
    array_range = np.nanmax(array) - array_min
    
    # Percentages of zero and peak histogram values
    zero_percent = (0 - array_min)/array_range * 100
    peak_percent = get_peak_counts_loc(array, bins_as_percent=True)
    
    thresh_percent = percent_of_peak/100 * (peak_percent - zero_percent) \
        + zero_percent
    thresh_bound = thresh_percent/100 * array_range + array_min
    
    return thresh_bound


def morph(array, morph_radius):
    """Perform morphological operations on a given array with an
    approximation to a disk structuring element of specified radius.
    The true elements applied are a sequence decomposition for reduced
    computational cost.
    
    Args
        array: Numpy array
        morph_radius: int pixel number for radius of approximated 
            disk structuring element in morphological operations
    Returns
        Array that has been processed via morphological opening and closing.
    """
    disk = morphology.disk(int(morph_radius), decomposition='sequence')
    
    open_array = morphology.binary_opening(array, disk)
    close_array = morphology.binary_closing(open_array, disk)
    
    return close_array


def fill_rm(array, min_size):
    """Fill small holes within regions and remove small holes outside regions.
    
    Args
        array: Numpy array
        min_size: int pixel number for area of smallest region to preserve
    Returns
        Array that has been processed via fill and removal of small holes.
    """
    fill_array = ndimage.binary_fill_holes(array)
    rm_array = morphology.remove_small_objects(fill_array, min_size=min_size)
    
    return rm_array


def get_ch_mask(array, percent_of_peak, morph_radius, min_size=MIN_PX_SIZE):
    """Retrieve a single segmentation. Applicable up to v0.5.
    
    Args
        array: image to process
        percent_of_peak: float percentage measured from the zero
            value up to or beyond the histogram value
        morph_radius: int pixel number for radius of disk structuring element
            in morphological operations
        min_size: int pixel number for area of smallest region to preserve
    Returns
        Binary coronal holes mask.
    """
    thresh_bound = get_thresh_bound(array, percent_of_peak)
    ch_mask = np.where(array > thresh_bound, 1, 0)
    ch_mask = morph(ch_mask, morph_radius)
    ch_mask = fill_rm(ch_mask, min_size)

    return ch_mask


def get_ch_mask_list_v0_5_1(pre_processed_map, percent_of_peak_list,
                            morph_radius_dist_list):
    """Retrieve a list of segmentations from a pre-processed Sunpy map.
    
    Args
        pre_processed_map: Sunpy map object to segment
        percent_of_peak_list: list of float percentage measured from the zero
            value up to or beyond the histogram value
        morph_radius_dist_list: list of float distances in Mm for radius of
            disk structuring element in morphological operations
    Returns
        Binary coronal holes mask list.
    """
    pre_processed_map_data = np.flipud(pre_processed_map.data)
    
    # Obtain distance to map pixel number conversion factors
    x_scale, y_scale = get_dist_scales(pre_processed_map)
    dist_per_px = np.mean([x_scale.value, y_scale.value])
    A_per_square_px = x_scale.value*y_scale.value
    
    # Convert distance variables to pixels
    morph_radius_px_list = morph_radius_dist_list/dist_per_px
    min_size_px = MIN_SIZE.to(u.Mm**2).value/A_per_square_px
    
    ch_mask_list = []
    
    for percent_of_peak, morph_radius_px in zip(
        percent_of_peak_list, morph_radius_px_list):
        
        thresh_bound = get_thresh_bound(
            pre_processed_map_data, percent_of_peak
        )
        ch_mask = np.where(pre_processed_map_data > thresh_bound, 1, 0)
        ch_mask = morph(ch_mask, morph_radius_px)
        ch_mask = fill_rm(ch_mask, min_size_px)
        ch_mask_list.append(ch_mask)

    return ch_mask_list


def get_ch_mask_list_vY(pre_processed_map, percent_of_peak_list,
                          morph_radius_dist_list):
    """Retrieve a list of segmentations from a pre-processed Sunpy map.
    
    Args
        pre_processed_map: Sunpy map object to segment
        percent_of_peak_list: list of float percentage measured from the zero
            value up to or beyond the histogram value
        morph_radius_dist_list: list of float distances in Mm for radius of
            disk structuring element in morphological operations
    Returns
        Binary coronal holes mask list.
    """
    pre_processed_map_data = np.flipud(pre_processed_map.data)
    
    # Obtain distance to map pixel number conversion factors
    x_scale, y_scale = get_dist_scales(pre_processed_map)
    dist_per_px = np.mean([x_scale.value, y_scale.value])
    A_per_square_px = x_scale.value*y_scale.value
    
    # Convert distance variables to pixels
    morph_radius_px_list = morph_radius_dist_list/dist_per_px
    min_size_px = MIN_SIZE.to(u.Mm**2).value/A_per_square_px
    
    ch_mask_list = []
    
    for percent_of_peak, morph_radius_px in zip(
        percent_of_peak_list, morph_radius_px_list):
        
        thresh_bound = get_thresh_bound(
            pre_processed_map_data, percent_of_peak
        )
        ch_mask = np.where(pre_processed_map_data > thresh_bound, 1, 0)
        ch_mask = morph(ch_mask, morph_radius_px)
        ch_mask = fill_rm(ch_mask, min_size_px)
        ch_mask_list.append(ch_mask)

    return ch_mask_list


# Ensemble Map Functions
def get_masked_candidates(array, ch_mask):
    """Retrieve masked array from a single segmentation.
    
    Args
        array: image to process
        ch_mask: binary coronal holes mask
    Returns
        Masked array of coronal holes mask on image.
    """
    inverted_ch_mask = np.where(ch_mask == 1, 0, 1)

    masked_candidates = np.ma.array(array, mask=inverted_ch_mask)
    
    return masked_candidates


def get_map_data_by_ch(map_data, ch_mask):
    """Retrieve a list of arrays with map data of distinct CHs.
    
    Args
        map_data: array of map data to fill distinct CH arrays
        ch_mask: binary coronal holes mask
    Returns
        List of isolated CH images from a segmentation.
    """
    # Array with number labels for separate detected CHs and number of CHs
    labeled_candidates, num_ch = ndimage.label(ch_mask)
    
    map_data_by_ch = [
        np.where(labeled_candidates == ch_num + 1, map_data, np.nan)
        for ch_num in range(num_ch)
    ]
    return map_data_by_ch


def get_ensemble_v0_2(array, percent_of_peak_list, morph_radius_list):
    """Retrieve an ensemble of segmentations sorted by CH pixel number
    detected.
    
    Args
        array: image to process
        percent_of_peak_list: list of float percentage values
            at which to take threshold
        morph_radius_list: list of int pixel number for radius of disk 
            structuring element in morphological operations
    Returns
        Ensemble greyscale coronal holes mask.
        List of binary coronal holes masks.
        List of confidence levels in mask layers.
        Confidence assignment metric list of pixel percentages.
    """
    ch_mask_list = [
        get_ch_mask(array, percent_of_peak, morph_radius)
        for percent_of_peak, morph_radius
        in zip(percent_of_peak_list, morph_radius_list)
    ]
    
    num_masks = len(ch_mask_list)
    confidence_list = [(c + 1)/num_masks *100
                       for c in range(num_masks)]

    # Sort masks by pixel percentage detected
    px_percent_list = get_px_percent_list(ch_mask_list)
    sorted_idxs = np.flip(np.argsort(px_percent_list))
    
    # Sort candidate CHs from greatest to least detected pixel percentage
    ch_mask_list = [ch_mask_list[i] for i in sorted_idxs]
    px_percent_list = [px_percent_list[i] for i in sorted_idxs]

    ensemble_map = np.where(~np.isnan(array), 0, np.nan)
    
    for ch_mask, confidence in zip(ch_mask_list, confidence_list):
        ensemble_map = np.where(
            ch_mask == 1, confidence, ensemble_map
        )
    return ensemble_map, ch_mask_list, confidence_list, px_percent_list


def get_ensemble_v0_3(pre_processed_map_data, percent_of_peak_list,
                      morph_radius_list):
    """Retrieve an ensemble of segmentations sorted by CH smoothness.
    
    Args
        pre_processed_map_data: map data to operate on
        percent_of_peak_list: list of float percentage values
            at which to take threshold
        morph_radius_list: list of int pixel number for radius of disk 
            structuring element in morphological operations
    Returns
        Ensemble greyscale coronal holes mask sorted by median gradient.
        List of coronal holes masks.
        List of confidence levels in mask layers.
        Confidence assignment metric list of gradient medians.
    """
    # Create global segmentations for varied design variable combinations
    ch_masks = [
        get_ch_mask(pre_processed_map_data, percent_of_peak, morph_radius)
        for percent_of_peak, morph_radius
        in zip(percent_of_peak_list, morph_radius_list)
    ]
    
    # Lists to hold pre processed map and gradient data respectively
    # for distinct CHs from all segmentations
    map_data_by_ch = []
    grad_data_by_ch = []
    
    for ch_mask in ch_masks:
        # Masked array of candidate CHs
        masked_candidates = get_masked_candidates(pre_processed_map_data, ch_mask)
        
        # Compute spatial gradient
        gradient_candidates = filters.sobel(masked_candidates)
        
        map_data_by_ch.extend(
            get_map_data_by_ch(pre_processed_map_data, ch_mask)
        )
        grad_data_by_ch.extend(
            get_map_data_by_ch(gradient_candidates, ch_mask)
        )
    
    # Obtain sorting indixes from greatest to least gradient median
    gradient_medians = [np.median(grad_data[~np.isnan(grad_data)])
                        for grad_data in grad_data_by_ch]
    sorted_idxs = np.flip(np.argsort(gradient_medians))
    
    # Sort candidate CHs from greatest to least gradient median
    map_data_by_ch = [map_data_by_ch[i] for i in sorted_idxs]
    gradient_medians = [gradient_medians[i] for i in sorted_idxs]
    
    # Assign confidence by direct ranking
    num_ch = len(map_data_by_ch)
    confidence_list = [(c + 1)*100/num_ch
                       for c in range(num_ch)]

    # Construct ensemble map by adding distinct CHs with assigned
    # confidence level values to an empty base disk
    ensemble_map = np.where(~np.isnan(pre_processed_map_data),
                            0, np.nan)
    
    for distinct_ch, confidence in zip(map_data_by_ch, confidence_list):
        ensemble_map = np.where(
            ~np.isnan(distinct_ch), confidence, ensemble_map
        )
    return ensemble_map, map_data_by_ch, confidence_list, gradient_medians


def get_ensemble_v0_5(pre_processed_map, reprojected_mag_map,
                      percent_of_peak_list, morph_radius_list,
                      unipolarity_threshold):
    """Retrieve an ensemble of segmentations sorted by CH unipolarity.
    
    Args
        pre_processed_map: Sunpy map object to segment
        reprojected_mag_map: Sunpy map object of magnetogram reprojected
            to align with the ensemble map
        percent_of_peak_list: list of float percentage values
            at which to take threshold on pre-processed map
        morph_radius_list: list of int pixel number for radius of disk 
            structuring element in morphological operations
        unipolarity_threshold: float unipolarity in [0,1) at which to
            threshold candidate CHs
    Returns
        Ensemble greyscale coronal holes mask sorted by unipolarity.
        List of coronal holes masks.
        List of confidence levels in mask layers.
        Confidence assignment metric list of unipolarity.
    """
    pre_processed_map_data = np.flipud(pre_processed_map.data)
    
    # Create global segmentations for varied design variable combinations
    ch_masks = [
        get_ch_mask(pre_processed_map_data, percent_of_peak, morph_radius)
        for percent_of_peak, morph_radius
        in zip(percent_of_peak_list, morph_radius_list)
    ]
    
    # List to be extended by masks for distinct CHs from all segmentations
    masks_by_ch = []
    
    ones_array = np.ones_like(pre_processed_map_data)
    
    for ch_mask in ch_masks:
        masks_by_ch.extend(
            get_map_data_by_ch(ones_array, ch_mask)
        )
    
    num_ch = len(masks_by_ch)
    
    # Compute constant area per square pixel once for all CHs
    A_per_square_px = get_A_per_square_px(pre_processed_map)
    
    # List to hold unipolarity for distinct CHs from all segmentations
    unipolarity_by_ch = []
    
    for ch_label in range(num_ch):
        distinct_ch_mask = masks_by_ch[ch_label]
        
        # Not flipping works right
        distinct_ch_map = sunpy.map.Map(
            distinct_ch_mask, pre_processed_map.meta
        )
        outcome_dict = get_outcomes(
            distinct_ch_map, reprojected_mag_map, A_per_square_px
        )
        unipolarity_by_ch.append(outcome_dict['unipolarity'])
    
    # Sort unipolarities from greatest to least
    sorted_idxs = np.argsort(unipolarity_by_ch)
    
    # Sort candidate CHs from greatest to least gradient median
    masks_by_ch = [masks_by_ch[i] for i in sorted_idxs]
    unipolarity_by_ch = [unipolarity_by_ch[i] for i in sorted_idxs]
    
    confidence_levels = np.array(
        [unipolarity*100 for unipolarity in unipolarity_by_ch]
    )
    confidence_levels = np.where(
        confidence_levels >= unipolarity_threshold*100, confidence_levels, 0
    )

    # Construct ensemble map by adding distinct CHs with assigned
    # confidence level values to an empty base disk    
    ensemble_map_data = np.where(
        ~np.isnan(pre_processed_map_data), 0, np.nan
    )
    for distinct_ch, confidence in zip(masks_by_ch, confidence_levels):
        ensemble_map_data = np.where(
            ~np.isnan(distinct_ch), confidence, ensemble_map_data
        )
    return ensemble_map_data, masks_by_ch, confidence_levels, unipolarity_by_ch


def get_ensemble_v0_5_1(pre_processed_map, reprojected_mag_map,
                        percent_of_peak_list, morph_radius_dist_list,
                        unipolarity_threshold):
    """Retrieve an ensemble of segmentations sorted by CH unipolarity.
    
    Args
        pre_processed_map: Sunpy map object to segment
        reprojected_mag_map: Sunpy map object of magnetogram reprojected
            to align with the ensemble map
        percent_of_peak_list: list of float percentage measured from the zero
            value up to or beyond the histogram value
        morph_radius_dist_list: list of float distances in Mm for radius of
            disk structuring element in morphological operations
        unipolarity_threshold: float unipolarity in [0,1) at which to
            threshold candidate CHs
    Returns
        Ensemble greyscale coronal holes mask sorted by unipolarity.
        List of coronal holes masks.
        List of confidence levels in mask layers.
        Confidence assignment metric list of unipolarity.
    """
    pre_processed_map_data = np.flipud(pre_processed_map.data)
    
    # Create global segmentations for varied design variable combinations
    ch_mask_list = get_ch_mask_list_v0_5_1(
        pre_processed_map, percent_of_peak_list, morph_radius_dist_list
    )
    
    # List to be extended by masks for distinct CHs from all segmentations
    masks_by_ch = []
    
    ones_array = np.ones_like(pre_processed_map_data)
    
    for ch_mask in ch_mask_list:
        masks_by_ch.extend(
            get_map_data_by_ch(ones_array, ch_mask)
        )
    
    num_ch = len(masks_by_ch)
    
    # Compute constant area per square pixel once for all CHs
    A_per_square_px = get_A_per_square_px(pre_processed_map)
    
    # List to hold unipolarity for distinct CHs from all segmentations
    unipolarity_by_ch = []
    
    for ch_label in range(num_ch):
        distinct_ch_mask = masks_by_ch[ch_label]
        
        # Not flipping works right
        distinct_ch_map = sunpy.map.Map(
            distinct_ch_mask, pre_processed_map.meta
        )
        outcome_dict = get_outcomes(
            distinct_ch_map, reprojected_mag_map, A_per_square_px
        )
        unipolarity_by_ch.append(outcome_dict['unipolarity'])
    
    # Sort unipolarities from greatest to least
    sorted_idxs = np.argsort(unipolarity_by_ch)
    
    # Sort candidate CHs from greatest to least gradient median
    masks_by_ch = [masks_by_ch[i] for i in sorted_idxs]
    unipolarity_by_ch = [unipolarity_by_ch[i] for i in sorted_idxs]
    
    # Assign confidence by unipolarity above a threshold 
    confidence_levels = np.array(
        [unipolarity*100 for unipolarity in unipolarity_by_ch]
    )
    confidence_levels = np.where(
        confidence_levels >= unipolarity_threshold*100, confidence_levels, 0
    )

    # Construct ensemble map by adding distinct CHs with assigned
    # confidence level values to an empty base disk    
    ensemble_map_data = np.where(
        ~np.isnan(pre_processed_map_data), 0, np.nan
    )
    for distinct_ch, confidence in zip(masks_by_ch, confidence_levels):
        ensemble_map_data = np.where(
            ~np.isnan(distinct_ch), confidence, ensemble_map_data
        )
    return ensemble_map_data, masks_by_ch, confidence_levels, unipolarity_by_ch


def get_ensemble_vY(pre_processed_map, reprojected_mag_map,
                      percent_of_peak_list, morph_radius_dist_list,
                      unipolarity_threshold):
    """Retrieve an ensemble of segmentations sorted by CH unipolarity.
    
    Args
        pre_processed_map: Sunpy map object to segment
        reprojected_mag_map: Sunpy map object of magnetogram reprojected
            to align with the ensemble map
        percent_of_peak_list: list of float percentage measured from the zero
            value up to or beyond the histogram value
        morph_radius_dist_list: list of float distances in Mm for radius of
            disk structuring element in morphological operations
        unipolarity_threshold: float unipolarity in [0,1) at which to
            threshold candidate CHs
    Returns
        Ensemble greyscale coronal holes mask sorted by unipolarity.
        List of coronal holes masks.
        List of confidence levels in mask layers.
        Confidence assignment metric list of unipolarity.
    """
    pre_processed_map_data = np.flipud(pre_processed_map.data)
    
    # Create global segmentations for varied design variable combinations
    ch_mask_list = get_ch_mask_list_vY(
        pre_processed_map, percent_of_peak_list, morph_radius_dist_list
    )
    
    # List to be extended by masks for distinct CHs from all segmentations
    masks_by_ch = []
    
    ones_array = np.ones_like(pre_processed_map_data)
    
    for ch_mask in ch_mask_list:
        masks_by_ch.extend(
            get_map_data_by_ch(ones_array, ch_mask)
        )
    
    num_ch = len(masks_by_ch)
    
    # Compute constant area per square pixel once for all CHs
    A_per_square_px = get_A_per_square_px(pre_processed_map)
    
    # List to hold unipolarity for distinct CHs from all segmentations
    unipolarity_by_ch = []
    
    for ch_label in range(num_ch):
        distinct_ch_mask = masks_by_ch[ch_label]
        
        # Not flipping works right
        distinct_ch_map = sunpy.map.Map(
            distinct_ch_mask, pre_processed_map.meta
        )
        outcome_dict = get_outcomes(
            distinct_ch_map, reprojected_mag_map, A_per_square_px
        )
        unipolarity_by_ch.append(outcome_dict['unipolarity'])
    
    # Sort unipolarities from greatest to least
    sorted_idxs = np.argsort(unipolarity_by_ch)
    
    # Sort candidate CHs from greatest to least gradient median
    masks_by_ch = [masks_by_ch[i] for i in sorted_idxs]
    unipolarity_by_ch = [unipolarity_by_ch[i] for i in sorted_idxs]
    
    # Assign confidence by unipolarity above a threshold 
    confidence_levels = np.array(
        [unipolarity*100 for unipolarity in unipolarity_by_ch]
    )
    confidence_levels = np.where(
        confidence_levels >= unipolarity_threshold*100, confidence_levels, 0
    )

    # Construct ensemble map by adding distinct CHs with assigned
    # confidence level values to an empty base disk    
    ensemble_map_data = np.where(
        ~np.isnan(pre_processed_map_data), 0, np.nan
    )
    for distinct_ch, confidence in zip(masks_by_ch, confidence_levels):
        ensemble_map_data = np.where(
            ~np.isnan(distinct_ch), confidence, ensemble_map_data
        )
    return ensemble_map_data, masks_by_ch, confidence_levels, unipolarity_by_ch


def get_ensemble_v1_0(pre_processed_map, reprojected_mag_map,
                      percent_of_peak_list, morph_radius_dist_list,
                      unipolarity_threshold):
    """Retrieve an ensemble of segmentations sorted by CH unipolarity.
    
    Args
        pre_processed_map: Sunpy map object to segment
        reprojected_mag_map: Sunpy map object of magnetogram reprojected
            to align with the ensemble map
        percent_of_peak_list: list of float percentage measured from the zero
            value up to or beyond the histogram value
        morph_radius_dist_list: list of float distances in Mm for radius of
            disk structuring element in morphological operations
        unipolarity_threshold: float unipolarity in [0,1) at which to
            threshold candidate CHs
    Returns
        Ensemble greyscale coronal holes mask sorted by unipolarity.
        List of coronal holes masks.
        List of confidence levels in mask layers.
        Confidence assignment metric list of unipolarity.
    """
    pre_processed_map_data = np.flipud(pre_processed_map.data)
    
    # Create global segmentations for varied design variable combinations
    ch_mask_list = get_ch_mask_list_v0_5_1(
        pre_processed_map, percent_of_peak_list, morph_radius_dist_list
    )
    
    # List to be extended by masks for distinct CHs from all segmentations
    masks_by_ch = []
    
    ones_array = np.ones_like(pre_processed_map_data)
    
    for ch_mask in ch_mask_list:
        masks_by_ch.extend(
            get_map_data_by_ch(ones_array, ch_mask)
        )
    
    num_ch = len(masks_by_ch)
    
    # Compute constant area per square pixel once for all CHs
    A_per_square_px = get_A_per_square_px(pre_processed_map)
    
    # List to hold unipolarity for distinct CHs from all segmentations
    unipolarity_by_ch = []
    
    for ch_label in range(num_ch):
        distinct_ch_mask = masks_by_ch[ch_label]
        
        # Not flipping works right
        distinct_ch_map = sunpy.map.Map(
            distinct_ch_mask, pre_processed_map.meta
        )
        outcome_dict = get_outcomes(
            distinct_ch_map, reprojected_mag_map, A_per_square_px
        )
        unipolarity_by_ch.append(outcome_dict['unipolarity'])
    
    # Sort unipolarities from greatest to least
    sorted_idxs = np.argsort(unipolarity_by_ch)
    
    # Sort candidate CHs from greatest to least gradient median
    masks_by_ch = [masks_by_ch[i] for i in sorted_idxs]
    unipolarity_by_ch = [unipolarity_by_ch[i] for i in sorted_idxs]
    
    # Assign confidence by unipolarity above a threshold
    confidence_levels = np.array(
        [(unipolarity - unipolarity_threshold)/(1 - unipolarity_threshold)
        for unipolarity in unipolarity_by_ch]
    )
    confidence_levels = np.where(
        confidence_levels > 0, confidence_levels, 0
    )

    # Construct ensemble map by adding distinct CHs with assigned
    # confidence level values to an empty base disk    
    ensemble_map_data = np.where(
        ~np.isnan(pre_processed_map_data), 0, np.nan
    )
    for distinct_ch, confidence in zip(masks_by_ch, confidence_levels):
        ensemble_map_data = np.where(
            ~np.isnan(distinct_ch), confidence, ensemble_map_data
        )
    return ensemble_map_data, masks_by_ch, confidence_levels, unipolarity_by_ch


def get_ensemble_v1_1(he_map_data, pre_processed_map, reprojected_mag_map,
                      percent_of_peak_list, morph_radius_dist_list, lda,
                      probability_threshold):
    """Retrieve an ensemble of segmentations sorted by CH unipolarity.
    
    Args
        he_map_data: Numpy array of He I observation
        pre_processed_map: Sunpy map object to segment
        reprojected_mag_map: Sunpy map object of magnetogram reprojected
            to align with the ensemble map
        percent_of_peak_list: list of float percentage measured from the zero
            value up to or beyond the histogram value
        morph_radius_dist_list: list of float distances in Mm for radius of
            disk structuring element in morphological operations
        lda: sklearn LinearDiscriminantAnalysis object to predict true CH
            probabilities of candidate regions
        probability_threshold: float probability in [0,1) at which to
            threshold candidate CHs
    Returns
        Ensemble greyscale coronal holes mask sorted by unipolarity.
        List of coronal holes masks.
        List of confidence levels in mask layers.
        Confidence assignment metric list of unipolarity.
    """
    # Create segmentation masks across the full solar disk of candidate
    # regions for varied design variable combinations
    full_disk_cand_mask_list = get_ch_mask_list_v0_5_1(
        pre_processed_map, percent_of_peak_list, morph_radius_dist_list
    )

    # List to be extended by masks for distinct CHs from all segmentations
    cand_masks = []
    ones_array = np.ones_like(he_map_data)

    for full_disk_cand_mask in full_disk_cand_mask_list:
        cand_masks_in_full_disk_mask = get_map_data_by_ch(
            ones_array, full_disk_cand_mask
        )
        cand_masks.extend(cand_masks_in_full_disk_mask)

    num_cand = len(cand_masks)

    # Compute constant area per square pixel once for all CHs
    A_per_square_px = get_A_per_square_px(pre_processed_map)

    # Array to hold candidate feature values
    cand_feature_array = np.zeros((num_cand, len(V1_1_CLASSIFY_FEATURES)))

    # TODO: Takes 10s, speed-up?
    for cand_idx in range(num_cand):
        distinct_cand_mask = cand_masks[cand_idx]
        distinct_cand_map = sunpy.map.Map(
            distinct_cand_mask, # Not flipping works right
            pre_processed_map.meta
        )
        outcome_dict = get_outcomes(
            distinct_cand_map, reprojected_mag_map, A_per_square_px,
            he_map_data=he_map_data
        )
        cand_feature_values = [
            outcome_dict[feature] for feature in V1_1_CLASSIFY_FEATURES
        ]
        cand_feature_array[cand_idx, :] = cand_feature_values

    cand_probabilities = lda.predict_proba(cand_feature_array)[:,1]

    # Sort unipolarities from greatest to least
    sorted_idxs = np.argsort(cand_probabilities)

    # Sort candidate regions from greatest to least predicted probability
    cand_masks = [cand_masks[i] for i in sorted_idxs]
    cand_probabilities = [
        cand_probabilities[i] for i in sorted_idxs
    ]
    # Assign confidence by probability above a threshold
    confidence_levels = np.array(
        [(probability - probability_threshold)/(1 - probability_threshold)
        for probability in cand_probabilities]
    )
    confidence_levels = np.where(
        confidence_levels > 0, confidence_levels, 0
    )

    # Construct ensemble map by adding distinct CHs with assigned
    # confidence level values to an empty base disk
    ensemble_map_data = np.where(
        ~np.isnan(np.flipud(pre_processed_map.data)), 0, np.nan
    )
    for distinct_cand, confidence in zip(cand_masks, confidence_levels):
        ensemble_map_data = np.where(
            ~np.isnan(distinct_cand), confidence, ensemble_map_data
        )
    return ensemble_map_data, cand_masks, confidence_levels, cand_feature_array


# Outcome Calculation Functions
def get_px_percent_list(ch_mask_list):
    """Retrieve the percentage of pixels detected in each segmentation
    in a list.
    
    Args
        array: image to process
        ch_mask_list: binary coronal holes mask list
    Returns
        List of pixel percentage detected in segmentations.
    """
    disk_px_count = np.count_nonzero(~np.isnan(ch_mask_list[0]))
    return [np.count_nonzero(ch_mask)*100/disk_px_count
            for ch_mask in ch_mask_list]


def get_num_ch(ch_mask):
    """Retrieve the number of CHs detected in a segmentation.
    
    Args
        ch_mask: binary coronal holes mask
    Returns
        Number of CHs detected in a segmentation.
    """    
    return ndimage.label(ch_mask)[1]


def get_open_area(ensemble_map, confidence_level):
    """Retrieve detected open area in an ensemble map at a given
    confidence level as a percentage and in Mm^2.
    
    Args
        ensemble_map: Sunpy map object of ensemble detection map
        confidence_level: confidence level at which to threshold
            ensemble maps for computing area
    Returns
        Detected area as a percentage of total solar surface area and in Mm^2.
    """
    A_per_square_px = get_A_per_square_px(ensemble_map)
    
    detected_hg_coords = get_detected_hg_coords(
        ensemble_map, confidence_level
    )[0]
    
    pixel_areas = get_pixel_areas(
        ensemble_map, A_per_square_px, detected_hg_coords
    )

    # Sum area detected in all pixels
    open_area = np.sum(pixel_areas)
    area_percent = open_area/SOLAR_AREA*100
    
    return area_percent.value, open_area.value


def get_unsigned_open_flux(ensemble_map, reprojected_mag_map,
                           confidence_level):
    """Retrieve detected unsigned open magnetic flux in an ensemble map
    at a given confidence level in Wb.
    
    Args
        ensemble_map: Sunpy map object of ensemble detection map
        reprojected_mag_map: Sunpy map object of magnetogram reprojected
            to align with the ensemble map
        confidence_level: confidence level at which to threshold
            ensemble maps for computing area
    Returns
        Detected magnetix flux in Wb.
    """
    A_per_square_px = get_A_per_square_px(ensemble_map)
        
    detected_hg_coords, failed_coord_idxs = get_detected_hg_coords(
        ensemble_map, confidence_level
    )

    pixel_areas = get_pixel_areas(
        ensemble_map, A_per_square_px, detected_hg_coords
    )

    pixel_B_LOS, failed_mag_idxs = get_pixel_B_LOS(
        ensemble_map, reprojected_mag_map, confidence_level, failed_coord_idxs
    )
    pixel_areas = np.delete(pixel_areas, failed_mag_idxs)
    
    unsigned_open_flux = np.sum(np.abs(pixel_B_LOS)*pixel_areas).to(u.Wb)
    
    return unsigned_open_flux.value


def get_pixel_areas(ensemble_map, A_per_square_px, detected_hg_coords):
    """Retrieve array of area per detected pixel in Mm^2 while
    accounting for foreshortening.
    
    Args
        ensemble_map: Sunpy map object of ensemble detection map
        A_per_square_px: Astropy quantity object of square area per pixel
        detected_hg_coords: array of astropy quantities of Heliographic
            lon, lat coordinates
    Returns
        Array of astropy quantities of areas per detected pixel.
    """
    if ensemble_map.coordinate_frame.name == 'helioprojective':
    
        foreshort_factors = get_hp_map_foreshort_factors(
            ensemble_map, detected_hg_coords
        )
        pixel_areas = A_per_square_px/foreshort_factors

    elif ensemble_map.coordinate_frame.name == 'heliographic_stonyhurst':
        pixel_areas = np.ones(detected_hg_coords.shape)*A_per_square_px
    
    return pixel_areas


def get_pixel_B_LOS(ensemble_map, reprojected_mag_map, confidence_level,
                    failed_coord_idxs):
    """Retrieve array of line of sight magnetic field strength per detected
    pixel in G as well as indices of failed retrieval for removal from
    further data.
    
    Detected pixel indices per CH are obtained, followed by magnetic field
    data at these indices. This is equivalent to obtaining Helioprojective
    coordinates of pixels per CH and then obtaining magnetic data at these
    coordinates due to prerequisite reprojection.
    
    Args
        ensemble_map: Sunpy map object of ensemble detection map
        reprojected_mag_map: Sunpy map object of magnetogram reprojected
            to align with the ensemble map
        confidence_level: confidence level at which to threshold ensemble maps
            for computing area
        failed_coord_idxs: array of indices of failed coordinate conversion
            from Helioprojective to Heliographic coordinates
    Returns
        Array of astropy quantities of magnetic field strengths per
            detected pixel.
        Array of failed magnetic data retrieval indices for further inspection.
    """
    if confidence_level <= 0:
        confidence_level = 1e-3
    
    # Magnetic field strength per detected pixel
    # Flip upside down to align Sunpy coordinates and Numpy indices
    detected_px_coords = np.where(
        np.flipud(ensemble_map.data) >= confidence_level
    )
    pixel_B_LOS = reprojected_mag_map.data[detected_px_coords]*u.G
    
    # Remove pixels with failed coordinate conversion
    pixel_B_LOS = np.delete(pixel_B_LOS, failed_coord_idxs)
    
    # Remove pixels with failed magnetic data retrieval
    failed_mag_idxs = np.where(np.isnan(pixel_B_LOS))
    pixel_B_LOS = np.delete(pixel_B_LOS, failed_mag_idxs)
    
    return pixel_B_LOS, failed_mag_idxs


# Outcome Collection Functions
def get_outcomes(ensemble_map, reprojected_mag_map, A_per_square_px,
                 confidence_level=0, he_map_data=[]):
    """Retrieve outcomes in an ensemble map at a given confidence level.
    
    Outcomes include detected open area in Mm^2, center of mass latitude
    and longitude in deg, unsigned open magnetic flux in Wb, signed open
    magnetic flux in Mx, magnetic flux skewness, unipolarity, and signed
    magnetic flux per pixel.
    
    Args
        ensemble_map: Sunpy map object of ensemble detection map
        reprojected_mag_map: Sunpy map object of magnetogram reprojected
            to align with the ensemble map
        A_per_square_px: astropy quantity of square area per pixel
        confidence_level: confidence level at which to threshold
            ensemble maps for computing area
        he_map_data: Numpy array of He I observation. Specify as []
            to avoid computing the smoothness metric
    Returns
        Dictionary with open area, center of mass latitude and longitude,
            unsigned, signed open flux, flux skewnwss, unipolarity,
            and flux per pixel.
    """
    outcome_dict = {}
    
    # Global variation outcomes ----------------------------------------------
    detected_hg_coords, failed_coord_idxs = get_detected_hg_coords(
        ensemble_map, confidence_level
    )
    pixel_areas = get_pixel_areas(
        ensemble_map, A_per_square_px, detected_hg_coords
    )
    
    # Sum area detected in all pixels
    outcome_dict['area'] = np.sum(pixel_areas).value
    
    # Center of mass coordinates
    cm_lat = (
        np.sum(detected_hg_coords.lat*pixel_areas)/np.sum(pixel_areas)
    )
    cm_lon = (
        np.sum(detected_hg_coords.lon*pixel_areas)/np.sum(pixel_areas)
    )
    B0 = ensemble_map.observer_coordinate.lat
    
    outcome_dict['cm_lat'] = cm_lat.value
    outcome_dict['cm_lon'] = cm_lon.value
    outcome_dict['cm_foreshort'] = (
        np.cos(cm_lon.to(u.rad).value)
        *np.cos(cm_lat.to(u.rad).value - B0.to(u.rad).value)
    )
    
    # Magnetic outcomes ------------------------------------------------------
    pixel_B_LOS, failed_mag_idxs = get_pixel_B_LOS(
        ensemble_map, reprojected_mag_map, confidence_level,
        failed_coord_idxs
    )
    pixel_areas_for_mag = np.delete(pixel_areas, failed_mag_idxs)
    
    # Global unsigned quantities
    pixel_unsigned_fluxes = (np.abs(pixel_B_LOS)*pixel_areas_for_mag).to(u.Wb)
    outcome_dict['unsigned_flux'] = np.sum(pixel_unsigned_fluxes).value
    
    # Per CH signed quantities
    pixel_signed_fluxes = (pixel_B_LOS*pixel_areas_for_mag).to(u.Mx).value
    outcome_dict['pixel_signed_fluxes'] = pixel_signed_fluxes
    outcome_dict['signed_flux'] = np.sum(pixel_signed_fluxes)
    outcome_dict['mag_skew'] = stats.skew((pixel_signed_fluxes))
    
    # Unipolarity ------------------------------------------------------------
    foreshort_factors = get_hp_map_foreshort_factors(
        ensemble_map, detected_hg_coords
    )
    foreshort_factors = np.delete(foreshort_factors, failed_mag_idxs)
    
    pixel_B_r = pixel_B_LOS/foreshort_factors
    signed_B_r = np.abs(np.nanmean(pixel_B_r))
    unsigned_B_r = np.nanmean(np.abs(pixel_B_r))
    outcome_dict['unipolarity'] = (signed_B_r/unsigned_B_r).value
    
    # Smoothness -------------------------------------------------------------
    if np.any(he_map_data):
        # Detected He I array masked at a confidence level
        detected_he_map_data = np.where(
            ensemble_map.data >= confidence_level, he_map_data, np.nan
        )
        
        # Compute median of spatial gradient on pre-processed He I
        detected_gradient_data = filters.sobel(detected_he_map_data)
        outcome_dict['grad_median'] = np.nanmedian(detected_gradient_data)
    else:
        outcome_dict['grad_median'] = np.nan
    
    return outcome_dict


def get_outcomes_by_ch(ensemble_map, he_map_data,
                       reprojected_mag_map, confidence_level):
    """Retrieve outcomes per CH in an ensemble map at a given confidence
    level.
    
    See get_outcomes for retrieved outcomes.
    
    Args
        ensemble_map: Sunpy map object of ensemble detection map
        he_map_data: Numpy array of He I observation
        reprojected_mag_map: Sunpy map object of magnetogram reprojected
            to align with the ensemble map
        confidence_level: confidence level at which to threshold
            ensemble maps for computing area
    Returns
        Dictionary with outcome keys, each with a list of outcomes per CH as
            its value.
    """
    if confidence_level <= 0:
        confidence_level = 1e-3
        
    # Compute constant area per square pixel once for all CHs
    A_per_square_px = get_A_per_square_px(ensemble_map)        

    # Mask of detected CHs at the given confidence level
    confidence_ch_mask = np.where(
        ensemble_map.data >= confidence_level, 1, 0
    )

    # List of ensemble map data for distinct CHs
    ensemble_map_data_by_ch = get_map_data_by_ch(
        ensemble_map.data, confidence_ch_mask
    )

    num_ch = len(ensemble_map_data_by_ch)
    
    outcome_by_ch_dict = {}
    for outcome_key in OUTCOME_KEY_LIST:
        outcome_by_ch_dict[outcome_key] = np.zeros(num_ch)
    
    outcome_by_ch_dict['pixel_signed_fluxes'] = []

    for ch_label in range(num_ch):
        
        distinct_ch_ensemble_map = sunpy.map.Map(
            np.flipud(ensemble_map_data_by_ch[ch_label]), ensemble_map.meta
        )
        outcome_dict = get_outcomes(
            distinct_ch_ensemble_map, reprojected_mag_map, A_per_square_px,
            confidence_level, he_map_data
        )
        for outcome_key in OUTCOME_KEY_LIST:
            outcome = outcome_dict[outcome_key]
            outcome_by_ch_dict[outcome_key][ch_label] = outcome
        
        outcome_by_ch_dict['pixel_signed_fluxes'].append(
            outcome_dict['pixel_signed_fluxes']
        )
        
    return outcome_by_ch_dict


def get_thresh_outcome_time_series_dfs(he_date_str_list, percent_of_peak_list):
    """Retrieve dataframes with thresholded map outcomes at specified
    threshold levels over time.
    
    Args
        he_date_str_list: list of date strings for ensemble maps
        percent_of_peak_list: list of float percentage values
            at which to take threshold
    Returns
        Dataframes of outcomes by confidence level over time.
    """
    # List for outcomes at varied confidence levels and datetimes
    num_ch_by_thresh_list = []
    area_percent_by_thresh_list = []
    area_by_thresh_list = []
    px_percent_by_thresh_list = []

    for he_date_str in he_date_str_list:
        
        he_file = f'{ALL_HE_DIR}{he_date_str}.fts'
        he_map = prepare_data.get_nso_sunpy_map(he_file)
        if not he_map:
            print(f'{he_date_str} He I observation extraction failed.')
            continue
        
        # Extract saved pre-processed map
        pre_process_file = (PREPROCESS_SAVE_DIR + he_date_str
                            + '_pre_processed_map.npy')
        pre_processed_map_data = np.load(pre_process_file, allow_pickle=True)[-1]
        pre_processed_map = sunpy.map.Map(
            np.flipud(pre_processed_map_data), he_map.meta
        )
        
        # Threshold pre-processed map with varied lower bounds
        thresh_bound_list = [
            get_thresh_bound(pre_processed_map_data, percent_of_peak)
            for percent_of_peak in percent_of_peak_list
        ]
        thresh_maps = [
            np.where(pre_processed_map_data >= thresh_bound, 
                     pre_processed_map_data, 0)
            for thresh_bound in thresh_bound_list
        ]
        
        # Lists of outcomes of CH detected at given or greater
        # confidence levels
        num_ch_by_thresh_list.append(
            [get_num_ch(thresh_map) for thresh_map in thresh_maps]
        )
        area_tuple_by_thresh_list = [
            get_open_area(pre_processed_map, thresh_bound)
            for thresh_bound in thresh_bound_list
        ]
        area_percent_by_thresh_list.append(
            [area_tuple[0] for area_tuple in area_tuple_by_thresh_list]
        )
        area_by_thresh_list.append(
            [area_tuple[1] for area_tuple in area_tuple_by_thresh_list]
        )
        px_percent_by_thresh_list.append(
            get_px_percent_list(thresh_maps)
        )
    
    # Convert to dataframes
    datetime_list = [datetime.strptime(he_date_str, DICT_DATE_STR_FORMAT)
                     for he_date_str in he_date_str_list]
    num_ch_df = pd.DataFrame(
        num_ch_by_thresh_list, columns=percent_of_peak_list,
        index=datetime_list
    )
    area_percent_df = pd.DataFrame(
        area_percent_by_thresh_list, columns=percent_of_peak_list,
        index=datetime_list
    )
    area_df = pd.DataFrame(
        area_by_thresh_list, columns=percent_of_peak_list,
        index=datetime_list
    )
    px_percent_df = pd.DataFrame(
        px_percent_by_thresh_list, columns=percent_of_peak_list,
        index=datetime_list
    )
    return num_ch_df, area_percent_df, area_df, px_percent_df


def get_outcome_time_series_dict_v0_1(he_date_str_list, detection_save_dir):
    """Retrieve a dictionary of series with single mask outcomes over time.
    
    Args
        he_date_str_list: list of date strings for ensemble maps
        detection_save_dir: path to saved ensemble maps
    Returns
        Dictionary of Series of outcomes by confidence level over time.
    """
    # List for outcomes at varied confidence levels and datetimes
    num_ch_list = []
    area_percent_list = []
    area_list = []
    px_percent_list = []

    for he_date_str in he_date_str_list:
        
        he_fits_file = DATA_FITS_FORMAT.format(
            data_dir=ALL_HE_DIR, date_str=he_date_str
        )
        he_map = prepare_data.get_nso_sunpy_map(he_fits_file)
        if not he_map:
            print(f'{he_date_str} He I observation extraction failed.')
            continue
        
        # Extract saved single mask
        mask_file = f'{detection_save_dir}{he_date_str}_ensemble_map.npy'
        mask_data = np.load(mask_file, allow_pickle=True)[-1]
        mask_map = sunpy.map.Map(np.flipud(mask_data), he_map.meta)
        
        # Lists of CH outcomes
        num_ch_list.append(get_num_ch(mask_data))
        area_tuple = get_open_area(mask_map, confidence_level=0)
        area_percent_list.append(area_tuple[0])
        area_list.append(area_tuple[1])
        px_percent_list.append(get_px_percent_list([mask_data])[0])
    
    # Convert to Series
    datetime_list = [datetime.strptime(he_date_str, DICT_DATE_STR_FORMAT)
                    for he_date_str in he_date_str_list]
    outcome_time_series_dict = {
        'num_ch': pd.Series(num_ch_list, index=datetime_list),
        'area_percent': pd.Series(area_percent_list, index=datetime_list),
        'area': pd.Series(area_list, index=datetime_list),
        'px_percent': pd.Series(px_percent_list, index=datetime_list)
    }
    return outcome_time_series_dict


def get_outcome_time_series_dict(he_date_str_list, confidence_level_list,
                                 detection_save_dir):
    """Retrieve a dictionary of dataframes with ensemble map outcomes
    at specified confidence levels over time.
    
    Args
        he_date_str_list: list of date strings for ensemble maps
        confidence_level_list: list of float confidence levels at which
            to threshold ensemble maps for computing outcomes
        detection_save_dir: path to saved ensemble maps
    Returns
        Dictionary of dataframes of outcomes by confidence level over time.
    """
    # List for outcomes over time series
    # Will hold lists of outcomes by confidence level
    num_ch_time_series = []
    area_percent_time_series = []
    area_time_series = []
    px_percent_time_series = []

    for he_date_str in he_date_str_list:
        
        he_file = f'{ALL_HE_DIR}{he_date_str}.fts'
        he_map = prepare_data.get_nso_sunpy_map(he_file)
        if not he_map:
            print(f'{he_date_str} He I observation extraction failed.')
            continue
        
        # Extract saved ensemble map
        ensemble_file = f'{detection_save_dir}{he_date_str}_ensemble_map.npy'
        ensemble_map_data = np.load(ensemble_file, allow_pickle=True)[-1]
        ensemble_map = sunpy.map.Map(np.flipud(ensemble_map_data), he_map.meta)
        
        confidence_masks = [
            np.where(ensemble_map_data >= confidence_level, 1, 0)
            for confidence_level in confidence_level_list
        ]
        
        # Lists of outcomes of CH detected at given or greater
        # confidence levels
        num_ch_time_series.append(
            [get_num_ch(confidence_mask)
             for confidence_mask in confidence_masks]
        )
        area_tuple_by_confidence_list = [
            get_open_area(ensemble_map, confidence_level)
            for confidence_level in confidence_level_list
        ]
        area_percent_time_series.append(
            [area_tuple[0] for area_tuple in area_tuple_by_confidence_list]
        )
        area_time_series.append(
            [area_tuple[1] for area_tuple in area_tuple_by_confidence_list]
        )
        px_percent_time_series.append(
            get_px_percent_list(confidence_masks)
        )
    
    # Convert to dataframes
    datetime_list = [datetime.strptime(he_date_str, DICT_DATE_STR_FORMAT)
                    for he_date_str in he_date_str_list]
    
    outcome_time_series_dict = {
        'num_ch': pd.DataFrame(
            num_ch_time_series, columns=confidence_level_list,
            index=datetime_list
            ),
        'area_percent': pd.DataFrame(
            area_percent_time_series, columns=confidence_level_list,
            index=datetime_list
            ),
        'area': pd.DataFrame(
            area_time_series, columns=confidence_level_list,
            index=datetime_list
            ),
        'px_percent': pd.DataFrame(
            px_percent_time_series, columns=confidence_level_list,
            index=datetime_list
            ),
    }
    return outcome_time_series_dict


def get_outcome_time_series_dict_v0_5_1(he_date_str_list, confidence_level_list,
                                         detection_save_dir):
    """Retrieve a dictionary of dataframes with ensemble map outcomes
    at specified confidence levels over time.
    
    Args
        he_date_str_list: list of date strings for ensemble maps
        confidence_level_list: list of float confidence levels at which
            to threshold ensemble maps for computing outcomes
         detection_save_dir: path to saved ensemble maps
    Returns
        Dictionary of dataframes of outcomes by confidence level over time.
    """
    # List for outcomes over time series
    # Will hold lists of outcomes by confidence level
    num_ch_time_series = []
    area_percent_time_series = []
    area_time_series = []
    px_percent_time_series = []

    for he_date_str in he_date_str_list:
        
        # Extract saved ensemble map array and convert to Sunpy map
        ensemble_file = f'{detection_save_dir}{he_date_str}_ensemble_map.fits'
        ensemble_map = sunpy.map.Map(ensemble_file)
        ensemble_map_data = np.flipud(ensemble_map.data)
        
        confidence_masks = [
            np.where(ensemble_map_data >= confidence_level, 1, 0)
            for confidence_level in confidence_level_list
        ]
        
        # Lists of outcomes of CH detected at given or greater
        # confidence levels
        num_ch_time_series.append(
            [get_num_ch(confidence_mask)
             for confidence_mask in confidence_masks]
        )
        area_tuple_by_confidence_list = [
            get_open_area(ensemble_map, confidence_level)
            for confidence_level in confidence_level_list
        ]
        area_percent_time_series.append(
            [area_tuple[0] for area_tuple in area_tuple_by_confidence_list]
        )
        area_time_series.append(
            [area_tuple[1] for area_tuple in area_tuple_by_confidence_list]
        )
        px_percent_time_series.append(
            get_px_percent_list(confidence_masks)
        )
    
    # Convert to dataframes
    datetime_list = [datetime.strptime(he_date_str, DICT_DATE_STR_FORMAT)
                     for he_date_str in he_date_str_list]

    outcome_time_series_dict = {
        'num_ch': pd.DataFrame(
            num_ch_time_series, columns=confidence_level_list,
            index=datetime_list
            ),
        'area_percent': pd.DataFrame(
            area_percent_time_series, columns=confidence_level_list,
            index=datetime_list
            ),
        'area': pd.DataFrame(
            area_time_series, columns=confidence_level_list,
            index=datetime_list
            ),
        'px_percent': pd.DataFrame(
            px_percent_time_series, columns=confidence_level_list,
            index=datetime_list
            ),
    }
    return outcome_time_series_dict


def get_mad_by_confidences(outcome_df, confidence_level_list):
    """Retrieve an array of median absolute deviation in an outcome
    for varied confidence levels and a normalized metric by the 
    time series median.
    """
    mad_by_confidences = [
        np.median(np.abs(outcome_df[cl] - outcome_df[cl].median()))
        for cl in confidence_level_list
    ]
    norm_mad_by_confidences = [
        mad/outcome_df[cl].median()*100
        for mad, cl in zip(mad_by_confidences, confidence_level_list)
    ]
    return np.array(mad_by_confidences), np.array(norm_mad_by_confidences)


def write_video(output_dir, fps):
    """Write images in a directory to a video with
    the given frames per second.
    
    Args
        output_dir: path to output video to
        fps: int number of frames per second
    """
    image_files = [
        os.path.join(output_dir,img)
        for img in os.listdir(output_dir)
        if img.endswith('.jpg')
    ]
    image_files.sort()
    clip = ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(f'{output_dir}video_{fps}fps.mp4')
