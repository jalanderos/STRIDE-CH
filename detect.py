
"""Library of functions to detect coronal holes.
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
from sunpy.map.header_helper import make_heliographic_header

from settings import *
import prepare_data

# Module variables
DICT_DATE_STR_FORMAT = '%Y_%m_%d__%H_%M'
HE_OBS_DATE_STR_FORMAT = '%Y-%m-%dT%H:%M:%S'
SOLAR_AREA = 4*np.pi*(1*u.solRad).to(u.Mm)**2
MIN_SIZE = 5000


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


def pre_process_v0_5(he_map):
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
    """Reproject a Sunpy map to to a Cylindrical Equal Area projection.
    Equivalent to a Heliographic Stonyhurst frame, but with sine latitude,
    longitude coordinates.
    
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
    reproject_row_num = 2*Rs_dim
    # reproject_row_num = sunpy_map.data.shape[0]
    
    # Create Heliographic frame header
    hg_header = make_heliographic_header(
        sunpy_map.date, sunpy_map.observer_coordinate,
        shape=(reproject_row_num, 2*reproject_row_num), frame='stonyhurst'
    )
    
    # Convert keywords to CEA
    # Via https://github.com/dstansby/pfsspy/blob/main/pfsspy/utils.py
    hg_header['ctype1'] = 'HGLN-CEA'
    hg_header['ctype2'] = 'HGLT-CEA'
    hg_header['cdelt2'] = 180 / np.pi * 2 / reproject_row_num
    
    reprojected_map = sunpy_map.reproject_to(
        hg_header, algorithm='adaptive'
    )
    return reprojected_map
    


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


def get_ch_mask(array, percent_of_peak, morph_radius, min_size=MIN_SIZE):
    """Retrieve a single segmentation.
    
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


# Candidate Coronal Hole Functions
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


def get_ch_medians(map_data_by_ch):
    """Retrieve a list of median values for each detected CH.
    
    Args
        map_data_by_ch: list of isolated CH images from a segmentation
    """
    medians = [np.median(map_data[~np.isnan(map_data)])
               for map_data in map_data_by_ch]
    return medians


def get_ch_band_widths(map_data_by_ch):
    """Retrieve a list of 5th to 95th percentile band widths for each
    detected CH.
    
    Args
        map_data_by_ch: list of isolated CH images from a segmentation
    """
    percentiles = [5, 95]
    bound_list = [np.percentile(map_data[~np.isnan(map_data)], percentiles)
                  for map_data in map_data_by_ch]
    
    hole_band_widths = [bounds[1] - bounds[0]
                        for bounds in bound_list]
    return hole_band_widths


def get_ch_lower_tail_widths(map_data_by_ch):
    """Retrieve a list of lower tail widths for each detected CH.
    
    Args
        map_data_by_ch: list of isolated CH images from a segmentation
    """
    filt_map_data_by_ch = [map_data[~np.isnan(map_data)]
                         for map_data in map_data_by_ch]
    
    # List of the 1st percentile brightness value of each CH
    first_percentile_list = [np.percentile(map_data, 1)
                             for map_data in filt_map_data_by_ch]
        
    # List of peak count of each CH
    peak_counts_value_list = [
        get_peak_counts_loc(map_data, bins_as_percent=False)
        for map_data in filt_map_data_by_ch
    ]

    # List of lower tail widths of each CH
    ch_lower_tail_width_list = [
        peak_count - first_percentile
        for peak_count, first_percentile 
        in zip(peak_counts_value_list, first_percentile_list)]
    
    return ch_lower_tail_width_list


# v0.1 Segmentation Outcome Functions
def get_px_percent(ch_mask):
    """Retrieve the percentage of pixels detected in a segmentation.
    
    Args
        ch_mask: binary coronal holes mask
    Returns
        Pixel percentage detected a segmentation.
    """
    disk_px_count = np.count_nonzero(~np.isnan(ch_mask))
    return np.count_nonzero(ch_mask)*100/disk_px_count


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


def get_thresh_px_percent_list(array, percent_of_peak_list):
    """Retrieve the area percentage of pixels accepted by varied thresholds.
    """
    thresh_bound_list = [
        get_thresh_bound(array, percent_of_peak)
        for percent_of_peak in percent_of_peak_list
    ]
    px_percent_list = [
        np.count_nonzero(array > thresh_bound)*100/array.size
        for thresh_bound in thresh_bound_list
    ]
    return px_percent_list
    

def get_parameter_stats(outcome_list):
    """Retrieve maximum difference between segmentations in area percentage
    detected, the average area percentage at the max difference for a cutoff,
    the number selected below this cutoff, and differences in area percentage.
    """    
    outcome_diffs = np.abs(np.diff(outcome_list))

    max_diff_i = np.argmax(outcome_diffs)
    max_diff = np.max(outcome_diffs)*100/outcome_list[max_diff_i]
    
    cutoff = np.mean([outcome_list[max_diff_i], 
                      outcome_list[max_diff_i + 1]])

    selected_parameter_num = np.count_nonzero(outcome_list > cutoff)
    
    return max_diff, cutoff, selected_parameter_num, outcome_diffs


def get_ch_gradient_median_list(array, ch_mask_list): 
    """Retrieve the median of the gradient across CHs
    for each segmentation in a list. UNTESTED
    
    Args
        array: image to process
        ch_mask_list: binary coronal holes mask list
    Returns
        List of gradient median of CHs detected in segmentations.
    """
    # List of masked array of candidate CHs
    masked_candidates_list = [
        get_masked_candidates(array, ch_mask)
        for ch_mask in ch_mask_list
    ]
    # Compute spatial gradient
    gradient_candidates_list = [
        filters.sobel(masked_candidates)
        for masked_candidates in masked_candidates_list
    ]
    
    # Sort candidate CHs by gradient median
    gradient_median_list = get_ch_medians(gradient_candidates_list)
        
    return gradient_median_list


def get_ch_lower_tail_width_list(array, ch_mask_list): 
    """Retrieve the average of the histogram lower tail width across CHs
    for each segmentation in a list.
    
    Args
        array: image to process
        ch_mask_list: binary coronal holes mask list
    Returns
        List of mean histogram tail widths of CHs detected in segmentations.
    """
    labeled_ch_list = [ndimage.label(ch_mask)[0]
                          for ch_mask in ch_mask_list]
    num_ch_list = [get_num_ch(ch_mask) for ch_mask in ch_mask_list]
    
    # List of average lower tail widths across all CH's of each segmentaion
    lower_tail_width_list = []
    
    count = 0
    for labeled_ch_mask, num_ch in zip(labeled_ch_list, num_ch_list):

        map_data_by_ch = get_map_data_by_ch(array, labeled_ch_mask, num_ch)
    
        ch_mask_lower_tail_width_list = get_ch_lower_tail_widths(map_data_by_ch)
        
        mean_lower_tail_width = np.mean(ch_mask_lower_tail_width_list)
        
        lower_tail_width_list.append(mean_lower_tail_width)
        count = count + 1
        
    return lower_tail_width_list


# Modern Segmentation Outcome Functions
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


def get_A_per_square_px(ensemble_map):
    """Retrieve area covered by a normal to line of sight square pixel
    in Mm^2. Helioprojective angular change per pixel is converted to
    distance change per pixel in a Heliocentric frame.
    
    Args
        ensemble_map: Sunpy map object of ensemble detection map
    Returns
        Astropy quantity of square area per pixel.
    """
    hp_delta_coords = frames.Helioprojective(
        ensemble_map.scale.axis1*u.pix,
        ensemble_map.scale.axis2*u.pix,
        observer='earth', obstime=ensemble_map.date
    )
    hc_delta_coords = hp_delta_coords.transform_to(
        frames.Heliocentric(observer='earth', obstime=ensemble_map.date)
    )

    A_per_square_px = np.abs(
        hc_delta_coords.x.to(u.Mm)*hc_delta_coords.y.to(u.Mm)
    )
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
    detected_locs = np.where(np.flipud(ensemble_map.data) >= confidence_level)

    # Convert detected pixels to Helioprojective Sky Coords
    detected_hp_coords = ensemble_map.pixel_to_world(
        detected_locs[1]*u.pix, detected_locs[0]*u.pix
    )
    
    # Convert detected Helioprojective Sky Coords to Heliographic lon, lat
    raw_detected_hg_coords = detected_hp_coords.transform_to(
        frames.HeliographicStonyhurst(obstime=ensemble_map.date)
    )
    
    # Remove pixels with failed coordinate conversion and longitudes
    # outside (-90,90)
    failed_coord_idxs = np.where(
        np.isnan(raw_detected_hg_coords.lon) 
        | (np.abs(raw_detected_hg_coords.lon.to(u.deg).value) >= 90)
    )
    detected_hg_coords = np.delete(raw_detected_hg_coords, failed_coord_idxs)
    
    return detected_hg_coords, failed_coord_idxs


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
    # B-angle to subtract from latitude
    B0 = ensemble_map.center.observer.lat
    
    pixel_lons = detected_hg_coords.lon.to(u.rad).value
    pixel_lats = detected_hg_coords.lat.to(u.rad).value - B0.to(u.rad).value
    pixel_areas = A_per_square_px/(np.cos(pixel_lons)*np.cos(pixel_lats))
    
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
    detected_idxs = np.where(np.flipud(ensemble_map.data) >= confidence_level)
    pixel_B_LOS = reprojected_mag_map.data[detected_idxs]*u.G
    
    # Remove pixels with failed coordinate conversion
    pixel_B_LOS = np.delete(pixel_B_LOS, failed_coord_idxs)
    
    # Remove pixels with failed magnetic data retrieval
    failed_mag_idxs = np.where(np.isnan(pixel_B_LOS))
    pixel_B_LOS = np.delete(pixel_B_LOS, failed_mag_idxs)
    
    return pixel_B_LOS, failed_mag_idxs


def get_outcomes(ensemble_map, reprojected_mag_map, A_per_square_px,
                 confidence_level=0):
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
    Returns
        Open area, center of mass latitude and longitude, unsigned, signed
            open flux, flux skewnwss, unipolarity, and flux per pixel.
    """
    detected_hg_coords, failed_coord_idxs = get_detected_hg_coords(
        ensemble_map, confidence_level
    )
    pixel_areas = get_pixel_areas(
        ensemble_map, A_per_square_px, detected_hg_coords
    )
    
    # Sum area detected in all pixels
    open_area = np.sum(pixel_areas).value
    
    pixel_B_LOS, failed_mag_idxs = get_pixel_B_LOS(
        ensemble_map, reprojected_mag_map, confidence_level,
        failed_coord_idxs
    )
    pixel_areas = np.delete(pixel_areas, failed_mag_idxs)
    
    # B-angle to subtract from latitude
    B0 = ensemble_map.center.observer.lat
    
    # Center of mass
    cm_lat = (np.sum(detected_hg_coords.lat*pixel_areas)
              /np.sum(pixel_areas)).value
    cm_lon = (np.sum(detected_hg_coords.lon*pixel_areas)
              /np.sum(pixel_areas)).value
    
    # Global unsigned quantities
    pixel_unsigned_fluxes = (np.abs(pixel_B_LOS)*pixel_areas).to(u.Wb)
    unsigned_open_flux = np.sum(pixel_unsigned_fluxes).value
    
    # Per CH signed quantities
    pixel_signed_fluxes = (pixel_B_LOS*pixel_areas).to(u.Mx).value
    signed_open_flux = np.sum(pixel_signed_fluxes)
    mag_skew = stats.skew((pixel_signed_fluxes))
    
    # Calculate unipolarity
    
    # Foreshortening factors per detected pixel
    pixel_lons = detected_hg_coords.lon.to(u.rad).value
    pixel_lats = detected_hg_coords.lat.to(u.rad).value - B0.to(u.rad).value
    foreshort_factors = np.cos(pixel_lons)*np.cos(pixel_lats)
    foreshort_factors = np.delete(foreshort_factors, failed_mag_idxs)
    
    pixel_B_r = pixel_B_LOS/foreshort_factors
    
    signed_B_r = np.abs(np.nanmean(pixel_B_r))
    unsigned_B_r = np.nanmean(np.abs(pixel_B_r))
    unipolarity = (signed_B_r/unsigned_B_r).value
    
    return open_area, cm_lat, cm_lon, unsigned_open_flux, signed_open_flux, \
        mag_skew, unipolarity, pixel_signed_fluxes


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
        he_map = prepare_data.get_solis_sunpy_map(he_file)
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


def get_outcome_time_series_dfs(he_date_str_list, confidence_level_list,
                                ensemble_maps_save_dir):
    """Retrieve dataframes with ensemble map outcomes at specified confidence
    levels over time.
    
    Args
        he_date_str_list: list of date strings for ensemble maps
        confidence_level_list: list of float confidence levels at which
            to threshold ensemble maps for computing outcomes
        ensemble_maps_save_dir: path to saved ensemble map arrays
    Returns
        Dataframes of outcomes by confidence level over time.
    """
    # List for outcomes over time series
    # Will hold lists of outcomes by confidence level
    num_ch_time_series = []
    area_percent_time_series = []
    area_time_series = []
    px_percent_time_series = []

    for he_date_str in he_date_str_list:
        
        he_file = f'{ALL_HE_DIR}{he_date_str}.fts'
        he_map = prepare_data.get_solis_sunpy_map(he_file)
        if not he_map:
            print(f'{he_date_str} He I observation extraction failed.')
            continue
        
        # Extract saved ensemble map
        ensemble_file = f'{ensemble_maps_save_dir}{he_date_str}_ensemble_map.npy'
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
    num_ch_df = pd.DataFrame(
        num_ch_time_series, columns=confidence_level_list,
        index=datetime_list
    )
    area_percent_df = pd.DataFrame(
        area_percent_time_series, columns=confidence_level_list,
        index=datetime_list
    )
    area_df = pd.DataFrame(
        area_time_series, columns=confidence_level_list,
        index=datetime_list
    )
    px_percent_df = pd.DataFrame(
        px_percent_time_series, columns=confidence_level_list,
        index=datetime_list
    )
    return num_ch_df, area_percent_df, area_df, px_percent_df


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


# Ensemble Map Functions
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
    """
    ch_mask_list = [
        get_ch_mask(array, percent_of_peak, morph_radius)
        for percent_of_peak, morph_radius
        in zip(percent_of_peak_list, morph_radius_list)
    ]
    ch_masks = np.array(ch_mask_list)
    
    num_masks = len(ch_mask_list)
    confidence_list = [(c + 1)/num_masks *100
                       for c in range(num_masks)]

    # Sort masks by pixel percentage detected
    px_percent_list = get_px_percent_list(ch_mask_list)
    sorted_mask_idxs = np.flip(np.argsort(px_percent_list))
    
    ch_masks = ch_masks[sorted_mask_idxs]

    ensemble_map = np.where(~np.isnan(array), 0, np.nan)
    
    for ch_mask, confidence in zip(ch_masks, confidence_list):
        ensemble_map = np.where(
            ch_mask == 1, confidence, ensemble_map
        )
    return ensemble_map, ch_masks, confidence_list


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
    gradient_medians = get_ch_medians(grad_data_by_ch)
    sorted_idxs = np.flip(np.argsort(gradient_medians))
    
    # Sort candidate CHs from greatest to least gradient median
    map_data_by_ch = [map_data_by_ch[i] for i in sorted_idxs]
    
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
    return ensemble_map, map_data_by_ch, confidence_list


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
    """
    pre_processed_map_data = np.flipud(pre_processed_map.data)
    
    # Create global segmentations for varied design variable combinations
    ch_masks = [
        get_ch_mask(pre_processed_map_data, percent_of_peak, morph_radius)
        for percent_of_peak, morph_radius
        in zip(percent_of_peak_list, morph_radius_list)
    ]
    
    # List to hold pre processed map data for distinct CHs
    # from all segmentations
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
    cm_lat = []
    
    for ch_label in range(num_ch):
        distinct_ch_mask = masks_by_ch[ch_label]
        
        # Not flipping works right
        distinct_ch_map = sunpy.map.Map(
            distinct_ch_mask, pre_processed_map.meta
        )
        outcomes = get_outcomes(
            distinct_ch_map, reprojected_mag_map, A_per_square_px
        )
        cm_lat.append(outcomes[1])
        unipolarity_by_ch.append(outcomes[6])
    
    # Sort unipolarities from greatest to least
    sorted_idxs = np.argsort(unipolarity_by_ch)
    
    # Sort candidate CHs from greatest to least gradient median
    masks_by_ch = [masks_by_ch[i] for i in sorted_idxs]
    cm_lat = [cm_lat[i] for i in sorted_idxs]
    unipolarity_by_ch = [unipolarity_by_ch[i] for i in sorted_idxs]
    
    confidence_levels = np.array(
        [unipolarity*100 for unipolarity in unipolarity_by_ch]
    )
    confidence_levels = np.where(
        confidence_levels >= unipolarity_threshold*100, confidence_levels, 0
    )

    # Construct ensemble map by adding distinct CHs with assigned
    # confidence level values to an empty base disk    
    ensemble_map_data = np.where(~np.isnan(pre_processed_map_data),
                                 0, np.nan)
    for distinct_ch, confidence in zip(masks_by_ch, confidence_levels):
        ensemble_map_data = np.where(
            ~np.isnan(distinct_ch), confidence, ensemble_map_data
        )
    return ensemble_map_data, masks_by_ch, confidence_levels


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
