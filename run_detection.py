"""
"""
import os
import numpy as np
from scipy import ndimage
from skimage import morphology, filters
from moviepy.video.io import ImageSequenceClip


# Module variables
DICT_DATE_STR_FORMAT = '%Y_%m_%d__%H_%M'
HE_OBS_DATE_STR_FORMAT = '%Y-%m-%dT%H:%M:%S'
MIN_SIZE = 5000
EMPTY_DISK_VAL = 0


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
def pre_process_eqw_v0_1(raw_eqw, peak_count_cutoff_percent=0.1):
    """Pre-process equivalent width array by setting background to NaN
    and cutting off at some percentage of histogram peak.
    
    Args
        peak_count_cutoff_percent: Vertical cutoff for histogram
            counts below a percentage
    """
    eqw_nan = np.where(raw_eqw == 0, np.NaN, raw_eqw)

    hist, edges = get_hist(eqw_nan)
    max_count = np.max(hist)

    cutoff = max_count*peak_count_cutoff_percent/100
    
    array_range = np.nanmax(eqw_nan) - np.nanmin(eqw_nan)
    val = edges[:-1]/100 * array_range + np.nanmin(eqw_nan)
    
    cutoff_edges = np.where(hist > cutoff, val, 0)

    eqw_high_cut = np.where(eqw_nan > np.max(cutoff_edges), np.NaN, eqw_nan)

    eqw_band_cut = np.clip(
        eqw_high_cut, np.min(cutoff_edges), np.max(cutoff_edges)
    )
    
    return eqw_band_cut, eqw_high_cut, eqw_nan


def pre_process_eqw_v0_3(raw_eqw):
    """Pre-process equivalent width array by setting background to NaN
    and a simple brightness band pass. WIP.
    """
    eqw_nan = np.where(raw_eqw == 0, np.NaN, raw_eqw)
    
    eqw_high_cut = np.where(eqw_nan > 100, np.NaN, eqw_nan)
    eqw_band_cut = np.clip(eqw_high_cut, -100, 100)
    
    return eqw_band_cut, eqw_nan


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
    """Perform morphological operations.
    
    Args
        array: Numpy array
        morph_radius: int pixel number for radius of disk structuring element
            in morphological operations
    Returns
        Array that has been processed via morphological opening and closing.
    """
    open_array = morphology.binary_opening(
        array, morphology.disk(morph_radius)
    )
    close_array = morphology.binary_closing(
        open_array, morphology.disk(morph_radius)
    )
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


def get_isolated_ch_im_list(array, ch_mask):
    """Retrieve a list of images with isolated individual CHs.
    
    Args
        array: image to process
        ch_mask: binary coronal holes mask
    Returns
        List of isolated CH images from a segmentation.
    """
    # Array with number labels for separate detected CHs and number of CHs
    labeled_candidates, num_ch = ndimage.label(ch_mask)
    
    isolated_ch_im_list = [
        np.where(labeled_candidates == ch_num + 1, array, np.NaN)
        for ch_num in range(num_ch)
    ]
    return isolated_ch_im_list


def get_ch_medians(isolated_ch_im_list):
    """Retrieve a list of median values for each detected CH.
    
    Args
        isolated_ch_im_list: list of isolated CH images from a segmentation
    """
    medians = [np.median(ch_im[~np.isnan(ch_im)])
               for ch_im in isolated_ch_im_list]
    return medians


def get_ch_band_widths(isolated_ch_im_list):
    """Retrieve a list of 5th to 95th percentile band widths for each
    detected CH.
    
    Args
        isolated_ch_im_list: list of isolated CH images from a segmentation
    """
    percentiles = [5, 95]
    bound_list = [np.percentile(ch_im[~np.isnan(ch_im)], percentiles)
                  for ch_im in isolated_ch_im_list]
    
    hole_band_widths = [bounds[1] - bounds[0]
                        for bounds in bound_list]
    return hole_band_widths


def get_ch_lower_tail_widths(isolated_ch_im_list):
    """Retrieve a list of lower tail widths for each detected CH.
    
    Args
        isolated_ch_im_list: list of isolated CH images from a segmentation
    """
    filt_ch_im_list = [ch_im[~np.isnan(ch_im)]
                         for ch_im in isolated_ch_im_list]
    
    # List of the 1st percentile brightness value of each CH
    first_percentile_list = [np.percentile(ch_im, 1)
                             for ch_im in filt_ch_im_list]
        
    # List of peak count of each CH
    peak_counts_value_list = [
        get_peak_counts_loc(ch_im, bins_as_percent=False)
        for ch_im in filt_ch_im_list
    ]

    # List of lower tail widths of each CH
    ch_lower_tail_width_list = [
        peak_count - first_percentile
        for peak_count, first_percentile 
        in zip(peak_counts_value_list, first_percentile_list)]
    
    return ch_lower_tail_width_list


def get_ranked_map(array, ch_mask, apply_gradient, hist_stat,
                   ascend_sort=True):
    """Retrieve a map of CHs from a single segmentation as ranked by a
    histogram statistic.
    
    Args
        array: image to process
        ch_mask: binary coronal holes mask
        apply_gradient: boolean to specify taking spatial gradient of image
        hist_stat: str to specify histogram sorting statistic
            'median', 'width', 'tail_width'
        ascend_sort: boolean to specify sorting CHs from least to greatest
            statistic
    Returns
        List of isolated CH images from a segmentation.
    """
    # Masked array of candidate CHs
    masked_candidates = get_masked_candidates(array, ch_mask)
    if apply_gradient:
        masked_candidates = filters.sobel(masked_candidates)
    
    # Isolated images of detected CHs
    isolated_ch_im_list = get_isolated_ch_im_list(
        masked_candidates, ch_mask
    )
    num_ch = len(isolated_ch_im_list)
    
    # Rank candidates by histogram statistic
    if hist_stat == 'median':
        medians = get_ch_medians(isolated_ch_im_list)
        sorted_candidate_idxs = np.argsort(medians)
    elif hist_stat == 'width':
        ch_band_widths = get_ch_band_widths(isolated_ch_im_list)
        sorted_candidate_idxs = np.argsort(ch_band_widths)
    elif hist_stat == 'tail_width':
        ch_lower_tail_width_list = get_ch_lower_tail_widths(
            isolated_ch_im_list
        )
        sorted_candidate_idxs = np.argsort(ch_lower_tail_width_list)
    
    if ascend_sort:
        sorted_candidate_idxs = np.flip(sorted_candidate_idxs)
    
    isolated_ch_ims = np.array(isolated_ch_im_list)
    ranked_ch_ims = isolated_ch_ims[sorted_candidate_idxs]
    
    ranked_map = np.where(~np.isnan(array), EMPTY_DISK_VAL, np.nan)
    
    for isolated_ch_im, ch_num in zip(ranked_ch_ims, range(num_ch)):
        ranked_map = np.where(
            ~np.isnan(isolated_ch_im), (ch_num + 1)*100/num_ch, ranked_map
        )
    return ranked_map


# Segmentation Summary Outcome Functions
def get_thresh_area_percent_list(array, percent_of_peak_list):
    """Retrieve the area percentage of pixels accepted by varied thresholds.
    """
    thresh_bound_list = [
        get_thresh_bound(array, percent_of_peak)
        for percent_of_peak in percent_of_peak_list
    ]
    area_percent_list = [
        np.count_nonzero(array > thresh_bound)*100/array.size
        for thresh_bound in thresh_bound_list
    ]
    return area_percent_list
    

def get_parameter_stats(area_percent_list):
    """Retrieve maximum difference between segmentations in area percentage
    detected,the average area percentage at the max difference for a cutoff,
    the number selected below this cutoff, and differences in area percentage.
    """    
    pixel_percent_diffs = np.abs(np.diff(area_percent_list))

    max_diff_i = np.argmax(pixel_percent_diffs)
    max_diff = np.max(pixel_percent_diffs)*100/area_percent_list[max_diff_i]
    
    cutoff = np.mean([area_percent_list[max_diff_i], 
                      area_percent_list[max_diff_i + 1]])

    selected_parameter_num = np.count_nonzero(area_percent_list > cutoff)
    
    return max_diff, cutoff, selected_parameter_num, pixel_percent_diffs


def get_area_percent_list(ch_mask_list):
    """Retrieve the percentage of pixels detected in each segmentation
    in a list.
    
    Args
        array: image to process
        ch_mask_list: binary coronal holes mask list
    Returns
        List of pixel percentage detected in segmentations.
    """
    return [np.count_nonzero(ch_mask)*100/ch_mask.size
            for ch_mask in ch_mask_list]


def get_num_ch_list(ch_mask_list):
    """Retrieve the number of CHs detected in each segmentation in a list.
    
    Args
        array: image to process
        ch_mask_list: binary coronal holes mask list
    Returns
        List of number of CHs detected in segmentations.
    """    
    return [ndimage.label(ch_mask)[1]
            for ch_mask in ch_mask_list]


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
    num_ch_list = get_num_ch_list(ch_mask_list)
    
    # List of average lower tail widths across all CH's of each segmentaion
    lower_tail_width_list = []
    
    count = 0
    for labeled_ch_mask, num_ch in zip(labeled_ch_list, num_ch_list):

        isolated_ch_im_list = get_isolated_ch_im_list(array, labeled_ch_mask, num_ch)
    
        ch_mask_lower_tail_width_list = get_ch_lower_tail_widths(isolated_ch_im_list)
        
        mean_lower_tail_width = np.mean(ch_mask_lower_tail_width_list)
        
        lower_tail_width_list.append(mean_lower_tail_width)
        count = count + 1
        
    return lower_tail_width_list


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

    # Sort masks by area percentage detected
    area_percent_list = get_area_percent_list(ch_mask_list)
    sorted_mask_idxs = np.flip(np.argsort(area_percent_list))
    
    ch_masks = ch_masks[sorted_mask_idxs]

    ensemble_map = np.where(~np.isnan(array), EMPTY_DISK_VAL, np.nan)
    
    for ch_mask, confidence in zip(ch_masks, confidence_list):
        ensemble_map = np.where(
            ch_mask == 1, confidence, ensemble_map
        )
    return ensemble_map, ch_masks, confidence_list


def get_ensemble_v0_3(array, percent_of_peak_list, morph_radius_list):
    """Retrieve an ensemble of segmentations sorted by CH smoothness.
    
    Args
        array: image to process
        percent_of_peak_list: list of float percentage values
            at which to take threshold
        morph_radius_list: list of int pixel number for radius of disk 
            structuring element in morphological operations
        percent_rank: boolean to specify ranking
            True to assign confidence by percentage of gradient
                median among values from other candidate CHs in [0,100]%
            False to assign confidence by direct ranking in (0,100]%
    Returns
        Ensemble greyscale coronal holes mask sorted by mean gradient.
        List of binary coronal holes masks.
        List of confidence levels in mask layers.
    """
    # Create global segmentations
    ch_mask_list = [
        get_ch_mask(array, percent_of_peak, morph_radius)
        for percent_of_peak, morph_radius
        in zip(percent_of_peak_list, morph_radius_list)
    ]
    
    # Isolate individual detected CHs from all segmenations
    isolated_ch_im_list = []
    isolated_gradient_im_list = []
    
    for ch_mask in ch_mask_list:
        # Masked array of candidate CHs
        masked_candidates = get_masked_candidates(array, ch_mask)
        
        # Compute spatial gradient
        gradient_candidates = filters.sobel(masked_candidates)
        
        # Isolated images of detected CHs and their gradient arrays
        isolated_ch_im_list.extend(
            get_isolated_ch_im_list(masked_candidates, ch_mask)
        )
        isolated_gradient_im_list.extend(
            get_isolated_ch_im_list(gradient_candidates, ch_mask)
        )
    
    # Sort gradient medians from greatest to least
    gradient_medians = get_ch_medians(isolated_gradient_im_list)
    sorted_candidate_idxs = np.flip(np.argsort(gradient_medians))
    gradient_medians.sort(reverse=True)
    
    # Sort candidate CHs from greatest to least gradient median
    isolated_ch_ims = np.array(isolated_ch_im_list)
    isolated_ch_ims = isolated_ch_ims[sorted_candidate_idxs]
    
    # Assign confidence by percentile or direct ranking
    num_ch = len(isolated_ch_im_list)
    confidence_list = [(c + 1)*100/num_ch
                       for c in range(num_ch)]

    ensemble_map = np.where(~np.isnan(array), EMPTY_DISK_VAL, np.nan)
    
    for isolated_ch_im, confidence in zip(isolated_ch_ims, confidence_list):
        ensemble_map = np.where(
            ~np.isnan(isolated_ch_im), confidence, ensemble_map
        )
    return ensemble_map, isolated_ch_ims, confidence_list


def get_smooth_ensemble(array, percent_of_peak_list, morph_radius_list,
                        percent_rank=False):
    """Retrieve an ensemble of segmentations sorted by CH smoothness.
    
    Args
        array: image to process
        percent_of_peak_list: list of float percentage values
            at which to take threshold
        morph_radius_list: list of int pixel number for radius of disk 
            structuring element in morphological operations
        percent_rank: boolean to specify ranking
            True to assign confidence by percentage of gradient
                median among values from other candidate CHs in [0,100]%
            False to assign confidence by direct ranking in (0,100]%
    Returns
        Ensemble greyscale coronal holes mask sorted by mean gradient.
        List of binary coronal holes masks.
        List of confidence levels in mask layers.
    """
    # Create global segmentations
    ch_mask_list = [
        get_ch_mask(array, percent_of_peak, morph_radius)
        for percent_of_peak, morph_radius
        in zip(percent_of_peak_list, morph_radius_list)
    ]
    
    # Isolate individual detected CHs from all segmenations
    isolated_ch_im_list = []
    isolated_gradient_im_list = []
    
    for ch_mask in ch_mask_list:
        # Masked array of candidate CHs
        masked_candidates = get_masked_candidates(array, ch_mask)
        
        # Compute spatial gradient
        gradient_candidates = filters.sobel(masked_candidates)
        
        # Isolated images of detected CHs and their gradient arrays
        isolated_ch_im_list.extend(
            get_isolated_ch_im_list(masked_candidates, ch_mask)
        )
        isolated_gradient_im_list.extend(
            get_isolated_ch_im_list(gradient_candidates, ch_mask)
        )
    
    # Sort gradient medians from greatest to least
    gradient_medians = get_ch_medians(isolated_gradient_im_list)
    sorted_candidate_idxs = np.flip(np.argsort(gradient_medians))
    gradient_medians.sort(reverse=True)
    
    # Sort candidate CHs from greatest to least gradient median
    isolated_ch_ims = np.array(isolated_ch_im_list)
    isolated_ch_ims = isolated_ch_ims[sorted_candidate_idxs]
    
    # Assign confidence by percentile or direct ranking
    num_ch = len(isolated_ch_im_list)
    if percent_rank:
        percent_conversion = 100 \
            / (np.max(gradient_medians) - np.min(gradient_medians))
        confidence_list = [
            100 - (median - np.min(gradient_medians)) *percent_conversion
            for median in gradient_medians
        ]
    else:
        confidence_list = [(c + 1)*100/num_ch
                           for c in range(num_ch)]

    ensemble_map = np.where(~np.isnan(array), EMPTY_DISK_VAL, np.nan)
    
    for isolated_ch_im, confidence in zip(isolated_ch_ims, confidence_list):
        ensemble_map = np.where(
            ~np.isnan(isolated_ch_im), confidence, ensemble_map
        )
    return ensemble_map, isolated_ch_ims, confidence_list


def get_saved_ensemble_map(save_file):
    """Load saved ensemble map lists from file.
    """
    save_list = np.load(save_file, allow_pickle=True)
    eqw_date = save_list[0]
    percent_of_peak_list = save_list[1]
    morph_radius_list = save_list[2]
    ensemble_map = save_list[3]
    return eqw_date, percent_of_peak_list, morph_radius_list, ensemble_map


def write_ensemble_video(output_dir, fps):
    """Write images in a directory to a video with the given frames per second.
    """
    image_files = [
        os.path.join(output_dir,img)
        for img in os.listdir(output_dir)
        if img.endswith('.jpg')
    ]
    image_files.sort()
    clip = ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(f'{output_dir}ensemble_vid_{fps}fps.mp4')
