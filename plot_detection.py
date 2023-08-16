"""
"""
import sunpy.map
import numpy as np
from scipy import ndimage
import plotly.express as px
from skimage import filters
from astropy.io import fits
from datetime import datetime
from matplotlib import colormaps
from matplotlib import colormaps, pyplot as plt
from sunpy.coordinates.sun import carrington_rotation_number

from settings import *
import prepare_data
import detect


# Module variables
DICT_DATE_STR_FORMAT = '%Y_%m_%d__%H_%M'
EMPTY_ARRAY = np.zeros((2,2))


# General Visualization
def plot_hists(image_list, title_list, semilogy=True):
    """Plot two images and their histograms. Optionally display a semilog
    y axis.
    
    Args
        image_list: list of two numpy arrays
        title_list: list of title strings
        semilogy: boolean to specify semilog y axis
    """
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(40, 10))
    ax = axes.ravel()
    
    hist, edges = detect.get_hist(image_list[0], bins_as_percent=False, n=200)

    ax[0].set_title(title_list[0], fontsize=24)
    ax[0].imshow(image_list[0], cmap=plt.cm.gray)
    if semilogy:
        ax[1].semilogy(edges[0:-1], hist)
    else:
        ax[1].plot(edges[0:-1], hist)
        
    ax[2].set_title(title_list[1], fontsize=24)
    ax[2].imshow(image_list[1], cmap=plt.cm.gray)
    hist, edges = detect.get_hist(image_list[1], bins_as_percent=False, n=200)
    if semilogy:
        ax[3].semilogy(edges[0:-1], hist)
    else:
        ax[3].plot(edges[0:-1], hist)
        
        
def plot_images(image_list, title_list=[''], cmaps=[], image_size=5.):
    """Plot a list of images in a row with modifiable color maps.
    
    Args
        image_list: list of numpy arrays
        title_list: list of title strings
        cmaps: list of color map strings
        image_size: float width of subplot for each image
    """
    num_ims = len(image_list)
    title_list.extend(['' for _ in range(num_ims - len(title_list))])

    fig = plt.figure(figsize=(image_size*num_ims, image_size))

    for i in range(0,num_ims):
        
        ax = fig.add_subplot(1, num_ims, i + 1)
        if cmaps:
            ax.imshow(image_list[i], cmap=cmaps[i])
        else:
            ax.imshow(image_list[i], cmap=plt.cm.gray)

        ax.set_title(title_list[i], fontsize=24)


def plot_image_grid(image_list, num_cols, cmap='gray', image_size=5.):
    """Plot a list of images in a grid with the specified number of columns.
    Axes dictionary is retrieved to post-process individual subplots.
    
    Args
        image_list: list of numpy arrays
        num_cols: int number of columns in grid plot
        image_size: float width of subplot for each image
    Returns
        Dictionary keyed by indices of pyplot subplots.
    """
    axes = {}
    num_rows = int(np.ceil(len(image_list)/num_cols))
    
    fig = plt.figure(figsize=(image_size*num_cols, image_size*num_rows))
    
    for i, image in enumerate(image_list):
        axes[i] = fig.add_subplot(num_rows, num_cols, i + 1)
        axes[i].imshow(image, cmap)
        
    return axes


def plot_raw_fits_content(fits_path, header_list,
                          cmaps=[], print_header=False):
    """
    """
    raw_fits = fits.open(fits_path)
        
    if print_header:
        print(repr(raw_fits[0].header))
    
    num_data_arrays = raw_fits[0].header.get('NAXIS3')
    
    title_list = [raw_fits[0].header[header_key]
                  for header_key in header_list]
        
    if not num_data_arrays:
        im_list = [np.flipud(raw_fits[0].data)]
        
        if not title_list:
            title_list = ['']
    else:
        im_list = [np.flipud(raw_fits[0].data[i])
                   for i in range(num_data_arrays)]
        
        missing_title_num = num_data_arrays - len(title_list)
        title_list.extend(['' for _ in range(missing_title_num)])

        missing_cmap_num = num_data_arrays - len(cmaps)
        cmaps.extend([plt.cm.viridis for _ in range(missing_cmap_num)])
                
    plot_images(im_list, title_list, cmaps=cmaps)
    
    return im_list


# Magnetogram Plotting
def plot_map_contours(ax, smooth_map):
    """Add contours of large scale neutral regions from a smoothed
    Sunpy map to axes 
    
    Args
        ax: 
        magnetogram: 
        smooth_size_percent: float to specify uniform smoothing kernel size
            as a percentage of image size
    """   
    # Plot large scale neutral lines
    contours = smooth_map.contour(0)
    for contour in contours:
        ax.plot_coord(contour, color='y')


# Segmentation Parameter Varying Plots
def plot_thresholds(array, bounds, bounds_as_percent, threshold_type='lower'):
    """Plot image array, its histogram, and 3 thresholded images
    with their histograms.
    
    Args
        array: Numpy array
        bounds: list of threshold bounds from 0-100 %
            List of lists of lower and upper threshold bounds from 0-100 %
            for 'band' threshold type
        threshold_type:
            'lower' to disregard values below bounds 
            'band' to disregard values outside bounds
        bounds_as_percent: boolean to specify bounds as percentage
            of histogram peak or as direct threshold values
    """
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(40, 15))
    ax = axes.ravel()

    ax[0].imshow(array, cmap=plt.cm.gray)
    hist, edges = detect.get_hist(array, bins_as_percent=False)
    ax[1].set_title('Histogram', fontsize=24)
    ax[1].plot(edges[0:-1], hist)
    
    percent_str = '%' if bounds_as_percent else ''

    for i in range(3):
        img_i = (i + 1)*2
        hist_i = (i + 1)*2 + 1

        ax[hist_i].semilogy(edges[0:-1], hist)

        if threshold_type == 'band':
            if bounds_as_percent:
                lower_percent = bounds[0][i]
                upper_percent = bounds[1][i]
                
                lower_bound = detect.get_thresh_bound(
                    array, lower_percent
                )
                upper_bound = detect.get_thresh_bound(
                    array, upper_percent
                )
            else:
                lower_bound = bounds[0][i]
                upper_bound = bounds[1][i]
            
            edit_im = np.where(
                (array > lower_bound) & (array < upper_bound), 1, 0
            )
            
            ax[hist_i].set_title(
                (f'Band: {bounds[0][i]:.2f}-' +
                f'{bounds[1][i]:.2f}{percent_str} | Semilog Histogram'),
                fontsize=24
            )
            ax[hist_i].axvline(lower_bound, color='r')
            ax[hist_i].axvline(upper_bound, color='r')
            ax[hist_i].axvspan(
                lower_bound, upper_bound, 
                color='y', alpha=0.1, lw=0
            )
        elif threshold_type == 'lower':
            if bounds_as_percent:
                percent_bound = bounds[i]
                bound = detect.get_thresh_bound(array, percent_bound)
            else:
                bound = bounds[i]
            
            edit_im = np.where(array > bound, 1, 0)
            
            ax[hist_i].set_title(
                (f'Threshold: >{bounds[i]:.2f}{percent_str} ' +
                '| Semilog Histogram'),
                fontsize=24
            )
            ax[hist_i].axvline(bound, color='r')
            ax[hist_i].axvspan(
                bound, np.max(edges), 
                color='y', alpha=0.1, lw=0
            )
        
        ax[img_i].imshow(edit_im)
        

# Ensemble Plotting Functions
def plot_he_neutral_lines_euv_comparison(fig, he_date_str, mag_date_str,
                                         euv_date_str, nrows=1):
    """Plot a 3 panel comparison of a saturated He I observation, ensemble map
    with neutral lines, and an EUV observation.
    """
    # Extract He I observation
    he_map = prepare_data.get_solis_sunpy_map(ALL_HE_DIR + he_date_str + '.fts')
    if not he_map:
        print(f'{he_date_str} He I observation extraction failed.')
        
    # Extract saved ensemble map array and convert to Sunpy map
    ensemble_file = f'{DETECTION_SAVE_DIR}{he_date_str}_ensemble_map.npy'
    ensemble_map_data = np.load(ensemble_file, allow_pickle=True)[-1]
    ensemble_map = sunpy.map.Map(np.flipud(ensemble_map_data), he_map.meta)
    ensemble_map.plot_settings['cmap'] = colormaps['magma']

    # Extract saved processed magnetogram
    reprojected_smooth_file = (f'{ROTATED_MAG_SAVE_DIR}Mag{mag_date_str}'
                               + f'_He{he_date_str}_smooth.fits')
    reprojected_smooth_map = sunpy.map.Map(reprojected_smooth_file)
    
    euv_map = sunpy.map.Map(ALL_EUV_DIR + euv_date_str + '.fts')
    
    # Plot He observation
    ax = fig.add_subplot(nrows, 3, 1, projection=he_map)
    he_map.plot(axes=ax, vmin=-100, vmax=100, title=he_date_str)
    
    # Plot ensemble map with overlayed neutral lines
    ax = fig.add_subplot(nrows, 3, 2, projection=he_map)
    ensemble_map.plot(axes=ax, title='')
    plot_map_contours(ax, reprojected_smooth_map)
    
    # Plot EUV observation
    ax = fig.add_subplot(nrows, 3, 3, projection=euv_map)
    euv_map.plot(axes=ax, title=euv_date_str)


def plot_ensemble(pre_processed_map_data, ensemble_map_data, map_data_by_ch,
                  confidence_list, metric_list, mask_contour=False):
    """Plot pre-processed map and ensemble map in upper panel. Plot grid
    of individial CH contours on pre-processed map.
    
    Args
        pre_processed_map_data: image array of pre-processed map
        ensemble_map_data: image array of ensemble coronal hole detection
        map_data_by_ch: list of images with isolated detected CHs
        confidence_list: list of confidence levels in mask layers
        metric_list: list of metrics for sorting mask layers
    """
    plot_images([pre_processed_map_data, ensemble_map_data],
                cmaps=['gray', 'magma'])

    image_list = [pre_processed_map_data for _ in range(len(map_data_by_ch))]
    axes = plot_image_grid(image_list, num_cols=5, cmap='afmhot')
    if mask_contour:
        ch_contour_list = map_data_by_ch
    else:
        ch_contour_list = [np.where(~np.isnan(ch_im), 1, 0)
                           for ch_im in map_data_by_ch]

    zipped_items = zip(axes.values(), confidence_list,
                       metric_list, ch_contour_list)
    for ax, confidence, metric, ch_contour in zipped_items:
        ax.set_title(f'{confidence:.1f}% Confidence | {metric:.2f} Metric')
        if mask_contour:
            ax.contour(ch_contour, cmap=plt.cm.gray)
        else:
            ax.contour(ch_contour, cmap=plt.cm.binary)
            
            
def plot_thresh_outcome_vs_time(ax, outcome_df, date_str, cmap, ylabel):
    """Plot outcome stacked plot vs time for thresholded maps.
    
    Args
        ax: matplotlib axes object to plot on
        outcome_df: Pandas dataframe of outcome by confidence level
            over time
        date_str: Date string at which to plot a vertical line
        cmap: Matplotlib colormap name
        ylabel: string for y axis label
    """
    # Array of differences in outcomes between confidence levels
    diff_df = outcome_df.diff(periods=-1, axis=1)
    diff_df[diff_df.columns[-1]] = outcome_df[outcome_df.columns[-1]]
    
    # Datetime and data array for stack plot
    datetimes = outcome_df.index
    stack_plot_array = np.flipud(diff_df.to_numpy().T)
    
    # Threshold labels for legend
    threshold_label_list = [
        f'{thresh_level}% of Histogram Peak Threshold'
        for thresh_level in reversed(outcome_df.columns)
    ]
    
    cmap = colormaps[cmap]
    color_list = cmap(np.linspace(0, 0.75, len(threshold_label_list)))

    ax.stackplot(datetimes, stack_plot_array,
                labels=threshold_label_list, colors=color_list)
    
    # Vertical line for datetime indicator
    vline_datetime = datetime.strptime(date_str, DICT_DATE_STR_FORMAT)
    min_outcome = outcome_df[max(outcome_df.columns)].min()
    max_outcome = outcome_df[min(outcome_df.columns)].max()
    ax.vlines(x=[vline_datetime, vline_datetime], ymax=2*max_outcome, ymin=0,
               colors='k', linestyles='dashed')
    
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    ax.set_ylabel(ylabel)
    ax.set_xlim([datetimes[0], datetimes[-1]])
    ax.set_ylim([0.9*min_outcome, 1.25*max_outcome])
    ax.legend(reverse=True)


def plot_outcome_vs_time(ax, outcome_df, date_str, cmap, ylabel):
    """Plot outcome stacked plot vs time for ensemble maps.
    
    Args
        ax: matplotlib axes object to plot on
        outcome_df: Pandas dataframe of outcome by confidence level
            over time
        date_str: Date string at which to plot a vertical line
        cmap: Matplotlib colormap name
        ylabel: string for y axis label
    """
    # Array of differences in outcomes between confidence levels
    diff_df = outcome_df.diff(periods=-1, axis=1)
    diff_df[diff_df.columns[-1]] = outcome_df[outcome_df.columns[-1]]
    
    # Datetime and data array for stack plot
    datetimes = outcome_df.index
    stack_plot_array = np.flipud(diff_df.to_numpy().T)
    
    # Confidence labels for legend
    confidence_label_list = [
        f'{confidence_level}% Confidence'
        for confidence_level in reversed(outcome_df.columns)
    ]
    
    cmap = colormaps[cmap]
    color_list = cmap(np.linspace(0, 0.75, len(confidence_label_list)))

    ax.stackplot(datetimes, stack_plot_array,
                labels=confidence_label_list, colors=color_list)
    
    # Vertical line for datetime indicator
    vline_datetime = datetime.strptime(date_str, DICT_DATE_STR_FORMAT)
    max_outcome = outcome_df[min(outcome_df.columns)].max()
    ax.vlines(x=[vline_datetime, vline_datetime], ymax=1.2*max_outcome, ymin=0,
               colors='k', linestyles='dashed')
    
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    ax.set_ylabel(ylabel)
    ax.set_xlim([datetimes[0], datetimes[-1]])
    ax.set_ylim([0, 1.1*max_outcome])
    ax.legend(reverse=True)

