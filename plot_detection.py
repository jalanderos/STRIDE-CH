"""
"""

import numpy as np
from scipy import ndimage
from skimage import filters
from astropy.io import fits
from datetime import datetime
from matplotlib import pyplot as plt
import plotly.express as px
from sunpy.coordinates.sun import carrington_rotation_number

import run_detection


# Module variables
DICT_DATE_STR_FORMAT = '%Y_%m_%d__%H_%M'
EMPTY_ARRAY = np.zeros((2,2))
GREEN = '#6ece58'
BLUE = '#3e4989'
ORANGE = '#fd9668'
PURPLE = '#721f81'


# Plotting Functions
def plot_hists(arrays, titles, semilogy=True):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(40, 10))
    ax = axes.ravel()
    
    hist, edges = run_detection.get_hist(arrays[0], bins_as_percent=False, n=200)

    ax[0].set_title(titles[0], fontsize=24)
    ax[0].imshow(arrays[0], cmap=plt.cm.gray)
    if semilogy:
        ax[1].semilogy(edges[0:-1], hist)
    else:
        ax[1].plot(edges[0:-1], hist)
        
    ax[2].set_title(titles[1], fontsize=24)
    ax[2].imshow(arrays[1], cmap=plt.cm.gray)
    hist, edges = run_detection.get_hist(arrays[1], bins_as_percent=False, n=200)
    if semilogy:
        ax[3].semilogy(edges[0:-1], hist)
    else:
        ax[3].plot(edges[0:-1], hist)
        
        
def plot_ims(im_list, title_list=[''], cmaps=False):
    """
    """
    num_ims = len(im_list)
    title_list.extend(['' for _ in range(num_ims - len(title_list))])
    
    if num_ims == 1:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        
        if cmaps:
            ax.imshow(im_list[0], cmap=cmaps[0])
        else:
            ax.imshow(im_list[0], cmap=plt.cm.gray)
            
        ax.set_title(title_list[0], fontsize=24)
    else:
        fig, axes = plt.subplots(nrows=1, ncols=num_ims, figsize=(10*num_ims, 10))
        ax = axes.ravel()
    
        for i in range(0,num_ims):
            if cmaps:
                ax[i].imshow(im_list[i], cmap=cmaps[i])
            else:
                ax[i].imshow(im_list[i], cmap=plt.cm.gray)

            ax[i].set_title(title_list[i], fontsize=24)


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
                
    plot_ims(im_list, title_list, cmaps=cmaps)
    
    return im_list


def plot_fits_content(fits_dict, date_str, fits_cmaps):
    """
    """
    fits_tuple = fits_dict[date_str]

    im_list = []

    num_arrays = len(fits_tuple) - 1
    title_list = fits_tuple[-1]

    for i in range(num_arrays):
        if i == 0:
            array = np.clip(fits_tuple[i], -50, 50)
        else:
            array = fits_tuple[i]
        
        im_list.append(array)        
        
    plot_ims(im_list, title_list, fits_cmaps)


# Magnetogram Plotting
def plot_magnetogram(ax, magnetogram, smooth_size_percent, bound):
   """Add magnetogram saturated at bounds and contour map of large scale
   neutral regions to axes
   
   Args
      ax: matplotlib axes
      magnetogram: magnetogram image array
      smooth_size_percent: float to specify uniform smoothing kernel size
         as a percentage of image size
      bound: float to clip magnetogram display to (G)
   """
   plot_contours(ax, magnetogram, smooth_size_percent)

   # Pre-process magnetogram
   process_magnetogram = np.where(magnetogram == 0, np.nan, magnetogram)
   process_magnetogram = np.clip(process_magnetogram, -bound, bound)

   # Plot the photospheric map
   ax.imshow(process_magnetogram, cmap=plt.cm.gray)


def plot_contours(ax, magnetogram, smooth_size_percent):
    """Add contour map of large scale neutral regions to axes
    
    Args
        ax: matplotlib axes
        magnetogram: magnetogram image array
        smooth_size_percent: float to specify uniform smoothing kernel size
            as a percentage of image size
        bound: float to clip magnetogram display to (G)
    """
    smooth_size = smooth_size_percent/100 *magnetogram.shape[0]
    smoothed_magnetogram = ndimage.uniform_filter(
        magnetogram, smooth_size
    )
    # Remove background after to avoid empty output 
    # from removal then smoothing
    smoothed_magnetogram = np.where(
        magnetogram == 0, np.nan, smoothed_magnetogram
    )
    # Plot large scale neutral lines
    ax.contour(
        smoothed_magnetogram, colors='yellow', 
        levels=[0], linewidths=1.5
    )


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


# EUV Plotting
def plot_euv(ax, euv):
   """Add calibrated EUV image to axes
   
   Args
      ax: matplotlib axes
      euv: EUV image array
   """
   # Plot the EUV map
   ax.imshow(euv, cmap=plt.cm.gray)


# NSO Plotting
def plot_ch_map(date_str_list, cr_str, ch_map_dict):
    """Plot NSO detected CH Carrington map.
    """
    # Display selected column number corresponding to date list
    selected_datetime_list = [
        datetime.strptime(
            date_str, DICT_DATE_STR_FORMAT)
        for date_str in date_str_list
    ]
    selected_cr_list = [
        carrington_rotation_number(selected_datetime)
        for selected_datetime in selected_datetime_list
    ]
    
    cr_str_list = cr_str.split('_')
    cr_num_list = [float(cr_str) for cr_str in cr_str_list]
    
    cr_range = cr_num_list[-1] - cr_num_list[0]
    cr_percent_list = [
        (selected_cr - cr_num_list[0])/cr_range
        for selected_cr in selected_cr_list
    ]
    
    ch_map = ch_map_dict[cr_str]
    rows, cols = ch_map.shape
    
    selected_col_list = [
        cols - cr_percent*cols
        for cr_percent in cr_percent_list
    ]
    
    print('Selected Date Columns:')
    for date_str, selected_col in zip(
        date_str_list, selected_col_list):
        print(f'{date_str}: {selected_col:.1f}px \t', end='')

    # Prepare the figure and axes with map projection
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot()
    ax.set_title(f'CR{cr_str}', fontsize=20)
    
    ax.imshow(ch_map, extent=[0,cols, rows, 0])
    ax.vlines(x=selected_col_list, ymin=rows, ymax=0, linestyles='dashed',
              colors='black')


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
    hist, edges = run_detection.get_hist(array, bins_as_percent=False)
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
                
                lower_bound = run_detection.get_thresh_bound(
                    array, lower_percent
                )
                upper_bound = run_detection.get_thresh_bound(
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
                bound = run_detection.get_thresh_bound(array, percent_bound)
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
        
        
def plot_pixel_percent_bars(ax, parameter_list, pixel_percent_list,
                            max_diff, cutoff, selected_parameter_num,
                            step, title, unit, xlabel, thresh=True):
    bar_width = 0.8*step
    selected_parameters = parameter_list[selected_parameter_num:]
    
    ax.set_title(f'{title}\n Cutoff: {selected_parameters[0]}{unit} | ' +
                 f'Max Difference: {max_diff:.1f}%' , fontsize=28)
    ax.set_xlabel(xlabel, fontsize=24)
    
    ax.set_ylabel('Pixel Percentage (%)', fontsize=24)
    
    ax.plot([parameter_list[0] - step/2, parameter_list[-1] + step/2], [cutoff, cutoff], 
               linestyle='--', color='k', linewidth=3)
    
    if thresh:
        ax.bar(parameter_list, pixel_percent_list, 
               width=bar_width, color=BLUE)
        ax.bar(selected_parameters, 
               pixel_percent_list[selected_parameter_num:], 
               width=bar_width, color=GREEN)
    else:
        ax.bar(parameter_list, pixel_percent_list,
               width=bar_width, color=PURPLE)
        ax.bar(selected_parameters,
               pixel_percent_list[selected_parameter_num:], 
               width=bar_width, color=ORANGE)


# Segmentation Outcome Plots
def plot_heat_map(outcome_list, title, percent_of_peak_list, morph_radius_list, 
                  color_scale='Magma'):
    # Reverse order to facilitate plotting
    y_axis_list = morph_radius_list.copy()
    y_axis_list.reverse()
    
    outcome_map = np.flipud(np.reshape(
        outcome_list, (len(percent_of_peak_list),len(morph_radius_list))).T)

    fig = px.imshow(outcome_map, 
                    labels=dict(x='Threshold Level as Percent of Peak (%)',
                                y='SE Disk Radius (px)'),
                    x=percent_of_peak_list, y=y_axis_list,
                    aspect='auto', color_continuous_scale=color_scale)
    fig.update_layout(title=title, width=700)
    fig.show()
    
    
def plot_heat_map_band(outcome_list, heat_map_title, lower_bound, upper_bound,
                       percent_of_peak_list, morph_radius_list, 
                       array, all_ch_masks_list, title_list, color_scale='Magma'):
    edit_outcome_list = outcome_list.copy()
    
    # Index list of outcomes within bounds 
    idx_list = [i for i in range(len(edit_outcome_list)) 
              if edit_outcome_list[i] >= lower_bound and edit_outcome_list[i] <= upper_bound]
    num_ch_masks = len(idx_list)
    
    if not num_ch_masks:
        print('No masks in outcome range')
        return
    
    max_outcome = max(edit_outcome_list)
    # Highlight outcomes within bounds
    for i in idx_list:
        edit_outcome_list[i] = 2*max_outcome

    plot_heat_map(edit_outcome_list, heat_map_title,
                  percent_of_peak_list, morph_radius_list, color_scale)
    
    if num_ch_masks == 1:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot()
        
        outcome_idx = idx_list[0]
        ax.imshow(array, cmap=plt.cm.afmhot)
        ax.contour(all_ch_masks_list[outcome_idx], linewidths=0.5, cmap=plt.cm.gray)
        ax.set_title(title_list[outcome_idx], fontsize=18)
    else:
        fig, axes = plt.subplots(nrows=1, ncols=num_ch_masks, figsize=(6*num_ch_masks, 6))
        ax = axes.ravel()
    
        for i in range(num_ch_masks):
            outcome_idx = idx_list[i]
            ax[i].imshow(array, cmap=plt.cm.afmhot)
            ax[i].contour(all_ch_masks_list[outcome_idx], linewidths=0.5, cmap=plt.cm.gray)
            ax[i].set_title(title_list[outcome_idx], fontsize=18)
        

def plot_sorted_ch_hists(array, ch_mask, apply_gradient, hist_stat,
                         descend_sort=False):
    """Plot segmented CH histograms sorted by histogram statistics.
    
    Args
        array: image to process
        ch_mask: binary coronal holes mask
        apply_gradient: boolean to specify taking spatial gradient of image
        hist_stat: str to specify histogram sorting statistic
            'median', 'width', 'tail_width'
        descend_sort: boolean to specify sorting CHs from greatest to least
            statistic
    """
    # Masked array of candidate CHs
    masked_candidates = run_detection.get_masked_candidates(array, ch_mask)
    if apply_gradient:
        masked_candidates = filters.sobel(masked_candidates)
    
    # Isolated images of detected CHs
    isolated_ch_im_list = run_detection.get_isolated_ch_im_list(
        masked_candidates, ch_mask
    )
    num_ch = len(isolated_ch_im_list)
    
    # Compute statistics for each CH
    medians = run_detection.get_ch_medians(isolated_ch_im_list)
    ch_band_widths = run_detection.get_ch_band_widths(isolated_ch_im_list)
    
    # Histogram x limit bounds
    hist_xlim_min = np.mean(medians)
    if not apply_gradient:
        hist_xlim_min = hist_xlim_min - 2*np.max(ch_band_widths)
    hist_xlim_max = np.mean(medians) + 2*np.max(ch_band_widths)
    
    # Obtain indices of candidates sorted by specifed mode
    if hist_stat == 'median':
        sorted_candidate_idxs = np.argsort(medians)
        titles = [f'Median: {median:.1f}'
                  for median in medians]
    elif hist_stat == 'width':
        sorted_candidate_idxs = np.argsort(ch_band_widths)
        titles = [f'90% Band Width: {ch_band_width:.1f}'
                  for ch_band_width in ch_band_widths]
    elif hist_stat == 'tail_width':
        ch_lower_tail_width_list = run_detection.get_ch_lower_tail_widths(
            isolated_ch_im_list
        )
        sorted_candidate_idxs = np.argsort(ch_lower_tail_width_list)
        titles = [f'1% to Peak Width: {ch_lower_tail_width:.1f}'
                  for ch_lower_tail_width in ch_lower_tail_width_list]
    
    if descend_sort:
        sorted_candidate_idxs = np.flip(sorted_candidate_idxs)

    for r in range(int(np.ceil(num_ch/2))):
        fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(60, 10))
        ax = axes.ravel()
        
        for c in range(2):
            i = 2*r + c
            ax_i = 3*c
            if i + 1 > num_ch:
                return
            
            # Retrieve isolated CH image and contour
            ch_num = sorted_candidate_idxs[i]
            ch_im = isolated_ch_im_list[ch_num]
            ch_contour = np.where(~np.isnan(ch_im), 1, 0)

            # Zoom in on an isolated CH
            y, x = np.where(~np.isnan(ch_im))
            ch_zoom = ch_im[np.min(y) - 10:np.max(y) + 10,
                             np.min(x) - 10:np.max(x) + 10]
                
            hist, edges = run_detection.get_hist(
                ch_zoom[~np.isnan(ch_zoom)], bins_as_percent=False, n=200
            )
            
            ax[ax_i].set_title(f'Hole {ch_num + 1}', fontsize=32)
            ax[ax_i].imshow(array, cmap=plt.cm.afmhot)
            ax[ax_i].contour(ch_contour, cmap=plt.cm.binary)
            
            if apply_gradient:
                cmap = plt.cm.viridis
            else:
                cmap = plt.cm.magma

            ax[ax_i + 1].imshow(ch_zoom, cmap)

            ax[ax_i + 2].set_title(titles[ch_num], fontsize=32)
            ax[ax_i + 2].bar(edges[0:-1], hist)
            ax[ax_i + 2].set_xlim([hist_xlim_min, hist_xlim_max])


# Ensemble Plotting Functions
def plot_ensemble(array, ensemble_map, confidence_list, ch_ims):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    ax = axes.ravel()
    
    ax[0].imshow(array, cmap=plt.cm.gray)
    ax[1].imshow(ensemble_map, cmap=plt.cm.magma)
    
    num_ch = len(ch_ims)
    num_cols = 5
    num_rows = int(np.ceil(num_ch/num_cols))

    for row in range(num_rows):
        fig, axes = plt.subplots(nrows=1, ncols=num_cols,
                                 figsize=(10*num_cols, 10))
        ax = axes.ravel()
        
        for col in range(num_cols):
            i = num_cols*row + col
            if i + 1 > len(ch_ims):
                return

            title = f'{confidence_list[i]:.1f}% Confidence'
            
            ch_im = ch_ims[i]
            ch_im = np.where(ch_im, ch_im, np.nan)
            ch_contour = np.where(~np.isnan(ch_im), 1, 0)
            
            ax[col].set_title(title, fontsize=28)
            ax[col].imshow(array, cmap=plt.cm.afmhot)
            ax[col].contour(ch_contour, cmap=plt.cm.binary)
            
            
def plot_ensemble_comparison(eqw, date, ensemble_map, euv):
    fig, axes = plt.subplots(nrows=1, ncols=3, 
                             figsize=(30, 10))
    ax = axes.ravel()
    
    ax[0].set_title(date, fontsize=24)
    ax[0].imshow(eqw, cmap=plt.cm.gray)

    ax[1].imshow(ensemble_map, cmap=plt.cm.magma)
    
    ax[2].imshow(euv)


def plot_mag_ensemble_comparison(eqw, date, ensemble_map, magnetogram, euv):
    """Work in progress"""
    fig, axes = plt.subplots(nrows=1, ncols=4, 
                             figsize=(40, 10))
    ax = axes.ravel()
    
    ax[0].set_title(date, fontsize=24)
    ax[0].imshow(eqw, cmap=plt.cm.gray)

    ax[1].imshow(ensemble_map, cmap=plt.cm.magma)
    
    plot_magnetogram(ax[2], magnetogram, smooth_size_percent=9, bound=50)
    
    ax[3].imshow(euv)


# UNUSED
def plot_euv_comparison(eqw_date_list, eqw_dict, euv_dict):
    """
    """
    eqw_date_iterator = iter(eqw_date_list)

    for eqw_date in eqw_date_iterator:
        next_eqw_date = next(eqw_date_iterator, '')
        im_list = [
            eqw_dict[eqw_date],
            euv_dict.get(eqw_date, EMPTY_ARRAY),
            eqw_dict.get(next_eqw_date, EMPTY_ARRAY),
            euv_dict.get(next_eqw_date, EMPTY_ARRAY)
        ]
        title_list = [eqw_date, '', next_eqw_date, '']

        plot_ims(im_list, title_list)


def plot_cr_magnetogram_comparison(date_str, smooth_filter_size, bound, gong_dict,
                                eqw_dict, euv_dict):
    """CR map magnetogram comparison
    """
    date_str = '2014_06_26__14_19'

    # # Size of uniform kernel for magnetogram smoothing in contour production (px)
    # smooth_filter_size = 10

    # # Bounds for clipping magnetogram display (G)
    # bound = 50

    # plot_detection.plot_magnetogram_comparison(
    #     date_str, smooth_filter_size, bound,
    #     gong_dict=GONG_DICT, eqw_dict=EQW_DICT, euv_dict=EUV_DICT
    # )

    plot_cr_magnetogram(date_str, smooth_filter_size, bound, gong_dict)

    im_list = [eqw_dict.get(date_str), euv_dict.get(date_str)]

    im_list = [im for im in im_list if im is not None]

    num_ims = len(im_list)
    title_list = [date_str]
    title_list.extend(['' for i in range(num_ims - 1)])
        
    plot_ims(im_list, title_list)
    

def plot_cr_magnetogram(date_str, smooth_filter_size, bound, gong_dict):
    """Carrington rotation map magnetogram plot
    """
    # Extract observation datetime
    selected_datetime = datetime.strptime(date_str, DICT_DATE_STR_FORMAT)

    cr_num = carrington_rotation_number(selected_datetime)
    cr_int = int(cr_num)
    cr_decimal = np.mod(cr_num,1)
    
    selected_lon = 360*(1 - cr_decimal)
    print(f'Selected Date Longitude: {selected_lon:.2f}deg')

    # Prepare the figure and axes with map projection
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot()
    ax.set_title(f'CR{cr_int}', fontsize=20)

    # Array of photospheric field intensity values
    magnetogram = gong_dict[cr_int]
    
    plot_contour_map(ax, magnetogram, smooth_filter_size)
        
    # Plot the photospheric map
    magnetogram = np.clip(magnetogram, -bound, bound)
    
    extent = [0, 360, -90, 90]
    ax.imshow(magnetogram, origin='upper', extent=extent, cmap=plt.cm.gray)
    ax.vlines(x=selected_lon, ymin=-90, ymax=90, linestyles='dashed',
              colors='black')


def plot_projected_magnetogram(date_str, smooth_filter_size, bound, gong_dict):
    """
    """
    # Extract observation datetime
    selected_datetime = datetime.strptime(date_str, DICT_DATE_STR_FORMAT)

    CR = carrington_rotation_number(selected_datetime)
    CR_int = int(CR)
    CR_decimal = np.mod(CR,1)
    
    selected_lon = 360*(1 - CR_decimal)
    print(f'Selected Date Longitude: {selected_lon:.2f}deg')
    
#     # Read variables from FITS file
#     carrlong = 0

#     # Set up the WSA coordinate system transform for plotting
#     wsa_coordsys_transform = crs.PlateCarree(
#         central_longitude=((carrlong - 180) % 360))

#     # Lookup the viewing direction of the satellite at the plot time.
#     lat_set, lon_set = get_viewing_direction(plot_time)

#     # Create map projection
#     projection = crs.Orthographic(central_latitude=lat_set,
#                                   central_longitude=lon_set)

    # Prepare the figure and axes with map projection
    fig = plt.figure(figsize=(10, 10)) #, dpi=100)
#     fig.patch.set_facecolor('black')
#     ax = fig.add_axes([0, .1, 1, .8], projection=projection, frameon=False)
    ax = fig.add_subplot()
    ax.set_title(f'CR{CR_int}', fontsize=20)

    # Array of photospheric field intensity values
    magnetogram = gong_dict[CR_int]
    
    plot_contour_map(ax, magnetogram, smooth_filter_size) #, wsa_coordsys_transform)
        
    # Plot the photospheric map
    magnetogram = np.clip(magnetogram, -bound, bound)
    
    # Display the magnetic field on the sphere. To handle meridian crossing,
    # a PlateCarree transform is used with a custom central longitude.
    extent = [0, 360, -90, 90]
    ax.imshow(magnetogram, origin='upper', extent=extent, cmap=plt.cm.gray)
#     , transform=wsa_coordsys_transform)


def plot_contour_map(ax, magnetogram, smooth_filter_size):
#     , grid, wsa_coordsys_transform):
    """Add contour map of neutral regions (showing margins between polarity changes) to axes.
    
    Args
       ax: matplotlib axes object
       field_int: array of magnetic field intensity
       grid: grid resolution value
       wsa_coordsys_transform: cartopy projection object for coordinate system transformation
    """
    # Latitudes for contour map    
    lat_axis = np.arange(-90, 90, dtype=np.float64)

    # Longitudes for contour map
    lon_axis = np.arange(360, dtype=np.float64)
    
    # Shift to cell centers
    lat_axis += 0.5
    lon_axis += 0.5

    # Photospheric field for contour map
    magnetogram = np.flipud(magnetogram)

    # Smooth contours
    smoothed_magnetogram = ndimage.uniform_filter(magnetogram, size=smooth_filter_size)
    
    x_coord, y_coord = np.meshgrid(lon_axis, lat_axis)
    ax.contour(x_coord, y_coord, smoothed_magnetogram, colors='yellow', 
               levels=[0], linewidths=1)
#     , transform=wsa_coordsys_transform)


# def get_viewing_direction(plot_time):
#     """This routine determines the direction that the visualization is facing
#     for the orthographic projection of the sun.
    
#     Given a satellite name and the casefile's SATLOCSDIR variable, it looks
#     for the ascii table holding the location data and interpolates to target
#     given day.

#     Args
#       plot_time: time of visualization (datetime UTC, no timezone)
#     Returns
#       lat_set: solar latitude (float)
#       lon_set: solar longitude (float)
#     """
#     sat_locs_file = '../SAT_LOCS/Earth_locs.dat'
#     locs_df = pd.read_csv(
#         sat_locs_file, delim_whitespace=True,
#         names=['sat_time_jd', 'sat_lon', 'carrot_num', 'sat_lat', 'radii']
#     )

#     plot_time_jd = Time(plot_time).jd

#     # Use simple interpolation for latitude
#     lat_set = np.interp(plot_time_jd, locs_df.sat_time_jd, locs_df.sat_lat)
    
#     # Manually handle interpolation of longitude to account for wrapping
#     # If it is not done this way, the interpolation will create erroneous
#     # "spinning" of the viewing direction for the times between daily
#     # records which wrap.
#     idx = locs_df.sat_time_jd.searchsorted(plot_time_jd)
#     time0, lon0 = (locs_df.sat_time_jd[idx - 1], locs_df.sat_lon[idx - 1])
#     time1, lon1 = (locs_df.sat_time_jd[idx], locs_df.sat_lon[idx])

#     if lon0 < lon1:
#         lon0 += 360    # handle wrapping

#     fade_param = (plot_time_jd - time0) / (time1 - time0) # between 0 and 1
#     lon_set = lon0 + fade_param * (lon1 - lon0)
#     lon_set = lon_set % 360

#     return lat_set, lon_set
