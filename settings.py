"""
Script for declaration of data locations in file system.
"""

# CH Detection Data Paths ----------------------------------------------------
DATA_DIR = 'assets/'
DICT_DATE_STR_FORMAT = '%Y_%m_%d__%H_%M'
DATA_FITS_FORMAT = '{data_dir}{date_str}.fts'

# Comparison Observation Paths -----------------------------------------------
HE_DIR = DATA_DIR + 'He/'
TEST_HE_DIR = DATA_DIR + 'Test_He/'

MAG_DIR = DATA_DIR + 'Mag/'

EUV_DIR = DATA_DIR + 'EUV/'

NSO_INPUT_DIR = DATA_DIR + 'NSO_Input/'
NSO_SINGLE_DIR = DATA_DIR + 'NSO_Output/single/'
NSO_MERGED_DIR = DATA_DIR + 'NSO_Output/merged/'

# Detection Data Product Paths -----------------------------------------------
OUTPUT_DIR = 'output/'

# Directories of intermediate data product FITS file sunpy maps
MAP_DIR = OUTPUT_DIR + 'FITS_Maps/'
ROTATED_MAG_SAVE_DIR = MAP_DIR + 'Rotated_Mag/'
HELIOGRAPH_MAG_SAVE_DIR = MAP_DIR + 'Heliographic_Mag/'
# Ecperimental He I/EUV ratio maps
RATIO_SAVE_DIR = MAP_DIR + 'Ratio/'

# Directories of preprocessed He I maps as Sunpy maps in FITS files
# or as arrays in numpy files in older versions
PREPROCESS_DIR = OUTPUT_DIR + 'Preprocess/'
# PREPROCESS_NPY_SAVE_DIR = PREPROCESS_DIR + 'Band_Pass/'
# PREPROCESS_NPY_SAVE_DIR = PREPROCESS_DIR + 'Rescale/'
# PREPROCESS_NPY_SAVE_DIR = PREPROCESS_DIR + 'Rescale_Center/'
PREPROCESS_NPY_SAVE_DIR = PREPROCESS_DIR + 'v0_1/'
# PREPROCESS_NPY_SAVE_DIR = PREPROCESS_DIR + 'v0_4/'
# PREPROCESS_NPY_SAVE_DIR = PREPROCESS_DIR + 'v0_4_Ratio/'

# v1.0 preprocessing method has been preserved from v0.5.1
PREPROCESS_MAP_SAVE_DIR = PREPROCESS_DIR + 'v0_5_1/'
# PREPROCESS_MAP_SAVE_DIR = PREPROCESS_DIR + 'vY/'

# Directories of data products for various versions
DETECT_DIR = OUTPUT_DIR + 'Detection/'
# DETECTION_VERSION_DIR = DETECT_DIR + 'v0_1/'
# DETECTION_VERSION_DIR = DETECT_DIR + 'v0_2/'
# DETECTION_VERSION_DIR = DETECT_DIR + 'v0_3/'
# DETECTION_VERSION_DIR = DETECT_DIR + 'v0_3_Band_Pass/'
# DETECTION_VERSION_DIR = DETECT_DIR + 'v0_3_Rescale/'
# DETECTION_VERSION_DIR = DETECT_DIR + 'v0_3_Rescale_Center/'
# DETECTION_VERSION_DIR = DETECT_DIR + 'v0_4/'
# DETECTION_VERSION_DIR = DETECT_DIR + 'v0_4_Single/'
# DETECTION_VERSION_DIR = DETECT_DIR + 'v0_4_Ratio_Thresh_80/'
# DETECTION_VERSION_DIR = DETECT_DIR + 'v0_4_Ratio_Thresh_100/'
# DETECTION_VERSION_DIR = DETECT_DIR + 'v0_4_Unipolar_Even/'
# DETECTION_VERSION_DIR = DETECT_DIR + 'v0_4_Unipolar/'
# DETECTION_VERSION_DIR = DETECT_DIR + 'v0_5/'
# DETECTION_VERSION_DIR = DETECT_DIR + 'v0_5_1_Conservative/'
# DETECTION_VERSION_DIR = DETECT_DIR + 'vY_Aggressive/'
# DETECTION_VERSION_DIR = DETECT_DIR + 'vY_Conservative/'
# DETECTION_VERSION_DIR = DETECT_DIR + 'v0_5_1/'
# DETECTION_VERSION_DIR = DETECT_DIR + 'v0_5_1_KPVT/'
# DETECTION_VERSION_DIR = DETECT_DIR + 'v0_5_1_No_Thresh/'
DETECTION_VERSION_DIR = DETECT_DIR + 'v1_0/'
# DETECTION_VERSION_DIR = DETECT_DIR + 'v1_0_No_Thresh/'


# Directories of ensemble segmentation maps as Sunpy maps in FITS files
# or as arrays in numpy files in older versions
DETECTION_NPY_SAVE_DIR = DETECTION_VERSION_DIR + 'Saved_npy_Files/'
DETECTION_MAP_SAVE_DIR = DETECTION_VERSION_DIR + 'Saved_fits_Files/'


# Date Range Options for Detection Data Products -----------------------------
# DATE_RANGE: tuple of start and end date or list of He I datetimes

# # KPVT period in declining Solar Cycle 23
# DATE_RANGE = ('2003_07_01__00_00', '2003_08_01__00_00')
# DATE_DIR = '2003_07/'

# # Rockwell 2004 period in declining Solar Cycle 23
# DATE_RANGE = ('2004_11_01__00_00', '2004_12_11__00_00')
# DATE_DIR = '2004_11_2004_12/'

# # Rockwell 2009 period in minimum post-Solar Cycle 23
# DATE_RANGE = ('2009_10_06__00_00', '2009_10_31__23_00')
# DATE_DIR = '2009_10/'

# # Sarnoff 2010 period in minimum post-Solar Cycle 23
# DATE_RANGE = ('2010_05_01__00_00', '2010_10_01__00_00')
# DATE_DIR = '2010_05_2010_09/'

# # Sarnoff 2012 period in rising Solar Cycle 24
# DATE_RANGE = ('2012_04_01__00_00', '2012_09_01__00_00')
# DATE_DIR = '2012_04_2012_08/'

# # April of 2012 period
# DATE_RANGE = ('2012_04_01__00_00', '2012_05_01__00_00')
# DATE_DIR = '2012_04/'

# # June of 2012 period
# DATE_RANGE = ('2012_06_01__00_00', '2012_06_30__00_00')
# DATE_DIR = '2012_06/'

# # August of 2012 period
# DATE_RANGE = ('2012_08_01__00_00', '2012_08_30__00_00')
# DATE_DIR = '2012_08/'

# # Sarnoff 2015 period in max of Solar Cycle 24
# DATE_RANGE = ('2015_01_01__00_00', '2015_07_01__00_00')
# DATE_DIR = '2015_01_2015_06/'

# COSPAR cases
# DATE_RANGE = [
#     '2015_01_04__20_30',
#     '2015_01_20__20_25',
#     '2015_02_10__18_45',
#     '2015_03_31__18_13',
#     '2015_04_18__17_22',
#     '2015_06_06__16_08'
# ]
# DATE_DIR = 'COSPAR/'

# # COSPAR case 02/10/2015
# DATE_RANGE = ('2015_02_03__00_00', '2012_02_17__00_00')
# DATE_DIR = 'COSPAR_2015_02_10/'

# # COSPAR case 06/06/2015
# DATE_RANGE = ('2015_06_06__00_00', '2015_06_12__00_00')
# DATE_DIR = 'COSPAR_2015_06_06/'

# Select few dates for analysis
# DATE_RANGE = [
#     '2013_02_16__18_23',
#     '2013_06_19__15_35',
#     '2015_03_29__18_00',
#     '2015_04_28__17_25',
#     '2015_02_27__20_39',
#     '2015_06_19__16_33',
#     '2015_07_16__17_12',
# ]
# DATE_DIR = 'Selected_Maps/'

# All dates
DATE_RANGE = ('2000_01_01__00_00', '2020_01_01__00_00')
DATE_DIR = 'All_Dates/'

# # TEST
# DATE_RANGE = ('2012_06_01__00_00', '2012_06_03__00_00')
# DATE_DIR = 'Test/'

# # CR 2136
# DATE_RANGE = ('2013_04_01__00_00', '2013_06_01__00_00')

# # CR 2151
# DATE_RANGE = ('2014_06_01__00_00', '2014_06_30__00_00')

DETECTION_IMAGE_DIR = DETECTION_VERSION_DIR + DATE_DIR
