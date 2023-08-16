"""
Script for declaration of He I CH Detection data settings
"""

# CH Detection Data Paths ----------------------------------------------------
DATA_DIR = 'assets/'
DICT_DATE_STR_FORMAT = '%Y_%m_%d__%H_%M'


# Comparison Observation Paths -----------------------------------------------
ALL_HE_DIR = DATA_DIR + 'All_He/'
SELECT_HE_DIR = DATA_DIR + 'Selected_He/'
TEST_HE_DIR = DATA_DIR + 'Test_He/'

ALL_MAG_DIR = DATA_DIR + 'All_Mag/'
SELECT_MAG_DIR = DATA_DIR + 'Selected_Mag/'

IMG_EUV_DIR = DATA_DIR + 'Img_EUV/'
ALL_EUV_DIR = DATA_DIR + 'All_EUV/'
SELECT_EUV_DIR = DATA_DIR + 'Selected_EUV/'

NSO_INPUT_DIR = DATA_DIR + 'NSO_Input/'
NSO_SINGLE_DIR = DATA_DIR + 'NSO_Output/single/'
NSO_MERGED_DIR = DATA_DIR + 'NSO_Output/merged/'


# Detection Data Product Paths -----------------------------------------------
OUTPUT_DIR = 'output/'

MAP_DIR = OUTPUT_DIR + 'FITS_Maps/'
PREPROCESS_DIR = OUTPUT_DIR + 'Preprocess/'
DETECT_DIR = OUTPUT_DIR + 'Detection/'

# Intermediate FITS maps
ROTATED_MAG_SAVE_DIR = MAP_DIR + 'Rotated_Mag/'
HELIOGRAPH_MAG_SAVE_DIR = MAP_DIR + 'Heliographic_Mag/'

RATIO_SAVE_DIR = MAP_DIR + 'Ratio/'

# Detection version path options
# PREPROCESS_SAVE_DIR = PREPROCESS_DIR + 'Band_Pass/'
# PREPROCESS_SAVE_DIR = PREPROCESS_DIR + 'Rescale/'
# PREPROCESS_SAVE_DIR = PREPROCESS_DIR + 'Rescale_Center/'
# PREPROCESS_SAVE_DIR = PREPROCESS_DIR + 'v0_1/'
PREPROCESS_SAVE_DIR = PREPROCESS_DIR + 'v0_4/'
# PREPROCESS_SAVE_DIR = PREPROCESS_DIR + 'v0_4_Ratio/'

PREPROCESS_MAP_SAVE_DIR = PREPROCESS_DIR + 'v0_6/'

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
DETECTION_VERSION_DIR = DETECT_DIR + 'v0_5/'
# DETECTION_VERSION_DIR = DETECT_DIR + 'v0_5_HG_No_U_Thresh/'
# DETECTION_VERSION_DIR = DETECT_DIR + 'v0_6/'

# Detection save files
DETECTION_SAVE_DIR = DETECTION_VERSION_DIR + 'Saved_npy_Files/'

DETECTION_MAP_SAVE_DIR = DETECTION_VERSION_DIR + 'Saved_fits_Files/'


# Date Range Options for Detection Data Products -----------------------------
# # Rockwell period in declining Solar Cycle 23
# DATE_RANGE = ('2004_11_01__00_00', '2005_01_01__00_00')
# DATE_DIR = '2004_11_2004_12/'

# # Sarnoff-GMU period in rising Solar Cycle 24
# DATE_RANGE = ('2012_04_01__00_00', '2012_09_01__00_00')
# DATE_DIR = '2012_04_2012_08/'

# # April of GMU period
# DATE_RANGE = ('2012_04_01__00_00', '2012_05_01__00_00')
# DATE_DIR = '2012_04/'

# June of GMU period
DATE_RANGE = ('2012_06_01__00_00', '2012_07_01__00_00')
DATE_DIR = '2012_06/'


# # Select few dates for analysis
# DATE_RANGE = None
# DATE_DIR = 'Selected_Maps/'

# # All dates
# DATE_RANGE = ('2000_01_01__00_00', '2020_01_01__00_00')

# # TEST
# DATE_RANGE = ('2012_06_01__00_00', '2012_06_03__00_00')
# DATE_DIR = 'Test/'

# # CR 2136
# DATE_RANGE = ('2013_04_01__00_00', '2013_06_01__00_00')

# # CR 2151
# DATE_RANGE = ('2014_06_01__00_00', '2014_06_30__00_00')

DETECTION_IMAGE_DIR = DETECTION_VERSION_DIR + DATE_DIR
