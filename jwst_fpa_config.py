"""Configuration parameters for focal plane alignment.

Authors
-------
    - Johannes Sahlmann
"""
import os
import numpy as np

observatory = 'JWST'

##username = os.getlogin()

home_dir = os.environ['HOME']
local_dir = os.path.dirname(os.path.abspath(__file__))

# Data directory (location of all _cal.fits files)
#data_dir = os.path.join(home_dir, 'TEL/OTE-10/FGS1-FGS2_alignment/FGS1-NRCA3')
#data_dir = os.path.join(home_dir, 'TEL/OTE-10/FGS1-FGS2_alignment/NRCA3-FGS2')
data_dir = os.path.join(home_dir, 'TEL/OTE-11/FGS1-NIRISS_alignment')
#data_dir = os.path.join(home_dir, 'TEL/OTE-11/NIRISS-FGS2_alignment')
#data_dir = os.path.join(home_dir, 'TEL/OTE-11/Confirmation')

nominalpsf = False

##################################################################

if 1:
    alignment_reference_apertures = ['FGS1_FULL']
    attitude_defining_apertures = ['FGS1_FULL']
    calibration_alignment_reference_aperture_names = ['FGS1_FULL']
    calibration_attitude_defining_aperture_names = ['FGS1_FULL']
    ##### IMPORTANT: Below will be used when I want to APPLY the fpa result to a given alignment result.
    ##### For example, if I want to do FGS1-NIRISS, and then NIRISS-FGS2 for aligning FGS1-FGS2, I would
    ##### set below True for the FGS1-NIRISS alignment, so by the time the script does NIRISS-FGS2,
    ##### NIRISS would have been already calibrated to have its zero point shifted based on the FGS1-NIRISS alignment.
    apply_fpa_calibration_array = [False]
    apertures_to_calibrate = ['NIS_CEN']

##################################################################
### Below would be for the FGS1-NIRISS-FGS2 alignment
if 0:
    alignment_reference_apertures                  = ['FGS1_FULL', 'NIS_CEN']
    attitude_defining_apertures                    = ['FGS1_FULL', 'NIS_CEN']
    calibration_alignment_reference_aperture_names = ['FGS1_FULL', 'NIS_CEN']
    calibration_attitude_defining_aperture_names   = ['FGS1_FULL', 'NIS_CEN']
    apply_fpa_calibration_array                    = [True, False]
    apertures_to_calibrate                         = ['NIS_CEN', 'FGS2_FULL']

##################################################################

if 0:
    alignment_reference_apertures = ['FGS1_FULL']
    attitude_defining_apertures = ['FGS1_FULL']
    calibration_alignment_reference_aperture_names = ['FGS1_FULL']
    calibration_attitude_defining_aperture_names = ['FGS1_FULL']
    apply_fpa_calibration_array = [False]
    apertures_to_calibrate = ['NRCA3_FULL']

if 0:
    alignment_reference_apertures = ['NRCA3_FULL']
    attitude_defining_apertures = ['NRCA3_FULL']
    calibration_alignment_reference_aperture_names = ['NRCA3_FULL']
    calibration_attitude_defining_aperture_names = ['NRCA3_FULL']
    apply_fpa_calibration_array = [False]
    apertures_to_calibrate = ['FGS2_FULL']

#
##overwrite = 0
###analysis_name = 'spherical' # or 'planar'
save_plot = True
verbose = True
verbose_figures = True
#visit_groups = [[0, 1, 2], [3, 4, 5]]
visit_groups = [0] ### TBD: Figure out what this parameter is used for

# Local distortion fit in v2, v3
k = 8 # or 6 or 4

# Global distortion fit for attitude determination
k_attitude_determination = 4 # or 8

generate_standardized_fpa_data     = True
overwrite_source_extraction        = False
overwrite_obs_xmatch_pickle        = False
overwrite_obs_collection           = True
overwrite_attitude_pickle          = True
overwrite_alignment_results_pickle = True
overwrite_alignment_results_pickle = True

###overwrite_siaf_pickle = False

make_summary_plots = True # or False
show_summary_results = True

### NOTE: below are for calibrations as a function of time, so no need for commissioning.
show_camera_evolution = False
show_attitude_evolution = False

# This is used for Monte-Carlo type simulations, but just keep it to [0] for direct solutions.
random_realisations = [0] # for e.g., 9 random simulations, set this to "np.arange(9)"

correct_dva = False # or True: Applies only to HST camera apertures, FGS data is already DVA corrected

# Rotation parameter that will be minimized during the iterative aperture correction
rotation_name = 'Rotation in Y' # or 'Global Rotation'

# Use 'spherical' below unless testing relative uncertainty for 'planar_approximation' (the other option).
idl_tel_method = 'spherical'

use_tel_boresight = False

# Below are various options on what type of calibrations will be performed.


# Below is for ultimately doing FGS1-FGS2 alignement. As an intermediary step, use NRCA3_FULL as reference.
if 0:
    alignment_reference_apertures = ['NRCA3_FULL']
    attitude_defining_apertures = ['NRCA3_FULL']
    calibration_alignment_reference_aperture_names = ['NRCA3_FULL']
    calibration_attitude_defining_aperture_names = ['NRCA3_FULL']
    apply_fpa_calibration_array = [False]
    apertures_to_calibrate = ['FGS2_FULL']

if 0:
    alignment_reference_apertures = ['FGS2_FULL']
    attitude_defining_apertures = ['FGS2_FULL']
    calibration_alignment_reference_aperture_names = ['FGS2_FULL']
    calibration_attitude_defining_aperture_names = ['FGS2_FULL']
    apply_fpa_calibration_array = [False]
    apertures_to_calibrate = ['NRCA3_FULL']


# OTE-10 Observation 4 would look like this
if 0:
    alignment_reference_apertures = ['FGS1_FULL']
    attitude_defining_apertures = ['FGS1_FULL']
    calibration_alignment_reference_aperture_names = ['FGS1_FULL']
    calibration_attitude_defining_aperture_names = ['FGS1_FULL']
    apply_fpa_calibration_array = [False]
    apertures_to_calibrate = ['NRCA1_FULL','NRCA2_FULL','NRCA3_FULL','NRCA4_FULL','NRCA5_FULL',
                              'NRCB1_FULL','NRCB2_FULL','NRCB3_FULL','NRCB4_FULL','NRCB5_FULL',]

# OTE-10 Observation 5 would look like this
if 0:
    alignment_reference_apertures = ['FGS2_FULL']
    attitude_defining_apertures = ['FGS2_FULL']
    calibration_alignment_reference_aperture_names = ['FGS2_FULL']
    calibration_attitude_defining_aperture_names = ['FGS2_FULL']
    apply_fpa_calibration_array = [False]
    apertures_to_calibrate = ['NRCA1_FULL','NRCA2_FULL','NRCA3_FULL','NRCA4_FULL','NRCA5_FULL',
                              'NRCB1_FULL','NRCB2_FULL','NRCB3_FULL','NRCB4_FULL','NRCB5_FULL',]

### Below are left here to see what options are available.
#if 0:
#    alignment_reference_apertures = ['FGS1_FULL', 'NIS_CEN']
#    attitude_defining_apertures = ['FGS1_FULL', 'NIS_CEN']
#    calibration_alignment_reference_aperture_names = [None, 'FGS1_FULL']
#    calibration_attitude_defining_aperture_names = [None, 'FGS1_FULL']
#    apply_fpa_calibration_array = [False, True]
#    apertures_to_calibrate = ['NIS_CEN']
#
# to perform calibrations
#if 0:
#    alignment_reference_apertures = ['FGS2_FULL']
#    attitude_defining_apertures = ['FGS2_FULL']
#    calibration_alignment_reference_aperture_names = [None]
#    calibration_attitude_defining_aperture_names = [None]
#    apply_fpa_calibration_array = [False]
#    apertures_to_calibrate = ['NIS_CEN']
#
# to perform calibrations
#if 0:
#    alignment_reference_apertures = ['NIS_CEN']
#    attitude_defining_apertures = ['NIS_CEN']
#    calibration_alignment_reference_aperture_names = [None]
#    calibration_attitude_defining_aperture_names = [None]
#    apply_fpa_calibration_array = [False]
#    apertures_to_calibrate = ['NIS_CEN']

restrict_analysis_to_these_apertures = None
restrict_to_sets_that_include_aperture_names = [None] * len(alignment_reference_apertures)

applied_calibration_file = None
write_calibration_result_file = True # or True # or False

# directory containing DVA correction source code and executable `compute-DVA.e`
## dva_source_dir = os.path.join(home_dir, 'jwst/tel/hst/focal_plane_calibration/DVA')
##dva_source_dir = os.path.join(local_dir, 'hst_dva_code')
