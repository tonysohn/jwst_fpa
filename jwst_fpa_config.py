"""Configuration parameters for focal plane alignment.
"""

import os
import numpy as np

home_dir = os.environ['HOME']
local_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(home_dir, 'JWST/Flight/OTE-10/FGS1-NIRCam/NRCB5/')
##################################################################
apertures_to_calibrate                         = ['NRCB5_FULL']
##
alignment_reference_apertures                  = ['FGS1_FULL']
attitude_defining_apertures                    = ['FGS1_FULL']
calibration_alignment_reference_aperture_names = ['FGS1_FULL']
calibration_attitude_defining_aperture_names   = ['FGS1_FULL']
apply_fpa_calibration_array                    = [False]
##################################################################

correct_dva = False
#sc_velocity = [-13.413720, -25.057451, -10.677287]

nominalpsf = True
reference_catalog_type = 'hawki'
distortion_coefficients_file = None
#distortion_coefficients_file = 'distortion_coeffs_nis_cen_jw01086001001_01101_00021_nis_cal.txt'


save_plot = True
verbose = True
verbose_figures = True
visit_groups = [0] ### TBD: This is probably used for HST multi-epoch analysis only

# Local distortion fit in v2, v3
k = 8 # or 6 or 4

# Global distortion fit for attitude determination
k_attitude_determination = 4

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

# Rotation parameter that will be minimized during the iterative aperture correction
rotation_name = 'Rotation in Y' # or 'Global Rotation'

# Use 'spherical' below unless testing relative uncertainty for 'planar_approximation' (the other option).
idl_tel_method = 'spherical'

use_tel_boresight = False

restrict_analysis_to_these_apertures = None
restrict_to_sets_that_include_aperture_names = [None] * len(alignment_reference_apertures)

applied_calibration_file = None
write_calibration_result_file = True # or True # or False

# directory containing DVA correction source code and executable `compute-DVA.e`
## dva_source_dir = os.path.join(home_dir, 'jwst/tel/hst/focal_plane_calibration/DVA')
##dva_source_dir = os.path.join(local_dir, 'hst_dva_code')
