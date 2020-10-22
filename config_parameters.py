"""Configuration parameters for focal plane alignment.

Authors
-------
    - Johannes Sahlmann
"""
import os
import stat

from astropy import units as u
from astropy.time import Time
import numpy as np

local_dir = os.path.dirname(os.path.abspath(__file__))
home_dir = os.environ['HOME']
username = os.getlogin()


# base directory for writing results
base_dir = os.path.join(home_dir, 'jwst/tel/hst/focal_plane_calibration')

# directory containing DVA correction source code and executable `compute-DVA.e`
# dva_source_dir = os.path.join(home_dir, 'jwst/tel/hst/focal_plane_calibration/DVA')
dva_source_dir = os.path.join(local_dir, 'hst_dva_code')

# directory containing the HST camera data
hst_camera_root_data_dir = '/grp/hst/OTA/focal_plane_alignment'

# directory containing the reduced HST FGS data
# hst_fgs_root_data_dir = os.path.join(base_dir, 'fgs_reduced_data')
hst_fgs_root_data_dir = '/grp/hst/OTA/focal_plane_alignment/fgs_reduced_data'