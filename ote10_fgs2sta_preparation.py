"""Script for the initial step in OTE-10

Authors
-------

    Tony Sohn

Use
---


"""
from __future__ import print_function
import os
import sys
import copy
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt

from astropy.table import Table, vstack, hstack
from astropy import units as u
# from astropy.utils.exceptions import AstropyWarning

import prepare_jwst_fpa_data
import alignment

import pysiaf
import photutils

deg2mas =  u.deg.to(u.mas)

#####################################
plt.close('all')
#####################################

### START OF CONFIGURATION PARAMETERS

idl_tel_method = 'spherical'

home_dir = os.environ['HOME']
data_dir = os.path.join(home_dir,'TEL/OTE-10/FGS-J_alignment')
#base_dir = os.path.join(data_dir,'distortion_calibration')
working_dir = os.path.join(data_dir, 'fgs2sta')
nominalpsf = False # or True

observatory = 'JWST'
prdopssoc_version = 'PRDOPSSOC-031'
#prdopssoc_version = 'PRDOPSSOC-H-015'

use_hawki_catalog = True

use_epsf = False # or False
overwrite_source_extraction = True # or False
###generate_standardized_fpa_data = True # or False
###overwrite_distortion_reference_table = True

save_plot = True # or False
verbose = True # or False
verbose_figures = True # or False
show_extracted_sources = True # or False
show_psfsubtracted_image = True

overwrite_obs_collection = True
overwrite_obs_xmatch_pickle = True

inspect_mode = True # or False
if inspect_mode is False:
    verbose_figures = False

camera_pattern = '_cal.fits'

### END OF CONFIGURATION PARAMETERS

#=============================================================================

def degree_to_mode(polynomial_degree):
    """Convert polynomial degree to mode parameter k.

    Parameters
    ----------
    polynomial_degree : int, float
        Degree of polynomial

    Returns
    -------
    k : int, float
        Mode parameter

    """
    k = 2 * (polynomial_degree + 1)
    return k

#=============================================================================

from jwcf import hawki
reference_catalog = hawki.hawki_catalog()
reference_catalog.rename_column('ra_deg', 'ra')
reference_catalog.rename_column('dec_deg', 'dec')

obs_collection = []
standardized_data_dir = os.path.join(working_dir,'fpa_data')
result_dir = os.path.join(working_dir,'results')

for dir in [working_dir, standardized_data_dir, result_dir]:
    if os.path.isdir(dir) is False:
        os.makedirs(dir)

naming_tag = 'photutils{}'.format(photutils.__version__)
extraction_parameters = {'nominalpsf': nominalpsf,
                         'use_epsf': use_epsf,
                         'show_extracted_sources': show_extracted_sources,
                         'show_psfsubtracted_image': show_psfsubtracted_image,
                         'naming_tag': naming_tag}

im = prepare_jwst_fpa_data.jwst_camera_fpa_data(data_dir, camera_pattern, standardized_data_dir,
                                                parameters=extraction_parameters,
                                                overwrite_source_extraction=overwrite_source_extraction)

plt.close('all')

# Load all siaf apertures
apertures_dict = {}
apertures_dict['instrument'] = ['NIRCAM']*10 + ['FGS']*2 + ['NIRISS']
apertures_dict['pattern'] = ['NRCA1_FULL', 'NRCA2_FULL', 'NRCA3_FULL', 'NRCA4_FULL', 'NRCA5_FULL',
                             'NRCB1_FULL', 'NRCB2_FULL', 'NRCB3_FULL', 'NRCB4_FULL', 'NRCB5_FULL',
                             'FGS1_FULL', 'FGS2_FULL', 'NIS_CEN']
siaf = pysiaf.siaf.get_jwst_apertures(apertures_dict, exact_pattern_match=True)

#siaf_detector_layout = pysiaf.read.read_siaf_detector_layout()
#apertures_dict={'instrument': siaf_detector_layout['InstrName'].data}
#master_aperture_names = siaf_detector_layout['AperName'].data
#apertures_dict['pattern'] = master_aperture_names
#siaf = pysiaf.siaf.get_jwst_apertures(apertures_dict)

# define pickle files
obs_xmatch_pickle_file = os.path.join(result_dir,'observations_xmatch.pkl')
obs_collection_pickle_file = os.path.join(result_dir,'obs_collection.pkl')

crossmatch_dir = os.path.join(standardized_data_dir, 'crossmatch')
if os.path.isdir(crossmatch_dir) is False: os.makedirs(crossmatch_dir)

crossmatch_parameters = {}
crossmatch_parameters['pickle_file'] = obs_xmatch_pickle_file
crossmatch_parameters['overwrite'] = overwrite_obs_xmatch_pickle
crossmatch_parameters['standardized_data_dir'] = standardized_data_dir
crossmatch_parameters['verbose_figures'] = verbose_figures
crossmatch_parameters['save_plot'] = save_plot
crossmatch_parameters['plot_dir'] = crossmatch_dir
crossmatch_parameters['observatory'] = observatory
crossmatch_parameters['correct_reference_for_proper_motion'] = False # or True
crossmatch_parameters['overwrite_pm_correction'] = False # or True
crossmatch_parameters['verbose'] = verbose
crossmatch_parameters['siaf'] = siaf
crossmatch_parameters['idl_tel_method'] = idl_tel_method
crossmatch_parameters['reference_catalog'] = reference_catalog
crossmatch_parameters['xmatch_radius'] = 0.2 * u.arcsec # 0.2 arcsec is about 3 pixels in NIRISS or FGS
crossmatch_parameters['rejection_level_sigma'] = 3 # or 5
crossmatch_parameters['restrict_analysis_to_these_apertures'] = None

# Call the crossmatch routine
observations = prepare_jwst_fpa_data.crossmatch_fpa_data(crossmatch_parameters)

obs = pickle.load(open(obs_xmatch_pickle_file,"rb"))
obs = obs[0]

# Write cross-matched catalog to an ASCII file
t1 = obs.star_catalog_matched['id','x_SCI','y_SCI','mag','v2_spherical_arcsec','v3_spherical_arcsec']
t1.rename_column('v2_spherical_arcsec','v2_obs')
t1.rename_column('v3_spherical_arcsec','v3_obs')

# Transform (x_SCI, y_SCI) to (x_IDL, y_IDL) and add these columns to table
fgs_siaf = pysiaf.Siaf('fgs')
fgs_aperture = fgs_siaf['FGS1_FULL']
x_IDL, y_IDL = fgs_aperture.tel_to_idl(t1['v2_obs'].data, t1['v3_obs'].data)
t1.add_columns([x_IDL, y_IDL], names=['x_IDL', 'y_IDL'])

t2 = obs.reference_catalog_matched['ID','ra','dec','fgs1_magnitude','v2_spherical_arcsec','v3_spherical_arcsec','d_xmatch_mas']
t2.rename_column('v2_spherical_arcsec','v2_ref')
t2.rename_column('v3_spherical_arcsec','v3_ref')

t_xmatch = hstack([t1, t2])
t_xmatch.remove_rows(np.where(np.isnan(t_xmatch['d_xmatch_mas']))) # remove all NaNs in match distance
t_xmatch.remove_rows(np.where(t_xmatch['fgs1_magnitude'] > 19))    # only keep the bright sources

xmatch_ascii = os.path.join(result_dir,'cross_matched.csv')
t_xmatch.write(xmatch_ascii, format='ascii.fixed_width', comment=False, delimiter=',', bookend=False, overwrite=True)

xyrd_ascii = os.path.join(result_dir,'xyrd.txt')
t_xyrd = t_xmatch['x_IDL', 'y_IDL', 'ra', 'dec']
t_xyrd.write(xyrd_ascii, format='ascii.fixed_width', comment=False, delimiter='   ', bookend=False, overwrite=True)

sys.exit(0)
