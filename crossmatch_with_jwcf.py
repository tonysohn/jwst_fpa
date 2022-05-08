"""Script for crossmatching sources in JWST images with Reference Catalog

Authors
-------

    Tony Sohn

Use
---


"""
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

import pysiaf
from astropy.table import hstack
from astropy import units as u
from jwcf import hawki, hst

from jwst import datamodels
import prepare_jwst_fpa_data, alignment


def crossmatch(file_path, reference_catalog_type='hawki', #'hst'
               save_plot=True, verbose_figures=False,
               nominalpsf=True, use_epsf=False):

    """
    This script runs the extraction and crossmatching codes on a single JWST image
    """

    plt.close('all')

    if not os.path.isfile(file_path): sys.exit("File not found.")

    im = datamodels.open(file_path)
    apername = getattr(im.meta.aperture, 'name')
    instname = getattr(im.meta.instrument, 'name')
    detector = getattr(im.meta.instrument, 'detector')
    filter   = getattr(im.meta.instrument, 'filter')
    pupil    = getattr(im.meta.instrument, 'pupil')

    full_path = os.path.split(file_path)
    if not full_path[0]:
        data_dir = "./" # if no path was given, use current dir
    else:
        data_dir = full_path[0]
    fname = full_path[1]
    fname_head = os.path.splitext(fname)[0]

    # Dump everything to the directory where the image is for this script
    standardized_data_dir = data_dir
    result_dir = data_dir

    if reference_catalog_type.lower() == 'hawki':
        reference_catalog = hawki.hawki_catalog()
        reference_catalog.rename_column('ra_deg', 'ra')
        reference_catalog.rename_column('dec_deg', 'dec')
        reference_catalog['j_magnitude'] = reference_catalog['j_2mass_extrapolated']
    elif reference_catalog_type.lower() == 'hst':
        reference_catalog = hst.hst_catalog(decimal_year_of_observation=2022.0) # Set the epoch to 2022.0
        reference_catalog.rename_column('ra_deg', 'ra')
        reference_catalog.rename_column('dec_deg', 'dec')
        reference_catalog['j_magnitude'] = reference_catalog['j_mag_vega']
    else:
        sys.exit('Unsupported Reference Catalog. Only HawkI and HST catalogs are currently supported.')

    extraction_parameters = {}
    extraction_parameters['nominalpsf'] = True
    extraction_parameters['use_centroid_2dg'] = False
    extraction_parameters['use_epsf'] = use_epsf
    extraction_parameters['show_extracted_sources'] = verbose_figures
    extraction_parameters['show_psfsubtracted_image'] = verbose_figures
    extraction_parameters['save_plot'] = save_plot

    im = prepare_jwst_fpa_data.jwst_camera_fpa_data(data_dir, fname,
                                                    standardized_data_dir,
                                                    parameters=extraction_parameters,
                                                    overwrite_source_extraction=True)

    plt.close('all')

    # Load all siaf apertures
    apertures_dict = {}
    apertures_dict['instrument'] = ['NIRCAM']*11 + ['FGS']*2 + ['NIRISS'] + ['MIRI'] + ['NIRSpec']*2
    apertures_dict['pattern'] = ['NRCA1_FULL', 'NRCA2_FULL', 'NRCA3_FULL', 'NRCA4_FULL', 'NRCA5_FULL',
                             'NRCB1_FULL', 'NRCB2_FULL', 'NRCB3_FULL', 'NRCB4_FULL', 'NRCB5_FULL',
                             'NRCA5_FULL_MASKLWB',
                             'FGS1_FULL', 'FGS2_FULL', 'NIS_CEN', 'MIRIM_FULL', 'NRS1_FULL', 'NRS2_FULL']


    if apername not in apertures_dict['pattern']:
        sys.exit("Aperture not supported.")

    siaf = pysiaf.siaf.get_jwst_apertures(apertures_dict, exact_pattern_match=True)

    # define pickle files
    obs_xmatch_pickle_file = os.path.join(result_dir, fname_head+'_xmatch.pkl')
    fpa_file_name = '{}_FPA_data.fits'.format(fname_head)

    crossmatch_parameters = {}
    crossmatch_parameters['pickle_file'] = obs_xmatch_pickle_file
    crossmatch_parameters['overwrite'] = True
    crossmatch_parameters['standardized_data_dir'] = standardized_data_dir
    crossmatch_parameters['verbose_figures'] = False
    crossmatch_parameters['save_plot'] = True
    crossmatch_parameters['data_dir'] = data_dir
    crossmatch_parameters['plot_dir'] = standardized_data_dir
    crossmatch_parameters['correct_reference_for_proper_motion'] = False # or True
    crossmatch_parameters['overwrite_pm_correction'] = False # or True
    crossmatch_parameters['verbose'] = True
    crossmatch_parameters['siaf'] = siaf
    crossmatch_parameters['idl_tel_method'] = 'spherical'
    crossmatch_parameters['reference_catalog'] = reference_catalog
    crossmatch_parameters['xmatch_radius'] = 0.4 * u.arcsec # 0.2 arcsec is about 3 pixels in NIRISS or FGS
    crossmatch_parameters['rejection_level_sigma'] = 4 # or 4 or 5?
    crossmatch_parameters['restrict_analysis_to_these_apertures'] = None
    crossmatch_parameters['use_default_siaf_distortion'] = False
    crossmatch_parameters['fpa_file_name'] = fpa_file_name # This ensures multiple FPA_data files are processed
    crossmatch_parameters['correct_dva'] = False
    crossmatch_parameters['sigma_crossmatch'] = 4.0
    crossmatch_parameters['sigma_fitting'] = 3.0
    crossmatch_parameters['xmatch_refcat_mag_range'] = [14, 20.5]

    # Call the crossmatch routine
    observations = prepare_jwst_fpa_data.crossmatch_fpa_data(crossmatch_parameters)

    # Output the obs_collection pickle file
    obs_collection = alignment.AlignmentObservationCollection(observations)
    obs_collection_file = os.path.join(result_dir, fname_head+'_obs_collection.pkl')
    pickle.dump(obs_collection, open(obs_collection_file, 'wb'))

    obs = pickle.load(open(obs_xmatch_pickle_file, "rb"))
    obs = obs[0]

    # Write cross-matched catalog to an ASCII file
    # First deal with the cross-matched sources in the observed catalog
    t1 = obs.star_catalog_matched['id','x_SCI','y_SCI','mag','v2_spherical_arcsec','v3_spherical_arcsec']
    t1.rename_column('x_SCI','x_sci')
    t1.rename_column('y_SCI','y_sci')
    t1.rename_column('v2_spherical_arcsec','v2_obs')
    t1.rename_column('v3_spherical_arcsec','v3_obs')

    # Transform (x_SCI, y_SCI) to (x_IDL, y_IDL) and add these columns to table
    aper = siaf[apername]
    x_idl, y_idl = aper.tel_to_idl(t1['v2_obs'].data, t1['v3_obs'].data)
    t1.add_columns([x_idl, y_idl], names=['x_idl', 'y_idl'])

    # Now deal with the cross-matched sources in the reference catalog

    # Figure out which instrument+magnitude to output in cross-matched catalog
    if 'FGS' in instname:
        mhead = instname.lower() + detector[-1]
    else:
        mhead = '{}_{}'.format(instname.lower(), filter+pupil.replace('CLEAR',''))
    mname = '{}_magnitude'.format(mhead)
    mname_short = mname.replace('magnitude','mag')

    if reference_catalog_type.lower() == 'hawki':
        t2 = obs.reference_catalog_matched['ID','ra','dec',mname,'v2_spherical_arcsec','v3_spherical_arcsec','d_xmatch_mas']
    else:
        t2 = obs.reference_catalog_matched['ID','ra','dec',mname,'v2_spherical_arcsec','v3_spherical_arcsec']

    t2.rename_column('ID', 'id_ref')
    t2.rename_column('v2_spherical_arcsec','v2_ref')
    t2.rename_column('v3_spherical_arcsec','v3_ref')
    t2.rename_column(mname, mname_short)

    # Paste the two tables (horizontally)
    t_xmatch = hstack([t1, t2])
    # Remove all NaNs in match distance
    if reference_catalog_type.lower() == 'hawki':
        t_xmatch.remove_rows(np.where(np.isnan(t_xmatch['d_xmatch_mas'])))
    # Round columns to the specified number of decimals

    if reference_catalog_type.lower() == 'hawki':
        t_xmatch.round({'x_sci':4, 'y_sci':4, 'mag':4, 'v2_obs':4, 'v3_obs':4,
                        'x_idl':4, 'y_idl':4, 'ra':9, 'dec':9, mname_short:4,
                        'v2_ref':4, 'v3_ref':4, 'd_xmatch_mas':6})
    else:
        t_xmatch.round({'x_sci':4, 'y_sci':4, 'mag':4, 'v2_obs':4, 'v3_obs':4,
                        'x_idl':4, 'y_idl':4, 'ra':9, 'dec':9, mname_short:4,
                        'v2_ref':4, 'v3_ref':4})

    outcsvfile = '{}_xmatch.csv'.format(fname_head)
    xmatch_ascii = os.path.join(result_dir, outcsvfile)
    t_xmatch.write(xmatch_ascii, format='ascii.fixed_width', comment=False, delimiter=',', bookend=False, overwrite=True)

    #xyrd_ascii = os.path.join(result_dir,'xyrd.txt')
    #t_xyrd = t_xmatch['x_idl', 'y_idl', 'ra', 'dec']
    #t_xyrd.write(xyrd_ascii, format='ascii.fixed_width', comment=False, delimiter='   ', bookend=False, overwrite=True)

    return
