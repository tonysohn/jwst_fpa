"""Central Script that calibrates the JWST Focal Plane Alignment.

Authors
-------

    Tony Sohn
    Original script by Johannes Sahlmann

Major Modifications
-------------------

Use
---
    From ipython session:
    > run jwst_fpa.py

To be done
----------
    - Script does not work correctly when having multiple inputs for apertures_to_calibrate.
      Check on how the results are being stored for each aperture.
    - Add an option to change simple parameters such as detection threshold.

"""

from __future__ import print_function

import os
import sys
import copy
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.time import Time
from astropy.table import Table, vstack

import pysiaf
from jwcf import hawki, hst

import prepare_jwst_fpa_data
import alignment

# Loding in configurations from config file
import jwst_fpa_config
import importlib
importlib.reload(jwst_fpa_config) # reload config file in case changes are made while in session

home_dir = jwst_fpa_config.home_dir
local_dir = jwst_fpa_config.local_dir
data_dir = jwst_fpa_config.data_dir

reference_catalog_type = jwst_fpa_config.reference_catalog_type

save_plot = jwst_fpa_config.save_plot
verbose = jwst_fpa_config.verbose
verbose_figures = jwst_fpa_config.verbose_figures
visit_groups = jwst_fpa_config.visit_groups
nominalpsf = jwst_fpa_config.nominalpsf
#
# Jun 03, 2021: For now, below is only used to calculate V3SciXAngle.
# TBD: Make changes to prepare_jwst_fpa_data.py so that it actually uses the
#      input distortion coefficient file for both FGS and SI.
#
distortion_coefficients_file = jwst_fpa_config.distortion_coefficients_file

k = jwst_fpa_config.k
k_attitude_determination = jwst_fpa_config.k_attitude_determination

generate_standardized_fpa_data = jwst_fpa_config.generate_standardized_fpa_data
overwrite_source_extraction = jwst_fpa_config.overwrite_source_extraction
overwrite_obs_xmatch_pickle = jwst_fpa_config.overwrite_obs_xmatch_pickle
overwrite_obs_collection = jwst_fpa_config.overwrite_obs_collection
overwrite_attitude_pickle = jwst_fpa_config.overwrite_attitude_pickle
overwrite_alignment_results_pickle = jwst_fpa_config.overwrite_alignment_results_pickle

make_summary_plots = jwst_fpa_config.make_summary_plots
show_camera_evolution = jwst_fpa_config.show_camera_evolution
show_attitude_evolution = jwst_fpa_config.show_attitude_evolution
correct_dva = jwst_fpa_config.correct_dva
show_summary_results = jwst_fpa_config.show_summary_results

rotation_name = jwst_fpa_config.rotation_name
idl_tel_method = jwst_fpa_config.idl_tel_method
use_tel_boresight = jwst_fpa_config.use_tel_boresight

alignment_reference_apertures = jwst_fpa_config.alignment_reference_apertures
attitude_defining_apertures = jwst_fpa_config.attitude_defining_apertures
calibration_alignment_reference_aperture_names = jwst_fpa_config.calibration_alignment_reference_aperture_names
calibration_attitude_defining_aperture_names = jwst_fpa_config.calibration_attitude_defining_aperture_names
apply_fpa_calibration_array = jwst_fpa_config.apply_fpa_calibration_array
apertures_to_calibrate = jwst_fpa_config.apertures_to_calibrate

restrict_analysis_to_these_apertures = jwst_fpa_config.restrict_analysis_to_these_apertures
restrict_to_sets_that_include_aperture_names = jwst_fpa_config.restrict_to_sets_that_include_aperture_names

applied_calibration_file = jwst_fpa_config.applied_calibration_file
write_calibration_result_file = jwst_fpa_config.write_calibration_result_file

#=============================================================================
# START OF MAIN PART

if reference_catalog_type.lower() == 'hawki':
    reference_catalog = hawki.hawki_catalog()
    reference_catalog.rename_column('ra_deg', 'ra')
    reference_catalog.rename_column('dec_deg', 'dec')
    reference_catalog.rename_column('j_2mass_extrapolated', 'j_magnitude')
elif reference_catalog_type.lower() == 'hst':
    reference_catalog = hst.hst_catalog(decimal_year_of_observation=2022.0)
    reference_catalog.rename_column('ra_deg', 'ra')
    reference_catalog.rename_column('dec_deg', 'dec')
else:
    sys.exit('Unsupported Reference Catalog. Only HawkI and HST catalogs are currently supported.')

print('{}\nFOCAL PLANE ALIGNMENT CALIBRATION'.format('='*100))

obs_collection = []

working_dir = os.path.join(data_dir,'focal_plane_calibration')
standardized_data_dir = os.path.join(working_dir, 'fpa_data')
result_dir = os.path.join(working_dir, 'results')
plot_dir = os.path.join(working_dir, 'plots')
##dva_dir = os.path.join(working_dir, 'dva_data')

#for dir in [standardized_data_dir, plot_dir, result_dir, out_dir, dva_dir]:
for dir in [standardized_data_dir, result_dir, plot_dir]:
    if os.path.isdir(dir) is False:
        os.makedirs(dir)

# define pickle files
obs_xmatch_pickle_file = os.path.join(result_dir, 'obs_xmatch.pkl')
obs_collection_pickle_file = os.path.join(result_dir, 'obs_collection.pkl')

use_weights_for_epsf = False # or True
test_run_on_single_niriss_file = False # or False
test_run_on_single_fgs_file = False # or Trie

camera_pattern = '_cal.fits'

# Load all relevant SIAF apertures
apertures_dict = {}
apertures_dict['instrument'] = ['NIRCAM']*10 + ['FGS']*2 + ['NIRISS'] + ['MIRI']
apertures_dict['pattern'] = ['NRCA1_FULL', 'NRCA2_FULL', 'NRCA3_FULL', 'NRCA4_FULL', 'NRCA5_FULL',
                             'NRCB1_FULL', 'NRCB2_FULL', 'NRCB3_FULL', 'NRCB4_FULL', 'NRCB5_FULL',
                             'FGS1_FULL', 'FGS2_FULL', 'NIS_CEN', 'MIRIM_FULL']
siaf = pysiaf.siaf.get_jwst_apertures(apertures_dict, exact_pattern_match=True)

if (generate_standardized_fpa_data) or (not glob.glob(os.path.join(standardized_data_dir, '*.fits'))):

    extraction_parameters = {'nominalpsf': nominalpsf,
                             'use_epsf': False, # change to True later
                             'show_extracted_sources': True,
                             'show_psfsubtracted_image': True,
                             #'naming_tag':  naming_tag
                             #'epsf_psf_size_pix': 20,
                             #'use_DAOStarFinder_for_epsf' : True,
                             #'use_weights_for_epsf': use_weights_for_epsf,
                             #'flux_threshold_percentile_lower': 20,
                             #'flux_threshold_percentile_upper': 95,
                             #'dao_detection_threshold': 5.,
                             #'final_extraction_niters': 5,
                             #'detection_fwhm': 1.5,
                             #'discard_stars_based_on_dq': 30 # threshold number of pixels with DQ!=0 within a cutout
                             }

    im = prepare_jwst_fpa_data.jwst_camera_fpa_data(data_dir, camera_pattern, standardized_data_dir,
                                                    parameters=extraction_parameters,
                                                    overwrite_source_extraction=overwrite_source_extraction)

for iii, alignment_reference_aperture_name in enumerate(alignment_reference_apertures):

    attitude_defining_aperture_name = attitude_defining_apertures[iii]
    calibration_alignment_reference_aperture_name = calibration_alignment_reference_aperture_names[iii]
    calibration_attitude_defining_aperture_name = calibration_attitude_defining_aperture_names[iii]
    restrict_to_sets_that_include_aperture = restrict_to_sets_that_include_aperture_names[iii]
    apply_fpa_calibration = apply_fpa_calibration_array[iii]

    plt.close('all')

    original_siaf = copy.deepcopy(siaf)
    for aperture_name, aperture in siaf.apertures.items():
        for attribute in 'V2Ref V3Ref'.split():
            setattr(siaf[aperture_name], '{}_original'.format(attribute), getattr(siaf[aperture_name], attribute))

    crossmatch_dir = os.path.join(standardized_data_dir, 'crossmatch')
    if os.path.isdir(crossmatch_dir) is False: os.makedirs(crossmatch_dir)

    if (not os.path.isfile(obs_collection_pickle_file)) | (overwrite_obs_collection):

        # crossmatch the stars in every aperture with the reference catalog (here Hawk-I)
        crossmatch_parameters = {}
        crossmatch_parameters['pickle_file'] = obs_xmatch_pickle_file
        crossmatch_parameters['overwrite'] = overwrite_obs_xmatch_pickle
        crossmatch_parameters['data_dir'] = data_dir
        crossmatch_parameters['standardized_data_dir'] = standardized_data_dir
        crossmatch_parameters['verbose_figures'] = verbose_figures
        crossmatch_parameters['save_plot'] = save_plot
        crossmatch_parameters['plot_dir'] = crossmatch_dir
        crossmatch_parameters['correct_reference_for_proper_motion'] = False # or True
        crossmatch_parameters['overwrite_pm_correction'] = False
        crossmatch_parameters['verbose'] = verbose
        crossmatch_parameters['siaf'] = siaf
        crossmatch_parameters['idl_tel_method'] = idl_tel_method
        crossmatch_parameters['reference_catalog'] = reference_catalog
        crossmatch_parameters['xmatch_radius'] = 0.2 * u.arcsec
        crossmatch_parameters['rejection_level_sigma'] = 3
        crossmatch_parameters['restrict_analysis_to_these_apertures'] = None
        crossmatch_parameters['distortion_coefficients_file'] = distortion_coefficients_file
#        crossmatch_parameters['camera_names'] = ['NIRISS','FGS1','FGS2']
#        crossmatch_parameters['xmatch_radius_camera'] = 0.5 * u.arcsec
#        crossmatch_parameters['xmatch_radius_fgs'] = None

        # create observations class
        observations = prepare_jwst_fpa_data.crossmatch_fpa_data(crossmatch_parameters)

        # generate an AlignmentObservationCollection object
        obs_collection = alignment.AlignmentObservationCollection(observations)

        # [STS] Below is required to avoid error
        obs_collection.group_by('obs_id')
        # unique observation index for cross-identification
        # obs_collection.generate_attitude_groups()
        obs_collection.T['attitude_id'] = obs_collection.T['group_id']
        obs_collection.T['attitude_group'] = obs_collection.T['group_id']
        obs_collection.T['align_params'] = 'default'

        obs_collection.T['INDEX'] = np.arange(len(obs_collection.T))

        # correct for DVA -- TBD: Should add a dva correction routine
        if correct_dva:
            correct_dva_parameters = {}
            correct_dva_parameters['dva_dir'] = dva_dir
            correct_dva_parameters['dva_source_dir'] = dva_source_dir
            correct_dva_parameters['verbose'] = False
            obs_collection = prepare_jwst_fpa_data.correct_dva(obs_collection, correct_dva_parameters)

        pickle.dump(obs_collection, open(obs_collection_pickle_file, "wb"))

    else:
        obs_collection = pickle.load(open(obs_collection_pickle_file, "rb"))
        print('Loaded pickled file {}'.format(obs_collection_pickle_file))

    obs_collection.assign_alignment_reference_aperture(alignment_reference_aperture_name)

    # observation data table for report
    if 1:
        report_table = copy.deepcopy(obs_collection.T)
        report_table.sort('DATAFILE')
        # keys_to_print = 'proposal_id visit instrument start_time number_of_files exptime duration filter'.split()
        keys_to_print = 'DATAFILE instrument_name SIAFAPER instrument_filter instrument_pupil'.split()
        report_table[keys_to_print].pprint()
        report_table[keys_to_print].write(sys.stdout, format='latex', formats={'DATAFILE': lambda s: s.replace('_','\_'), 'SIAFAPER': lambda s: s.replace('_','\_')})

        # 1/0

    # PERFORM FOCAL PLANE ALIGNMENT MEASUREMENT
    attitude_determination_parameters = {
        'attitude_defining_aperture_name': attitude_defining_aperture_name,
        'maximum_number_of_iterations': 50,
        'attenuation_factor': 0.9,
        'fractional_threshold_for_iterations': 0.1,
        'verbose': True, # or False
        'k': k,
        'k_attitude_determination': k_attitude_determination,
        'reference_frame_number': 0,  # reference frame, evaluation_frame_number is calibrated
        'evaluation_frame_number': 1,  # frame to calibrate
        'rotation_name': rotation_name,
        'use_v1_pointing': True,
        'save_plot': False, # or True
        'out_dir': '',
        'name_seed': 'attitude_error',
        'eliminate_omc_outliers_iteratively': True, # or False - used for final attitude determination fit
        'outlier_rejection_level_sigma': 2, # or 3 for stricter rejection. Used for final attitude determination fit
        'plot_dir': plot_dir,
        'show_final_fit': True, # or False
        'idl_tel_method': idl_tel_method,
        'use_tel_boresight': use_tel_boresight,
        'plot_residuals': False,
        'use_fgs_pseudo_aperture': False,
        'reference_point_setting': 'auto', # use v2ref,v3ref as origin for distortion fit when applicable
        ##'use_hst_fgs_fiducial_as_reference_point': use_hst_fgs_fiducial_as_reference_point, # allows to override above selection for HST FGS, to mitigate issues with scale error in FGS3 and correlation with v2,v3 offsets
        'perform_temporary_alignment_update': True,  # DEFAULT = True, if True, the apertures used for attitude determination are aligned iteratively
        'perform_temporary_distortion_correction': True,
        'skip_temporary_alignment_for_aligned_apertures': apply_fpa_calibration,
        'apertures_to_calibrate': apertures_to_calibrate,

        # 'reference_point': np.array([[0., 0.], [0., 0.]]),  # reference point for the differential coordinates
        # 'reference_point_setting': np.array([[0., 0.], [0., 0.]])
    }


    fpa_parameters = {}
    fpa_parameters['alignment_reference_aperture_name'] = alignment_reference_aperture_name
    fpa_parameters['attitude_defining_aperture_name'] = attitude_defining_aperture_name
    fpa_parameters['restrict_to_sets_that_include_aperture'] = restrict_to_sets_that_include_aperture
    fpa_parameters['calibration_alignment_reference_aperture_name'] = calibration_alignment_reference_aperture_name
    fpa_parameters['calibration_attitude_defining_aperture_name'] = calibration_attitude_defining_aperture_name
    fpa_parameters['correct_dva'] = correct_dva
    fpa_parameters['k'] = k
    fpa_parameters['k_attitude_determination'] = k_attitude_determination
    fpa_parameters['result_dir'] = result_dir
    fpa_parameters['original_siaf'] = original_siaf
    fpa_parameters['overwrite'] = overwrite_alignment_results_pickle
    fpa_parameters['idl_tel_method'] = idl_tel_method
    fpa_parameters['use_tel_boresight'] = use_tel_boresight
    fpa_parameters['plot_dir'] = plot_dir
    fpa_parameters['make_summary_plots'] = make_summary_plots
    fpa_parameters['overwrite_attitude_pickle'] = overwrite_attitude_pickle
    fpa_parameters['rotation_name'] = rotation_name
    fpa_parameters['write_calibration_result_file'] = write_calibration_result_file
    fpa_parameters['attitude_determination_parameters'] = attitude_determination_parameters
    fpa_parameters['visit_groups'] = visit_groups
    fpa_parameters['calibration_field_selection'] = 'calibrated'
    fpa_parameters['use_fgs_pseudo_aperture'] = False
    fpa_parameters['plot_residuals'] = True  # generate figures showing the fit residuals after an aperture's alignment
    fpa_parameters['verbose'] = verbose

    if restrict_analysis_to_these_apertures is not None:
        remove_index = np.array([j for j, name in enumerate(obs_collection.T['AperName']) if name not in restrict_analysis_to_these_apertures])
        obs_collection.delete_observations(remove_index)

    if restrict_analysis_to_these_apertures is not None:
        apertures_to_calibrate = [name for name in apertures_to_calibrate if name in restrict_analysis_to_these_apertures]

    fpa_parameters['apertures_to_calibrate'] = apertures_to_calibrate
    fpa_parameters['apply_fpa_calibration'] = apply_fpa_calibration
    # fpa_parameters['skip_temporary_alignment_for_aligned_apertures'] = skip_temporary_alignment_for_aligned_apertures
    fpa_parameters['skip_temporary_alignment_for_aligned_apertures'] = fpa_parameters['apply_fpa_calibration']

    ############## Main call for running the FPA routine
    obs_collection = alignment.determine_focal_plane_alignment(obs_collection, fpa_parameters)
    ##############

if show_summary_results:
    print('='*100)
    print('GENERATING PLOTS ...')
    save_plot = True

    plot_dir  = os.path.join(working_dir, 'plots')
    for dir in [plot_dir]:
        if os.path.isdir(dir) is False:
            os.makedirs(dir)

    siaf = copy.deepcopy(original_siaf)

    for iii, alignment_reference_aperture_name in enumerate(alignment_reference_apertures):

        attitude_defining_aperture_name = attitude_defining_apertures[iii]
        calibration_alignment_reference_aperture_name = calibration_alignment_reference_aperture_names[iii]
        restrict_to_sets_that_include_aperture = restrict_to_sets_that_include_aperture_names[iii]
        apply_fpa_calibration = apply_fpa_calibration_array[iii]

        obs_collection = []
        result_files = []
        T = []
        observations = []

        result_files = glob.glob(os.path.join(working_dir, 'results/alignment_results_*.pkl'))

#        if apply_fpa_calibration:
#            result_files = glob.glob(os.path.join(working_dir, 'results/alignment_results_*_CALIBRATION_APPLIED.pkl'))


        #result_files = [f for f in result_files if '{}'.format(idl_tel_method) in os.path.basename(f)]

        ### HACK ####
#        result_files = [f for f in result_files if os.path.basename(f).split('_')[2] in program_id[0].split('_')[0]]
        # result_files = [f for f in result_files if os.path.basename(f).split('_')[2] in program_id]

#        if apply_fpa_calibration:
#            result_files = [f for f in result_files if 'alignref_{}_attdef_{}'.format(alignment_reference_aperture_name, attitude_defining_aperture_name) in os.path.basename(f) and '_CALIBRATION_APPLIED' in os.path.basename(f)]
#        else:
#            result_files = [f for f in result_files if 'alignref_{}_attdef_{}'.format(alignment_reference_aperture_name, attitude_defining_aperture_name) in os.path.basename(f) and '_CALIBRATION_APPLIED' not in os.path.basename(f)]
#        if len(result_files) == 0:
#            raise RuntimeError('No Files to process.')

        for i, file in enumerate(result_files):
            obs_collection = pickle.load(open(file, "rb"))
            print('Loaded results from pickled file {}'.format(file))
            if i == 0:
                observations = copy.deepcopy(obs_collection.observations)
                info_dict = copy.deepcopy(obs_collection.info_dict)
                T = copy.deepcopy(obs_collection.T)
            else:
                for key in 'k correct_dva alignment_reference_aperture_name attitude_defining_aperture_name rotation_name'.split():
                    if obs_collection.info_dict[key] != info_dict[key]:
                        print('Skipped!')
                        continue
                observations = np.hstack((observations, obs_collection.observations))
                T = vstack((T, copy.deepcopy(obs_collection.T)))

        obs_collection = []
        obs_collection = alignment.AlignmentObservationCollection(observations)
        obs_collection.T = T
        obs_collection.sort_by('MJD')
        obs_collection.group_by('PROGRAM_VISIT')

        ###
        ### TBD: FIGURE OUT HOW THE visit_groups IS USED THROUGHOUT THIS SCRIPT AND CHANGE IF NECESSARY!! 10/09/2020
        ###
        if len(np.unique(obs_collection.T['group_id'])) == 4:
            info_dict['visit_groups'] = [[0], [1], [2], [3]]
        elif len(np.unique(obs_collection.T['group_id'])) == 2:
            info_dict['visit_groups'] = [[0], [1]]
        else:
            # info_dict['visit_groups'] = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
            info_dict['visit_groups'] = [list(np.arange(len(np.unique(obs_collection.T['group_id']))))]
        info_dict['visit_groups_parameters'] = {}
        info_dict['visit_groups_parameters']['star_marker'] = ['.', 'o', 'x', '+']
        info_dict['visit_groups_parameters']['color'] = ['b', 'g', '0.7', 'k']
        info_dict['siaf'] = siaf
        obs_collection.info_dict = info_dict

        plt.close('all')
#        make_plots = False
        make_plots = True
        evaluate_parameters = {}
        evaluate_parameters['offset_specifiers'] = ['delta_calibrated']
        # evaluate_parameters['offset_specifiers'] = ['delta_calibrated', 'delta_corrected']
        evaluate_parameters['idl_tel_method'] = idl_tel_method
        evaluate_parameters['magnification_factor'] = 1e4

        if make_plots:
            alignment.evaluate(obs_collection, evaluate_parameters,
                               make_summary_plots=make_summary_plots,
                               save_plot=save_plot, plot_dir=plot_dir)


        ######
        ###### Below is for when camera evolution is enabled. Probably not useful for JWST.
        ###### TBD: Remove all parameters ONLY used here (e.g., camera_names?)
        ######
        if show_camera_evolution:
            obs_collection.T['alignment_fit_rms_x'] = np.array([obs.lazAC.rms[1] for obs in obs_collection.observations])[:, 0]
            obs_collection.T['alignment_fit_rms_y'] = np.array([obs.lazAC.rms[1] for obs in obs_collection.observations])[:, 1]

            #
            plt.rcParams['axes.formatter.useoffset'] = False
            historic_data_dir = os.path.join(data_dir, 'pipeline', 'historic_data')

            ### Gotta change below to param later [STS]

            #camera_names = ['NIRISS']
            #aperture_names = {'NIRISS': np.array(['NIS_CEN'])}
            camera_names = ['NIRCam']
            aperture_names = {'NIRCam': np.array(['NRCA1_FULL', 'NRCA2_FULL', 'NRCA3_FULL', 'NRCA4_FULL', 'NRCA5_FULL',
                                                  'NRCB1_FULL', 'NRCB2_FULL', 'NRCB3_FULL', 'NRCB4_FULL', 'NRCB5_FULL'])}

            n_figure_columns = 3
            n_figure_rows = 1
            fig, axes = plt.subplots(n_figure_rows, n_figure_columns,
                                     figsize=(n_figure_columns * 8, n_figure_rows * 6),
                                     facecolor='w', edgecolor='k', sharex=False, sharey=False,
                                     squeeze=False)

            for j, camera_name in enumerate(camera_names):

                i0 = np.where(obs_collection.T['AperName'] == alignment_reference_aperture_name)[0]
                i1 = np.where(obs_collection.T['AperName'] == aperture_names[camera_name][0])[0]

                for jj, colname in enumerate(['delta_calibrated_v2_position_arcsec', 'delta_calibrated_v3_position_arcsec', 'delta_calibrated_v3_angle_arcsec']):
                # for jj, colname in enumerate(['delta_corrected_v2_position_arcsec', 'delta_corrected_v3_position_arcsec', 'delta_corrected_v3_angle_arcsec']):
                    fig_col = jj
                    fig_row = 0
                    axis = axes[fig_row][fig_col]
                    axis.plot(obs_collection.T['alignment_fit_rms_x'][i1]*1e3, np.abs(obs_collection.T[colname][i1]), 'bo', label='{} X rms'.format(camera_name), mfc='w')
                    axis.plot(obs_collection.T['alignment_fit_rms_y'][i1]*1e3, np.abs(obs_collection.T[colname][i1]), 'ro', label='{} Y rms'.format(camera_name), mfc='w')
                    axis.plot(obs_collection.T['alignment_fit_rms_x'][i0]*1e3, np.abs(obs_collection.T[colname][i1]), 'ko', label='{} X rms'.format(alignment_reference_aperture_name))
                    axis.plot(obs_collection.T['alignment_fit_rms_y'][i0]*1e3, np.abs(obs_collection.T[colname][i1]), 'go', label='{} Y rms'.format(alignment_reference_aperture_name))
                    # axis.set_title(colname)
                    axis.set_xlabel('Residual RMS (mas)')
                    axis.set_ylabel('{} {}'.format(aperture_names[camera_name][0], colname))
                    axis.set_title('{} {}'.format(aperture_names[camera_name][0], colname))
                plt.legend()

            plt.show()
#            1/0

            n_figure_columns = 1
            n_figure_rows = 1
            fig, axes = plt.subplots(n_figure_rows, n_figure_columns,
                                    figsize=(n_figure_columns * 8, n_figure_rows * 6),
                                    facecolor='w', edgecolor='k', sharex=False, sharey=False,
                                    squeeze=False)

            mean_table = None # Table(names=('Camera', 'Aperture', ''))
            for j, camera_name in enumerate(camera_names):

                fig_col = j
                fig_row = 0
                axis = axes[fig_row][fig_col]

                axis.set_xlabel('V2 (arcsec)')
                axis.set_ylabel('V3 (arcsec)')
                axis.set_aspect('equal')

                # % plot SIAF value
                v2_mean_siaf = np.mean(np.array([siaf[aper_name].V2Ref for aper_name in aperture_names[camera_name]]))
                v3_mean_siaf = np.mean(np.array([siaf[aper_name].V3Ref for aper_name in aperture_names[camera_name]]))
                axis.plot(v2_mean_siaf, v3_mean_siaf, 'ro', label='siaf.dat', ms=10)

                # retrieve alignment results
                i1 = np.where(obs_collection.T['AperName'] == aperture_names[camera_name][0])[0]
                # i2 = np.where(obs_collection.T['AperName'] == aperture_names[camera_name][1])[0]
                v2_mean_calib = np.mean([obs_collection.T['calibrated_v2_position_arcsec'][i1]], axis=0)
                v3_mean_calib = np.mean([obs_collection.T['calibrated_v3_position_arcsec'][i1]], axis=0)
                # v2_mean_calib = np.mean([obs_collection.T['calibrated_v2_position_arcsec'][i1], obs_collection.T['calibrated_v2_position_arcsec'][i2]], axis=0)
                # v3_mean_calib = np.mean([obs_collection.T['calibrated_v3_position_arcsec'][i1], obs_collection.T['calibrated_v3_position_arcsec'][i2]], axis=0)
                plt.figure()
                plt.plot(obs_collection.T['delta_calibrated_v3_angle_arcsec'][i1], obs_collection.T['alignment_fit_rms_x'][i1], 'bo')
                plt.plot(obs_collection.T['delta_calibrated_v3_angle_arcsec'][i1], obs_collection.T['alignment_fit_rms_y'][i1], 'ro')
                plt.show()


                if j == 0:
                    tex_columns = ['PROGRAM_VISIT', 'AperName']
                    for colname in ['calibrated_V2Ref', 'calibrated_V3Ref', 'calibrated_V3IdlYAngle']:
                        param_name = colname.split('_')[1]
                        sigma_colname = 'sigma_{}'.format(colname)
                        uniform_name = alignment.alignment_parameter_mapping['default_inverse'][param_name]

                        tex_columns.append(uniform_name)
                        tex_columns.append(colname)
                        tex_columns.append(sigma_colname)
                        obs_collection.T[sigma_colname] = obs_collection.T['sigma_calibrated_{}_arcsec'.format(uniform_name)]
                        if param_name == 'V3IdlYAngle':
                            obs_collection.T[sigma_colname] *= u.arcsec.to(u.deg)


                # if ix in [i1, i2]:
                formats = {}
                for key in tex_columns:
                    if 'calibrated' in key:
                        formats[key] = '%2.4f'

                obs_collection.T[tex_columns][np.hstack((i1))].write(os.path.join(plot_dir, '{}_alignment_evolution.tex'.format(camera_name)), format='ascii.latex', formats=formats)
                obs_collection.T[tex_columns][np.hstack((i1))].write(os.path.join(plot_dir, '{}_alignment_evolution.txt'.format(camera_name)), format='ascii.fixed_width', formats=formats)

                # obs_collection.T[tex_columns][np.hstack((i1, i2))].write(os.path.join(plot_dir, '{}_alignment_evolution.tex'.format(camera_name)), format='ascii.latex', formats=formats)

                for epoch, visit_group in enumerate(info_dict['visit_groups']):
                    obs_visit_index = \
                        np.where(np.in1d(obs_collection.T['group_id'][i1], visit_group))[0]
                    plot_color = obs_collection.info_dict['visit_groups_parameters']['color'][epoch]
                    # axis.plot(
                    #     obs_collection.T['{}_v2_position_arcsec'.format(offset_specifier)][
                    #         obs_index][obs_visit_index].data,
                    #     obs_collection.T['{}_v3_position_arcsec'.format(offset_specifier)][
                    #         obs_index][obs_visit_index].data, 'bo', mfc=plot_color,
                    #     mec=plot_color)

                    # axis.plot(v2_mean_calib, v3_mean_calib, 'bo-', label='this work')
                    # axis.errorbar(v2_mean_calib, v3_mean_calib, xerr=obs_collection.T['sigma_calibrated_v2_position_arcsec'][i1], yerr=obs_collection.T['sigma_calibrated_v2_position_arcsec'][i1], ecolor='b', fmt='none')
                axis.plot(v2_mean_calib, v3_mean_calib, 'bo-', label='this work')
                axis.errorbar(v2_mean_calib, v3_mean_calib,
                              xerr=obs_collection.T['sigma_calibrated_v2_position_arcsec'][i1],
                              yerr=obs_collection.T['sigma_calibrated_v2_position_arcsec'][i1],
                              ecolor='b', fmt='none')
                # info_dict['visit_groups_parameters']['color']
                axis.legend()

            fig.tight_layout(h_pad=0.0)
            if save_plot:
                figure_name = os.path.join(plot_dir, '{}_camera_evolution.pdf'.format(info_dict['figure_filename_tag']))
                plt.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)
            plt.show()


            plt.figure(figsize=(8, 8), facecolor='w', edgecolor='k')
            t = copy.deepcopy(obs_collection.T)
            t['v2'] = np.zeros(len(t))
            t['v3'] = np.zeros(len(t))
            linestyles = ['-', '--']
            for j, camera_name in enumerate(camera_names):
                for aperture_name in aperture_names[camera_name]:
                    index = np.where(t['AperName'] == aperture_name)[0]
                    v2 = np.array(t['calibrated_V2Ref'])
                    v3 = np.array(t['calibrated_V3Ref'])
                    v2 -= np.mean(v2)
                    v3 -= np.mean(v3)
                    t['v2'][index] = v2
                    t['v3'][index] = v3
                    plt.plot(v2, v3, 'k:', color='0.5')

            t = t.group_by(['PROGRAM_VISIT'])
            n = 0
            for key, group in zip(t.groups.keys, t.groups):
                col = info_dict['visit_groups_parameters']['color'][n]
                for j, camera_name in enumerate(camera_names):
                    for aperture_name in aperture_names[camera_name]:
                        label='{} {}'.format(camera_name, aperture_name)
                        index = np.where(group['AperName']==aperture_name)[0]
                        plt.plot(group['v2'][index], group['v3'][index], 'bo-',
                                label=label,
                                color=col, mfc=col, mec=col, ls=linestyles[j], lw=3)
                        plt.errorbar(group['v2'][index], group['v3'][index],
                                      xerr=group['sigma_calibrated_v2_position_arcsec'][index],
                                      yerr=group['sigma_calibrated_v3_position_arcsec'][index],
                                      ecolor=col, fmt='none', label='_')

                n += 1
            plt.legend(loc='best')
            plt.axis('equal')
            plt.xlabel('Offset in V2 (arcsec)')
            plt.ylabel('Offset in V3 (arcsec)')
            if save_plot:
                figure_name = os.path.join(plot_dir, '{}_camera_evolution_detail.pdf'.format(info_dict['figure_filename_tag']))
                plt.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)
            plt.show()

            # print average of offsets relative to SIAF
            mean_table = obs_collection.T.group_by(['AperName']).groups.aggregate(np.mean)
            mean_table_columns = ['AperName']
            formats = {}

            for colname in ['calibrated_V2Ref', 'calibrated_V3Ref', 'calibrated_V3IdlYAngle']:
                param_name = colname.split('_')[1]
                uniform_name = alignment.alignment_parameter_mapping['default_inverse'][param_name]
                mean_table_columns.append(uniform_name)
                mean_table_columns.append(colname)
                # mean_table_columns.append('sigma_{}'.format(colname))
                mean_table['discr_{}'.format(uniform_name)] = mean_table['calibrated_{}_arcsec'.format(uniform_name)] - mean_table['{}_arcsec'.format(uniform_name)]
                mean_table_columns.append('discr_{}'.format(uniform_name))
                formats['discr_{}'.format(uniform_name)] = '%2.3f'
                formats['{}'.format(uniform_name)] = '%2.3f'
                formats['{}'.format(colname)] = '%2.3f'
                # formats['sigma_{}'.format(colname)] = '%2.3f'
            mean_table[mean_table_columns].pprint()
            mean_table[mean_table_columns].write(os.path.join(plot_dir, 'alignment_averages.tex'), format='ascii.latex', formats=formats)
            1/0

        # show crossmatch statistics
        if 0:
            obs_collection.T.sort('DATAFILE')
            obs_collection.T['DATAFILE', 'INSTRUME', 'AperName', 'number_of_measured_stars', 'number_of_reference_stars', 'number_of_matched_stars', 'number_of_used_stars_for_aperture_correction'].write(sys.stdout, format='ascii.latex', formats={'DATAFILE': lambda s: s.replace('_','\_'), 'AperName': lambda s: s.replace('_','\_')})

###
### Do we need below? Possibly
###
if show_attitude_evolution:
    attitude_index = [i for i in range(obs_collection.n_observations) if obs_collection.T['AperName'][i] == attitude_defining_aperture_name]

    for i, index in enumerate(attitude_index):
        print('{} {} {}'.format(i, index, obs_collection.observations[i].corrected_attitude['apertures']))


    # Make latex table
    if 1:
        table = copy.deepcopy(obs_collection.T)
        for key in 'n_ap n_stars ra_corr dec_corr pa_corr rms'.split():
            table[key] = None
        table['n_ap'][attitude_index] = np.array([len(obs_collection.observations[i].corrected_attitude['apertures']) for i in attitude_index])
        table['n_stars'][attitude_index] = np.array([obs_collection.observations[i].corrected_attitude['n_stars_total'] for i in attitude_index])
        mapping = {'ra_corr': 'ra_star_arcsec_correction', 'dec_corr': 'dec_arcsec_correction', 'pa_corr': 'pa_arcsec_correction'}
        for key in 'ra_corr dec_corr pa_corr'.split():
            table[key][attitude_index] = ['${:2.3f}\\pm{:2.3f}$'.format(obs_collection.observations[i].corrected_attitude[mapping[key]].value, obs_collection.observations[i].corrected_attitude['sigma_{}'.format(mapping[key])].value) for i in attitude_index]

        table['rms'][attitude_index] = ['{0[0]:2.3f}, {0[1]:2.3f}'.format(obs_collection.observations[i].corrected_attitude['fit_residual_rms']) for i in attitude_index]

        table['pid\_visit'] = [s.replace('_', '\_') for s in table['PROGRAM_VISIT']]
        table[['DATAFILE', 'attitude_id']+'n_ap n_stars ra_corr dec_corr pa_corr rms'.split()][attitude_index].write(sys.stdout, format='latex', formats={'DATAFILE': lambda s: s.replace('_','\_'), 'SIAFAPER': lambda s: s.replace('_','\_')})
        # table[['pid\_visit', 'DATE-OBS', 'attitude_id']+'n_ap n_stars ra_corr dec_corr pa_corr rms'.split()][attitude_index].write(sys.stdout, format='latex')
        # 1/0



    attitudes = {}
    attitude_keys = 'ra_deg dec_deg pa_deg'.split()

    # map uncertainty designations to keys
    uncertainty_mapping = {'ra_deg': 'sigma_ra_star_arcsec_correction',
                           'dec_deg': 'sigma_dec_arcsec_correction',
                           'pa_deg': 'sigma_pa_arcsec_correction',
                           }
    # map FITS header designations to keys
    header_mapping = {'ra_deg': 'pointing_ra_v1',
                      'dec_deg': 'pointing_dec_v1',
                      'pa_deg': 'pointing_pa_v3',
                     }
    # # map FITS header designations to keys
    # header_mapping = {'ra_deg': 'RA_V1',
    #                        'dec_deg': 'DEC_V1',
    #                        'pa_deg': 'PA_V3',
    #                        }

    n_panels = len(attitude_keys) * 2
    n_figure_columns = 3
    n_figure_rows = np.int(np.ceil(n_panels / n_figure_columns))

    fig, axes = plt.subplots(n_figure_rows, n_figure_columns,
                            figsize=(n_figure_columns * 4, n_figure_rows * 3),
                            facecolor='w', edgecolor='k', sharex=True, sharey=False,
                            squeeze=False)
    t_plot = obs_collection.T['MJD'][attitude_index]
    t_plot = np.arange(len(attitude_index))
    for jj, key in enumerate(attitude_keys):
        attitudes['header_{}'.format(key)] = np.array(obs_collection.T[attitude_index][header_mapping[key]])
        attitudes[key] = np.array([obs_collection.observations[i].corrected_attitude[key] for i in attitude_index])
        attitudes['sigma_{}'.format(key)] = np.array([obs_collection.observations[i].corrected_attitude[uncertainty_mapping[key]].value for i in attitude_index])
        if key == 'pa_deg':
            attitudes[key][attitudes[key]>180] -= 180
            attitudes['header_{}'.format(key)][attitudes['header_{}'.format(key)]>180] -= 180
        attitudes[key] *= u.deg.to(u.arcsec)
        attitudes['header_{}'.format(key)] *= u.deg.to(u.arcsec)

        fig_col = jj % n_figure_columns
        fig_row = 0
        # fig_row = jj // n_figure_rows
        axis = axes[fig_row][fig_col]

        refernce_value = np.mean(attitudes[key])
        y_plot = attitudes[key] - refernce_value
        y_header_plot = attitudes['header_{}'.format(key)] - refernce_value

        axis.plot(t_plot, y_plot, 'bo-', label='calibrated')
        axis.plot(t_plot, y_header_plot, 'go-', label='FITS header')
        axis.errorbar(t_plot, y_plot, yerr=attitudes['sigma_{}'.format(key)], fmt='none', ecolor='b')
        axis.set_ylabel('{} variation (arcsec)'.format(key))
        if jj == 0:
            axis.legend(loc='best')

        fig_row = 1
        axis = axes[fig_row][fig_col]
        y_plot = attitudes[key]-attitudes['header_{}'.format(key)]
        axis.plot(t_plot, y_plot, 'ko-', label='difference')
        axis.errorbar(t_plot, y_plot, yerr=attitudes['sigma_{}'.format(key)], fmt='none', ecolor='k')
        axis.set_ylabel('{} difference (arcsec)'.format(key))

    fig.tight_layout(h_pad=0.0)

    if save_plot:
#        figure_name = os.path.join(plot_dir, 'corrected_attitude_from_cameras_{}.pdf'.format(info_dict['figure_filename_tag']))
        figure_name = os.path.join(plot_dir, 'corrected_attitude_from_cameras.pdf')
        plt.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()

# Last part: Write out a human-readable text file that shows the alignment results
result = pickle.load(open(result_files[0], 'rb'))

# Quick solution for adding V3SciXAngle to the results -- NOTE: this won't work for multiple calibration apertures
if distortion_coefficients_file is None or len(distortion_coefficients_file)==0:
    aper = siaf[apertures_to_calibrate[0]]
    AA = aper.get_polynomial_coefficients()['Sci2IdlX']
    BB = aper.get_polynomial_coefficients()['Sci2IdlY']
else:
    coeffs = Table.read(os.path.join(data_dir, distortion_coefficients_file), format='ascii.basic', delimiter=',')
    AA = coeffs['Sci2IdlX']
    BB = coeffs['Sci2IdlY']
betax = np.arctan2(-AA[1],BB[1])

siaf_table = Table()
siaf_table['AperName']         = result.T['aperture_name'][1:]
siaf_table['V3IdlYAngle']      = result.T['calibrated_V3IdlYAngle'][1:] # need to keep the ":" at the end to avoid error
siaf_table['V3SciXAngle']      = np.degrees(betax) + siaf_table['V3IdlYAngle']
siaf_table['V3SciYAngle']      = siaf_table['V3IdlYAngle']
siaf_table['V2Ref']            = result.T['calibrated_V2Ref'][1:]
siaf_table['V3Ref']            = result.T['calibrated_V3Ref'][1:]
###siaf_table['diff_V2Ref']       = result.T['delta_calibrated_v2_position_arcsec'][1]
###siaf_table['diff_V3Ref']       = result.T['delta_calibrated_v3_position_arcsec'][1]
###siaf_table['diff_V3IdlYAngle'] = result.T['delta_calibrated_v3_angle_arcsec'][1]

# For NIRISS and FGS, add another row that shows OSS parameters
if 'NIS' or 'FGS' in apertures_to_calibrate:
    c1 = apertures_to_calibrate[0]+'_OSS'
    c2 = siaf_table['V3IdlYAngle']
    c3 = siaf_table['V3SciXAngle']+180.
    c4 = siaf_table['V3SciYAngle']-180.
    c5 = siaf_table['V2Ref']
    c6 = siaf_table['V3Ref']
    siaf_table.add_row([c1, c2, c3, c4, c5, c6])

username = os.getlogin()
timestamp = Time.now()
instrument_name = result.T['instrument_name'][1]

comments = []
comments.append('{} alignment parameter reference file for SIAF'.format(instrument_name))
comments.append('')
comments.append('This file contains the focal plane alignment parameters calibrated during FGS-SI alignment.')
comments.append('')
comments.append('Generated {} {}'.format(timestamp.isot, timestamp.scale))
comments.append('by {}'.format(username))
comments.append('')
siaf_table.meta['comments'] = comments

result_txt = os.path.join(result_dir,'siaf_alignment.txt')
siaf_table.write(result_txt, format='ascii.fixed_width',
                 delimiter=',', delimiter_pad=' ', bookend=False, overwrite=True)

print('======================================================================================================')
print('END OF SCRIPT: ALL ANALYSES HAVE BEEN COMPLETED. RESULTS ARE AVAILABLE IN THE siaf_alignment.txt FILE. ')
print('======================================================================================================')
sys.exit(0)
