"""Script that determines the distortion coefficients of a JWST imager.

Authors
-------

    Tony Sohn
    (based on original script by Johannes Sahlmann)

Use
---
    From terminal (preferred):
        $ python jwst_distortion.py
    or from ipython session:
        In [1]: run jwst_distortion

"""
import os
import sys
import copy
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.table import Table, Column
from astropy.time import Time

import prepare_jwst_fpa_data
import alignment

from pystortion import distortion

from jwcf import hawki, hst

import pysiaf
from pysiaf.utils import tools
from pysiaf.constants import _DATA_ROOT

plt.close('all')

# Pre-defined distortion polynomial degrees for each instrument
distortion_polynomial_degree = {'niriss': 4, 'fgs': 4, 'nircam': 5, 'miri': 4}


#####################################
### START OF CONFIGURATION PARAMETERS

home_dir = os.environ['HOME']

data_dir = os.path.join(home_dir,'NIRISS/CAP-011b/F150W')

working_dir = os.path.join(data_dir, 'distortion_calibration')

prdopssoc_version = 'PRDOPSSOC-039'

reference_catalog_type = 'hst' # 'hst' for distortion calibrations

# SOURCE EXTRACTION
determine_siaf_parameters = True # Keep this to "True" since that'll take out the V2ref, V3ref, V3IdlYAngle
use_epsf = False # This doesn't work as intended for now, so keep it turned off until method is established
overwrite_source_extraction = True # or False
generate_standardized_fpa_data = True # or False
overwrite_distortion_reference_table = True

save_plot = True # or False
verbose = True # or False
verbose_figures = True # or False
show_extracted_sources = True # or False
show_psfsubtracted_image = True

overwrite_obs_collection = True # or False
overwrite_obs_xmatch_pickle = True # or False

inspect_mode = True # or False
if inspect_mode is False:
    verbose_figures = False

camera_pattern = '_cal.fits'
nominalpsf = True # False for OTE programs, True for NIS-011b
distortion_coefficients_file = None # 'distortion_coeffs_nis_cen_jw01086001001_01101_00021_nis_cal.txt'

### END OF CONFIGURATION PARAMETERS

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


def write_distortion_reference_file(coefficients_dict, verbose=False):
    """Write distortion reference file in SIAF source file format.

    Parameters
    ----------
    coefficients_dict

    Returns
    -------

    """
    siaf_index = []
    exponent_x = []
    exponent_y = []
    for i in range(polynomial_degree + 1):
        for j in np.arange(i + 1):
            siaf_index.append('{:d}{:d}'.format(i, j))
            exponent_x.append(i - j)
            exponent_y.append(j)

    distortion_reference_table = Table((siaf_index, exponent_x, exponent_y,
                                        coefficients_dict['Sci2IdlX'],
                                        coefficients_dict['Sci2IdlY'],
                                        coefficients_dict['Idl2SciX'],
                                        coefficients_dict['Idl2SciY']),
                                        names=('siaf_index',
                                               'exponent_x', 'exponent_y',
                                               'Sci2IdlX', 'Sci2IdlY',
                                               'Idl2SciX', 'Idl2SciY'))

    distortion_reference_table.add_column(
        Column([aperture_name] * len(distortion_reference_table), name='AperName'), index=0)
    distortion_reference_file_name = os.path.join(result_dir, 'distortion_coeffs_{}_{}.txt'.format(
        aperture_name.lower(), coefficients_dict['name_seed']))
    if verbose:
        distortion_reference_table.pprint()

    username = os.getlogin()
    timestamp = Time.now()

    comments = []
    comments.append('{} distortion coefficient file\n'.format(instrument_name))
    comments.append('Source file: {}'.format(file_name))
    comments.append('Aperture: {}'.format(aperture_name))
    comments.append('')
    comments.append('Generated {} {}'.format(timestamp.isot, timestamp.scale))
    comments.append('by {}'.format(username))
    comments.append('')
    distortion_reference_table.meta['comments'] = comments
    distortion_reference_table.write(distortion_reference_file_name, format='ascii.fixed_width',
                                     delimiter=',', delimiter_pad=' ', bookend=False,
                                     overwrite=overwrite_distortion_reference_table)

    return distortion_reference_file_name

def write_distortion_reference_oss_file(distortion_reference_file_name):
    """Write the OSS version of distortion reference file for NIRISS and FGS.

    Parameters
    ----------
    distortion_reference_file_name

    Returns
    -------

    """
    distortion_coefficients_table = Table.read(distortion_reference_file_name,
                                               format='ascii.basic', delimiter=',')

    comments = distortion_coefficients_table.meta['comments']
    if any('NIS_CEN' in s for s in comments):
        new_comments = [w.replace('NIS_CEN','NIS_CEN_OSS') for w in comments]
        distortion_coefficients_table.meta['comments'] = new_comments

    A = distortion_coefficients_table['Sci2IdlX']
    B = distortion_coefficients_table['Sci2IdlY']
    C = distortion_coefficients_table['Idl2SciX']
    D = distortion_coefficients_table['Idl2SciY']

    # Flip parity for certain coefficients
    A[[ 0, 3, 4, 5, 10, 11, 12, 13, 14]] *= -1
    B[[ 1, 2, 6, 7,  8,  9            ]] *= -1
    C[[ 2, 3, 5, 7,  9, 10, 12, 14    ]] *= -1
    D[[ 2, 3, 5, 7,  9, 10, 12, 14    ]] *= -1

    distortion_coefficients_table['Sci2IdlX'] = A
    distortion_coefficients_table['Sci2IdlY'] = B
    distortion_coefficients_table['Idl2SciX'] = C
    distortion_coefficients_table['Idl2SciY'] = D

    distortion_coefficients_table['temp'] = \
        [distortion_coefficients_table['AperName'][0]+'_OSS']*len(A)
    distortion_coefficients_table['AperName'] = distortion_coefficients_table['temp']
    distortion_coefficients_table.remove_column('temp')

    f = distortion_reference_file_name.split('.')
    distortion_reference_oss_file_name = f[0]+'_oss.'+f[1]

    distortion_coefficients_table.write(distortion_reference_oss_file_name, format='ascii.fixed_width',
                                        delimiter=',', delimiter_pad=' ', bookend=False,
                                        overwrite=overwrite_distortion_reference_table)

    return distortion_reference_oss_file_name

idl_tel_method = 'spherical' # or 'planar_approximation'

#=============================================================================
#
print('{}\nGEOMETRIC DISTORTION CALIBRATION'.format('='*100))

obs_collection = []

standardized_data_dir = os.path.join(working_dir, 'fpa_data')
result_dir = os.path.join(working_dir, 'results')
plot_dir = os.path.join(working_dir, 'plots')

for dir in [standardized_data_dir, plot_dir, result_dir]:
    if os.path.isdir(dir) is False: os.makedirs(dir)

if (generate_standardized_fpa_data) or (not glob.glob(os.path.join(standardized_data_dir, '*.fits'))):

    extraction_parameters = {'nominalpsf': nominalpsf,
                             'use_epsf': use_epsf,
                             'show_extracted_sources': show_extracted_sources,
                             'show_psfsubtracted_image': show_psfsubtracted_image}
                             #'epsf_psf_size_pix': 20,
                             #'use_DAOStarFinder_for_epsf' : use_DAOStarFinder_for_epsf,
                             #'use_weights_for_epsf': False,
                             #'use_weights_for_epsf': use_weights_for_epsf,
                             #'flux_threshold_percentile_lower':  5.,
                             #'flux_threshold_percentile_upper':  95.,
                             #'dao_detection_threshold': 30.}
                             #'final_extraction_niters': 5}
                             # 'use_epsf': False,
                             # 'show_extracted_sources': False}

    im = prepare_jwst_fpa_data.jwst_camera_fpa_data(data_dir, camera_pattern, standardized_data_dir,
                                                    parameters=extraction_parameters,
                                                    overwrite_source_extraction=overwrite_source_extraction)


# 1/0
plt.close('all')

# Load all relevant siaf apertures
apertures_dict = {}
apertures_dict['instrument'] = ['NIRCAM']*10 + ['FGS']*2 + ['NIRISS'] + ['MIRI']
apertures_dict['pattern'] = ['NRCA1_FULL', 'NRCA2_FULL', 'NRCA3_FULL', 'NRCA4_FULL', 'NRCA5_FULL',
                             'NRCB1_FULL', 'NRCB2_FULL', 'NRCB3_FULL', 'NRCB4_FULL', 'NRCB5_FULL',
                             'FGS1_FULL', 'FGS2_FULL', 'NIS_CEN', 'MIRIM_FULL']
siaf = pysiaf.siaf.get_jwst_apertures(apertures_dict, exact_pattern_match=True)

#siaf_detector_layout = pysiaf.read.read_siaf_detector_layout()
#apertures_dict={'instrument': siaf_detector_layout['InstrName'].data}
#master_aperture_names = siaf_detector_layout['AperName'].data
#apertures_dict['pattern'] = master_aperture_names
#siaf = pysiaf.siaf.get_jwst_apertures(apertures_dict)

# Prepare the reference catalog

if reference_catalog_type.lower() == 'hawki':
    reference_catalog = hawki.hawki_catalog()
    reference_catalog.rename_column('ra_deg', 'ra')
    reference_catalog.rename_column('dec_deg', 'dec')
    reference_catalog['j_magnitude'] = reference_catalog['j_2mass_extrapolated']
elif reference_catalog_type.lower() == 'hst':
    reference_catalog = hst.hst_catalog(decimal_year_of_observation=2022.0)
    reference_catalog.rename_column('ra_deg', 'ra')
    reference_catalog.rename_column('dec_deg', 'dec')
    reference_catalog['j_magnitude'] = reference_catalog['j_mag_vega']
else:
    sys.exit('Unsupported Reference Catalog. Only HawkI and HST catalogs are currently supported.')

# define pickle files
obs_xmatch_pickle_file = os.path.join(result_dir, 'obs_xmatch.pkl')
obs_collection_pickle_file = os.path.join(result_dir, 'obs_collection.pkl')
crossmatch_dir = os.path.join(standardized_data_dir, 'crossmatch')
if os.path.isdir(crossmatch_dir) is False: os.makedirs(crossmatch_dir)

if (not os.path.isfile(obs_collection_pickle_file)) | (overwrite_obs_collection):

    # crossmatch the stars in every aperture with the reference catalog (here Gaia)
    crossmatch_parameters = {}
    crossmatch_parameters['pickle_file'] = obs_xmatch_pickle_file
    crossmatch_parameters['overwrite'] = overwrite_obs_xmatch_pickle
    crossmatch_parameters['data_dir'] = data_dir
    crossmatch_parameters['standardized_data_dir'] = standardized_data_dir
    crossmatch_parameters['verbose_figures'] = verbose_figures
    crossmatch_parameters['save_plot'] = save_plot
    crossmatch_parameters['plot_dir'] = crossmatch_dir
    crossmatch_parameters['correct_reference_for_proper_motion'] = False # or True
    crossmatch_parameters['overwrite_pm_correction'] = False # or True
    crossmatch_parameters['verbose'] = verbose
    crossmatch_parameters['siaf'] = siaf
    crossmatch_parameters['idl_tel_method'] = idl_tel_method
    crossmatch_parameters['reference_catalog'] = reference_catalog
    crossmatch_parameters['xmatch_radius'] = 0.1 * u.arcsec # 0.2 arcsec is about 3 pixels in NIRISS or FGS
    crossmatch_parameters['rejection_level_sigma'] = 2.5 # or 5
    crossmatch_parameters['restrict_analysis_to_these_apertures'] = None
    crossmatch_parameters['distortion_coefficients_file'] = distortion_coefficients_file
    #        crossmatch_parameters['xmatch_radius_camera'] = 0.2 * u.arcsec
    #        crossmatch_parameters['xmatch_radius_fgs'] = None
    #        if run_on_single_niriss_file or run_on_single_fgs_file:
    #            crossmatch_parameters['file_pattern'] = '*{}'.format(camera_pattern)

    # Call the crossmatch routine
    observations = prepare_jwst_fpa_data.crossmatch_fpa_data(crossmatch_parameters)

    # Generate an AlignmentObservationCollection object
    obs_collection = alignment.AlignmentObservationCollection(observations)
    pickle.dump(obs_collection, open(obs_collection_pickle_file, "wb"))
else:
    obs_collection = pickle.load(open(obs_collection_pickle_file, "rb"))
    print('Loaded pickled file {}'.format(obs_collection_pickle_file))

for obs in obs_collection.observations:
    file_name = obs.fpa_data.meta['DATAFILE']

    instrument_name = obs.fpa_data.meta['INSTRUME']
    aperture_name   = obs.aperture.AperName

    plt.close('all')
    print('+'*100)
    print('Distortion calibration of {}'.format(os.path.basename(file_name)))

    name_seed = os.path.basename(file_name).replace('.fits', '')

    # compute ideal coordinates
    obs.reference_catalog_matched = alignment.compute_tel_to_idl_in_table(obs.reference_catalog_matched, obs.aperture)

    fieldname_dict = copy.deepcopy(obs.fieldname_dict)
    fieldname_dict['reference_catalog']['position_1'] = 'x_idl_arcsec'
    fieldname_dict['reference_catalog']['position_2'] = 'y_idl_arcsec'
    fieldname_dict['reference_catalog']['sigma_position_1'] = 'ra_error_mas'
    fieldname_dict['reference_catalog']['sigma_position_2'] = 'dec_error_mas'
    fieldname_dict['reference_catalog']['position_unit'] = u.arcsecond
    fieldname_dict['reference_catalog']['sigma_position_unit'] = u.milliarcsecond
    fieldname_dict['star_catalog']['position_1'] = 'x_SCI'
    fieldname_dict['star_catalog']['position_2'] = 'y_SCI'
    fieldname_dict['star_catalog']['sigma_position_1'] = 'sigma_x_mas'
    fieldname_dict['star_catalog']['sigma_position_2'] = 'sigma_y_mas'
    fieldname_dict['star_catalog']['position_unit'] = u.dimensionless_unscaled
    fieldname_dict['star_catalog']['sigma_position_unit'] = u.dimensionless_unscaled

    mp = distortion.prepare_multi_epoch_astrometry(obs.star_catalog_matched, obs.reference_catalog_matched, fieldname_dict=fieldname_dict)

    fieldname_dict_inverse = copy.deepcopy(obs.fieldname_dict)
    fieldname_dict_inverse['star_catalog']['position_1'] = 'x_idl_arcsec'
    fieldname_dict_inverse['star_catalog']['position_2'] = 'y_idl_arcsec'
    fieldname_dict_inverse['star_catalog']['sigma_position_1'] = 'ra_error_mas'
    fieldname_dict_inverse['star_catalog']['sigma_position_2'] = 'dec_error_mas'
    fieldname_dict_inverse['star_catalog']['position_unit'] = u.arcsecond
    fieldname_dict_inverse['star_catalog']['sigma_position_unit'] = u.milliarcsecond
    fieldname_dict_inverse['star_catalog']['identifier'] = 'ID'
    fieldname_dict_inverse['reference_catalog']['position_1'] = 'x_SCI'
    fieldname_dict_inverse['reference_catalog']['position_2'] = 'y_SCI'
    fieldname_dict_inverse['reference_catalog']['sigma_position_1'] = 'sigma_x_mas'
    fieldname_dict_inverse['reference_catalog']['sigma_position_2'] = 'sigma_y_mas'
    fieldname_dict_inverse['reference_catalog']['position_unit'] = u.dimensionless_unscaled
    fieldname_dict_inverse['reference_catalog']['sigma_position_unit'] = u.dimensionless_unscaled
    fieldname_dict_inverse['reference_catalog']['identifier'] = 'id'

    mp_inverse = distortion.prepare_multi_epoch_astrometry(obs.reference_catalog_matched, obs.star_catalog_matched, fieldname_dict=fieldname_dict_inverse)

    k = degree_to_mode(distortion_polynomial_degree[instrument_name.lower()])
    reference_frame_number  = 0  # reference frame, evaluation_frame_number is calibrated against this frame
    evaluation_frame_number = 1  # frame to calibrate
    # use_position_uncertainties = 1  # use the individual astrometric uncertainties in the
    # polynomial fit
    reference_point = np.array([[obs.aperture.XSciRef, obs.aperture.YSciRef], [0., 0.]])
    lazAC, index_masked_stars = distortion.fit_distortion_general(mp, k,
                                            eliminate_omc_outliers_iteratively=1,
                                            outlier_rejection_level_sigma=3.,
                                            reference_frame_number=reference_frame_number,
                                            evaluation_frame_number=evaluation_frame_number,
                                            reference_point=reference_point,
                                            verbose=True)

    # pysiaf.utils.polynomial.polyfit(A, xin, yin, order=)


    reference_point_inverse = np.array([[0., 0.], [obs.aperture.XSciRef, obs.aperture.YSciRef]])
    lazAC_inverse, index_masked_stars_inverse = distortion.fit_distortion_general(mp_inverse, k,
                                            eliminate_omc_outliers_iteratively=1,
                                            outlier_rejection_level_sigma=3.,
                                            reference_frame_number=reference_frame_number,
                                            evaluation_frame_number=evaluation_frame_number,
                                            reference_point=reference_point_inverse,
                                            verbose=True)


    scale_factor_for_residuals = 1000.

    lazAC.display_results(evaluation_frame_number=evaluation_frame_number,
                          scale_factor_for_residuals=scale_factor_for_residuals,
                          display_correlations=0)
    # print('Parameters of %s polynomial computed from SIAF' % parameter_set)
    # pyDistortion.display_RotScaleSkew(coefficients_x * 1e3, coefficients_y * 1e3)
    print('Parameters of fitted polynomial coefficients')
    distortion.display_RotScaleSkew(
    lazAC.Alm[evaluation_frame_number, 0:lazAC.Nalm] * scale_factor_for_residuals,
    lazAC.Alm[evaluation_frame_number, lazAC.Nalm:] * scale_factor_for_residuals)

    # pystortion.distortion.displayRotScaleSkew(lazAC, i=evaluation_frame_number,
    #                                  scaleFactor=scale_factor_for_residuals)

    xy_unit = u.arcsec
    xy_unitStr = xy_unit.to_string()
    xy_scale = 1.
    lazAC.plotResiduals(evaluation_frame_number, plot_dir, name_seed,
                    omc_scale=scale_factor_for_residuals, save_plot=1,
                    omc_unit='mas', xy_scale=xy_scale, xy_unit=xy_unitStr)
    if 0:
        lazAC.plotResults(evaluation_frame_number, plot_dir, name_seed, saveplot=1,
                          xy_scale=xy_scale, xy_unit=xy_unitStr)
        lazAC.plotDistortion(evaluation_frame_number, plot_dir, name_seed,
                             reference_point[reference_frame_number], save_plot=1,
                             xy_scale=xy_scale, xy_unit=xy_unitStr, detailed_plot_k=k)
        # lazAC.plotDistortion(evaluation_frame_number,outDir,name_seed_2,
        # referencePointForProjection_Pix,save_plot=save_plot,xy_scale=xy_scale,
        # xy_unit=xy_unitStr)
        # lazAC.plotLinearTerms(evaluation_frame_number,outDir,name_seed_2,
        # referencePointForProjection_Pix,save_plot=save_plot,xy_scale=xy_scale,
        # xy_unit=xy_unitStr)

    ############################################################

    print('Number of xmatches between reference catalog and detected sources: %d' % len(
    obs.star_catalog_matched))
    print('Polynomial fit residuals: %3.3e native = %3.3f mas' % (
    np.mean(lazAC.rms[1, :]), np.mean(lazAC.rms[1, :] * scale_factor_for_residuals)))

    if determine_siaf_parameters:

        siaf_aper = siaf[aperture_name]

        print('*' * 100)
        print('Distortion parameter preparation for SIAF')

        polynomial_degree =  distortion_polynomial_degree[instrument_name.lower()]
        number_of_coefficients = pysiaf.polynomial.number_of_coefficients(polynomial_degree)

        A = lazAC.Alm[evaluation_frame_number][0:number_of_coefficients]
        B = lazAC.Alm[evaluation_frame_number][number_of_coefficients:]

        # THIS IS WRONG, NEEDS UPDATE
        # Determine the inverse coefficients and perform roundtrip verification
        # C = A
        # D = B
        C = lazAC_inverse.Alm[evaluation_frame_number][0:number_of_coefficients]
        D = lazAC_inverse.Alm[evaluation_frame_number][number_of_coefficients:]

        coefficients_dict_prep = {'Sci2IdlX': A, 'Sci2IdlY': B, 'Idl2SciX': C, 'Idl2SciY': D,
                              'out_dir': plot_dir, 'aperture_name': aperture_name,
                              'instrument_name': instrument_name,
                              'name_seed': '{}_prep'.format(name_seed)}

        distortion_reference_file_name_prep = write_distortion_reference_file(coefficients_dict_prep)
        new_aperture_prep = pysiaf.aperture.Aperture()
        new_aperture_prep.set_distortion_coefficients_from_file(distortion_reference_file_name_prep)
        linear_parameters_prep = new_aperture_prep.get_polynomial_linear_parameters()
        linear_parameters_inverse = new_aperture_prep.get_polynomial_linear_parameters(coefficient_seed='Idl2Sci')
        # print(linear_parameters_prep)

        # Take out Y-rotation and offsets in both forward and reverse coefficients
        # This step is required to make the results consistent with SIAF convention.
        # By default, the polynomial coefficients derived up until now include not only
        # geometric distortion-related component but also lateral offsets in X, Y, and rotation.
        # Corrections for these latter components are carried out via aperture-specific
        # alignment parmeters (V2Ref, V3Ref, and V3IdlAngle). So, here we take out contributions
        # from these three parameters using pysiaf.polynomial.add_rotation as below.
        AR = copy.deepcopy(A)
        BR = copy.deepcopy(B)
        AR[0] = 0 # Set the first coeff (zero point) to zero
        BR[0] = 0 # Set the first coeff (zero point) to zero
        (AR, BR) = pysiaf.polynomial.add_rotation(AR, BR, -1*linear_parameters_prep['rotation_y'])
        CR = copy.deepcopy(C)
        DR = copy.deepcopy(D)
        CR[0] = 0
        DR[0] = 0
        (CR, DR) = pysiaf.polynomial.add_rotation(CR, DR, -1*linear_parameters_inverse['rotation_y'])

        poly_coeffs = pysiaf.utils.tools.convert_polynomial_coefficients(A, B, C, D)
        siaf_params_file = os.path.join(result_dir, 'siaf_params_{}_{}.txt'.format(
         aperture_name.lower(), coefficients_dict_prep['name_seed']))
        with open(siaf_params_file, 'w') as f:
            print('V2Ref       =', poly_coeffs[6], file=f)
            print('V3Ref       =', poly_coeffs[7], file=f)
            print('V3SciXAngle =', poly_coeffs[4], file=f)
            print('V3SciYAngle =', poly_coeffs[5], file=f)

        # check roundtrip errors for these coefficients
        for attribute in 'XSciRef YSciRef XSciSize InstrName AperName'.split():
            setattr(new_aperture_prep, attribute, getattr(siaf_aper, attribute))

        roundtrip_errors = tools.compute_roundtrip_error(AR, BR, CR, DR,
                                                         offset_x=new_aperture_prep.XSciRef,
                                                         offset_y=new_aperture_prep.YSciRef,
                                                         instrument=new_aperture_prep.InstrName,
                                                         grid_amplitude=new_aperture_prep.XSciSize)

        print('Roundtrip errors: {0[1]} and {0[2]} mean; {0[3]} and {0[4]} RMS'.format(roundtrip_errors))
        threshold_pix = 0.05 # original: 1e-2
        ####
        #### Turned off for now [STS]
        ####
        #            for j in [1,2,3,4]:
        #                assert np.abs(roundtrip_errors[j]) < threshold_pix

        # plot roundtrip errors
        if 0:
            data = roundtrip_errors[-1]
            plt.figure(figsize=(6, 6), facecolor='w', edgecolor='k')
            plt.quiver(data['x'], data['y'], data['x'] - data['x2'], data['y'] - data['y2'],
                  angles='xy')
            plt.xlabel('x_sci')
            plt.ylabel('y_sci')
            new_aperture_prep.plot(frame='sci', ax=plt.gca())
            ax = plt.gca()
            plt.text(0.5, 0.9, 'Maximum arrow length {:3.3f} pix'.format(
                np.max(np.linalg.norm([data['x'] - data['x2'], data['y'] - data['y2']], axis=0))),
                horizontalalignment='center', transform=ax.transAxes)
            plt.title('{} Roundtrip error sci->idl->sci'.format(new_aperture_prep.AperName))
            if inspect_mode: plt.show()

        # write pysiaf source data file with distortion coefficients
        coefficients_dict = {'Sci2IdlX': AR, 'Sci2IdlY': BR, 'Idl2SciX': CR, 'Idl2SciY': DR,
        'out_dir': plot_dir, 'aperture_name': aperture_name, 'instrument_name': instrument_name, 'name_seed': name_seed}

        distortion_reference_file_name = write_distortion_reference_file(coefficients_dict)
        if 'NIRISS' or 'FGS' in instrument_name:
            distortion_reference_oss_file_name = write_distortion_reference_oss_file(distortion_reference_file_name)
        new_aperture = pysiaf.Aperture()
        new_aperture.set_distortion_coefficients_from_file(distortion_reference_file_name)
        linear_parameters = new_aperture.get_polynomial_linear_parameters()

        #  verify that rotation is close to zero
        assert np.abs(linear_parameters['rotation_y']) < 1e-12

        verify_distortion_requirement = True
        if verify_distortion_requirement:
        # verify requirement, like in Anderson 2016.
        # the idea is to compare against the input SIAF transformation
        # the requiremnt in < 5 mas RMS per axis

            for attribute in 'XSciRef YSciRef'.split():
                setattr(new_aperture, attribute, getattr(siaf_aper, attribute))

            # get the SIAF version used in the simulations
            ref_siaf = pysiaf.siaf.Siaf(instrument_name,
                                        basepath=os.path.join(_DATA_ROOT, 'JWST',
                                                              prdopssoc_version,
                                                              'SIAFXML', 'SIAFXML'))
            # SIAF transformation
            x_idl_siaf, y_idl_siaf = ref_siaf[aperture_name].sci_to_idl(
                    obs.star_catalog_matched['x_SCI'].data,
                    obs.star_catalog_matched['y_SCI'].data)

            # transformation using newly determined coefficients
            x_idl_check, y_idl_check = new_aperture.sci_to_idl(
                    obs.star_catalog_matched['x_SCI'].data,
                    obs.star_catalog_matched['y_SCI'].data)

            # Plot difference

            data = {}
            data['reference'] = {'x': x_idl_siaf, 'y': y_idl_siaf}
            data['comparison_0'] = {'x': x_idl_check, 'y': y_idl_check}

            plt.figure(figsize=(10,10), facecolor='w', edgecolor='k')
            delta_x = data['comparison_0']['x'] - data['reference']['x']
            delta_y = data['comparison_0']['y'] - data['reference']['y']

            plt.quiver(data['reference']['x'], data['reference']['y'],
                       delta_x, delta_y, angles='xy', scale=None)
            offsets = np.linalg.norm([delta_x, delta_y], axis=0)

            plt.title('Max difference {:2.3f} mas'.format(np.max(offsets)*1e3))
            plt.axis('tight')
            plt.axis('equal')
            plt.xlabel('X (arcsec)')
            plt.ylabel('Y (arcsec)')
            plt.legend(loc='best')
            ax = plt.gca()
            ax.invert_yaxis()
            if save_plot:
                figname = os.path.join(plot_dir,'spatial_difference.pdf')
                plt.savefig(figname, transparent=True, bbox_inches='tight', pad_inches=0)
            plt.show()

            rms_x = np.std(x_idl_check - x_idl_siaf)
            rms_y = np.std(y_idl_check - y_idl_siaf)
            print("rms_x =",rms_x)
            print("rms_y =",rms_y)
            #assert rms_x < 0.005 # Mission requirement is <5 mas per axis
            #assert rms_y < 0.005 # Mission requirement is <5 mas per axis

print('================================================')
print('END OF SCRIPT: ALL ANALYSES HAVE BEEN COMPLETED.')
print('================================================')
sys.exit(0)
