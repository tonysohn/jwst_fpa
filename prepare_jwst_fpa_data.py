import os
import copy
import glob
import numpy as np
import pickle
import sys
import time
import warnings
import corner
import subprocess
from collections import OrderedDict

import astropy.io.fits as fits
from astropy.table import Table, vstack
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning
from astropy.stats import sigma_clip, sigma_clipped_stats, gaussian_sigma_to_fwhm
from astropy.visualization import SqrtStretch, LogStretch, simple_norm
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.nddata import NDData, StdDevUncertainty, NDUncertainty
from astropy.modeling.fitting import LevMarLSQFitter

from jwst import datamodels
from photutils import IRAFStarFinder, DAOStarFinder, find_peaks
from photutils import CircularAperture, RectangularAperture, EPSFBuilder
from photutils.psf import DAOGroup, IntegratedGaussianPRF
from photutils.psf import extract_stars, IterativelySubtractedPSFPhotometry
from photutils.background import MMMBackground, MADStdBackgroundRMS
from photutils.centroids import centroid_2dg, centroid_com

import pysiaf
from pystortion import crossmatch
from scipy.spatial import cKDTree

from alignment import AlignmentObservation, compute_idl_to_tel_in_table

# Below was added to figure out the initial offset between reference and observed catalog positions. [STS]
from tweakwcs import matchutils

from matplotlib import style, pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker

def select_isolated_sources(extracted_sources, nearest_neighbour_distance_threshold_pix):
    """
    select isolated stars
    https://stackoverflow.com/questions/57129448/find-distance-to-nearest-neighbor-in-2d-array

    Parameters
    ----------
    extracted_sources
    nearest_neighbour_distance_threshold_pix

    Returns
    -------

    """
    stars_xy = np.array([extracted_sources['xcentroid'], extracted_sources['ycentroid']]).T
    tree = cKDTree(stars_xy)
    dists = tree.query(stars_xy, 2)
    nearest_neighbour_distance = dists[0][:, 1]

    extracted_sources.remove_rows(
        np.where(nearest_neighbour_distance < nearest_neighbour_distance_threshold_pix)[0])

    return extracted_sources

def jwst_camera_fpa_data(data_dir, pattern, standardized_data_dir, parameters,
                         overwrite_source_extraction=False):
    """Generate standardized focal plane alignment (fpa) data based on JWST camera data.
    """

    file_list = glob.glob(os.path.join(data_dir, '*{}'.format(pattern)))

    if len(file_list) == 0:
        raise RuntimeError('No data found')

    file_list.sort()
    for f in file_list:

        plt.close('all')

        print()
        print('Data directory: {}'.format(data_dir))
        print('Image being processed: {}'.format(f))

        im = datamodels.open(f)
        if hasattr(im, 'data') is False:
            im.data = fits.getdata(f)
            #im.dq    = np.zeros(im.data.shape)

        header_info = OrderedDict()

        for attribute in 'telescope'.split():
            header_info[attribute] = getattr(im.meta, attribute)

        # observations
        for attribute in 'date time visit_number visit_id visit_group activity_id program_number'.split():
            header_info['observation_{}'.format(attribute)] = getattr(im.meta.observation, attribute)

        header_info['epoch_isot'] = '{}T{}'.format(header_info['observation_date'], header_info['observation_time'])

        #  instrument
        for attribute in 'name filter pupil detector'.split():
            header_info['instrument_{}'.format(attribute)] = getattr(im.meta.instrument, attribute)

        # subarray
        for attribute in 'name'.split():
            header_info['subarray_{}'.format(attribute)] = getattr(im.meta.subarray, attribute)

        # aperture
        for attribute in 'name position_angle pps_name'.split():
            try:
                value = getattr(im.meta.aperture, attribute)
            except AttributeError:
                value = None

            header_info['aperture_{}'.format(attribute)] = value

        header_info['INSTRUME'] = header_info['instrument_name']
        header_info['SIAFAPER'] = header_info['aperture_name']

        # temporary solution, this should come from populated aperture attributes
        #if header_info['subarray_name'] == 'FULL':
        #    master_apertures = pysiaf.read.read_siaf_detector_layout()
        #    if header_info['instrument_name'].lower() in ['niriss', 'miri']:
        #        header_info['SIAFAPER'] = master_apertures['AperName'][np.where(master_apertures['InstrName']==header_info['instrument_name'])[0][0]]
        #    elif header_info['instrument_name'].lower() in ['fgs']:
        #        header_info['SIAFAPER'] = 'FGS{}_FULL'.format(header_info['instrument_detector'][-1])
        #    elif header_info['instrument_name'].lower() in ['nircam']:
        #        header_info['SIAFAPER'] = header_info['aperture_name']
        #else:
        #    sys.exit('Only FULL arrays are currently supported.')

        # target
        for attribute in 'ra dec catalog_name proposer_name'.split():
            header_info['target_{}'.format(attribute)] = getattr(im.meta.target, attribute)

        # pointing
        for attribute in 'ra_v1 dec_v1 pa_v3'.split():
            try:
                value = getattr(im.meta.pointing, attribute)
            except AttributeError:
                value = None
            header_info['pointing_{}'.format(attribute)] = value

        # add HST style keywords
        header_info['PROGRAM_VISIT'] = '{}_{}'.format(header_info['observation_program_number'], header_info['observation_visit_id'])
        header_info['PROPOSID'] = header_info['observation_program_number']
        header_info['DATE-OBS'] = header_info['observation_date']
        header_info['TELESCOP'] = header_info['telescope']
        header_info['INSTRUME'] = header_info['instrument_name']
        try:
            header_info['APERTURE'] = header_info['SIAFAPER']
        except KeyError:
            header_info['APERTURE'] = None
        header_info['CHIP'] = 0

        extracted_sources_dir = os.path.join(standardized_data_dir, 'extraction')
        if os.path.isdir(extracted_sources_dir) is False: os.makedirs(extracted_sources_dir)
        extracted_sources_file = os.path.join(extracted_sources_dir,
                                              '{}_extracted_sources.fits'.format(os.path.basename(f).split('.')[0]))

        mask_extreme_slope_values = False
        parameters['maximum_slope_value'] = 1000.

        # parameters['use_DAOStarFinder_for_epsf'] = False
        #parameters['use_DAOStarFinder_for_epsf'] = use_DAOStarFinder_for_epsf

        # Check if extracted_sources_file exists, or overwrite_source_extraction is set to True
        if (not os.path.isfile(extracted_sources_file)) or (overwrite_source_extraction):
            data = copy.deepcopy(im.data)
            #dq = copy.deepcopy(im.dq)

            # Convert image data to counts per second
            photmjsr = getattr(im.meta.photometry,'conversion_megajanskys')
            data_cps = data/photmjsr

            if mask_extreme_slope_values:
                # clean up extreme slope values
                bad_index = np.where(np.abs(data) > parameters['maximum_slope_value'])
                data[bad_index] = 0.
                dq[bad_index] = -1

            bkgrms = MADStdBackgroundRMS()
            mmm_bkg = MMMBackground()
            bgrms = bkgrms(data_cps)
            bgavg = mmm_bkg(data_cps)

            # Default criterions
            round_lo, round_hi = 0.0, 0.6
            sharp_lo, sharp_hi = 0.3, 1.4
            fwhm_lo, fwhm_hi   = 1.0, 2.5

            # Use different criteria for selecting good stars
            if parameters['nominalpsf']:
                # If using Nominal PSF models
                if 'nis' in f:
                    #fwhm_lo, fwhm_hi = 1.0, 2.0
                    sharp_lo, sharp_hi = 0.6, 1.4
                elif 'g1' or 'g2' in f:
                    #fwhm_lo, fwhm_hi = 1.0, 1.4
                    sharp_lo, sharp_hi = 0.6, 1.4
                elif 'nrca' or 'nrcb' in f:
                    sharp_lo, sharp_hi = 0.6, 1.4  ### PLACEHOLDER!!! CHANGE AFTER INSPECTION
            else:
                # If using Commissioning (coarsely-phased) PSF models
                if 'nis' in f:
                    sharp_lo, sharp_hi = 0.6, 1.4
                    fwhm_lo, fwhm_hi   = 1.4, 2.4
                elif 'g1' or 'g2' in f:
                    sharp_lo, sharp_hi = 0.8, 1.4  ### PLACEHOLDER!!! CHANGE AFTER INSPECTION
                elif 'nrca' or 'nrcb' in f:
                    sharp_lo, sharp_hi = 0.8, 1.4

            # Use IRAFStarFinder for detecting stars
            ### NOTE: minsep_fwhm > 5 is required for rejecting false sources around saturated stars
            iraffind = IRAFStarFinder(threshold=50*bgrms+bgavg, fwhm=2.0, minsep_fwhm=7,
                                      roundlo=round_lo, roundhi=round_hi, sharplo=sharp_lo, sharphi=sharp_hi)
            #### but for NIRCam CommissingPSF images, 0.7<sharp<1.4 works!
            iraf_extracted_sources = iraffind(data_cps)

            # Remove sources based on flux percentile (too faint or saturated sources)
            flux_min = np.percentile(iraf_extracted_sources['flux'], 10)
            flux_max = np.percentile(iraf_extracted_sources['flux'], 99)
            iraf_extracted_sources.remove_rows(np.where(iraf_extracted_sources['flux']<flux_min))
            iraf_extracted_sources.remove_rows(np.where(iraf_extracted_sources['flux']>flux_max))

            # Also remove sources based on fwhm
            iraf_extracted_sources.remove_rows(np.where(iraf_extracted_sources['fwhm']<fwhm_lo))
            iraf_extracted_sources.remove_rows(np.where(iraf_extracted_sources['fwhm']>fwhm_hi))

            print('Number of extracted sources after filtering: {} sources'.format(len(iraf_extracted_sources)))

            if parameters['use_epsf'] is True:
                size = 25
                hsize = (size-1)/2
                x = iraf_extracted_sources['xcentroid']
                y = iraf_extracted_sources['ycentroid']
                mask = ((x>hsize) & (x<(data_cps.shape[1]-1-hsize)) & (y>hsize) & (y<(data_cps.shape[0]-1-hsize)))
                stars_tbl = Table()
                stars_tbl['x'] = x[mask]
                stars_tbl['y'] = y[mask]
                print('Using {} stars to build epsf'.format(len(stars_tbl)))

                data_cps_bkgsub = data_cps.copy()
                data_cps_bkgsub -= bgavg
                nddata = NDData(data=data_cps_bkgsub)
                stars = extract_stars(nddata, stars_tbl, size=size)

                # Create plot showing all PSF stars
                if 1:
                    nrows = 10
                    ncols = 10
                    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20), squeeze=True)
                    ax = ax.ravel()
                    for i in range(nrows * ncols):
                        if i <= len(stars)-1:
                            norm = simple_norm(stars[i], 'log', percent=99.)
                            ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
                    plt.title('{} sample stars for epsf'.format(header_info['APERTURE']))
                    psf_plot_file = os.path.join(extracted_sources_dir,
                                                 '{}_sample_psfs.pdf'.format(os.path.basename(f).split('.')[0]))
                    plt.savefig(psf_plot_file)

                tic = time.perf_counter()
                epsf_builder = EPSFBuilder(oversampling=4, maxiters=3, progress_bar=False)
                print("Building ePSF ...")
                epsf, fitted_stars = epsf_builder(stars)
                toc = time.perf_counter()
                print("Time elapsed for building ePSF:", toc-tic)

                # Create ePSF plot
                if 1:
                    norm_epsf = simple_norm(epsf.data, 'log', percent=99.)
                    plt.figure()
                    plt.imshow(epsf.data, norm=norm_epsf, origin='lower', cmap='viridis')
                    plt.colorbar()
                    plt.title('{} epsf using {} stars'.format(header_info['APERTURE'], len(stars_tbl)))
                    epsf_plot_file = os.path.join(extracted_sources_dir,
                                                  '{}_epsf.pdf'.format(os.path.basename(f).split('.')[0]))
                    plt.savefig(epsf_plot_file)

                daogroup = DAOGroup(5.0*2.0)
                psf_model = epsf.copy()

                tic = time.perf_counter()
                photometry = IterativelySubtractedPSFPhotometry(finder=iraffind, group_maker=daogroup,
                                                                bkg_estimator=mmm_bkg, psf_model=psf_model,
                                                                fitter=LevMarLSQFitter(), niters=1,
                                                                fitshape=(11,11), aperture_radius=5)
                print('Performing source extraction and photometry ...')
                epsf_extracted_sources = photometry(data_cps)
                toc = time.perf_counter()
                print("Time elapsed for PSF photometry:", toc - tic)
                print('Final source extraction with epsf: {} sources'.format(len(epsf_extracted_sources)))

                epsf_extracted_sources['xcentroid'] = epsf_extracted_sources['x_fit']
                epsf_extracted_sources['ycentroid'] = epsf_extracted_sources['y_fit']
                extracted_sources = epsf_extracted_sources
                extracted_sources.write(extracted_sources_file, overwrite=True)

                if parameters['show_psfsubtracted_image']:
                    norm = simple_norm(data_cps, 'sqrt', percent=99.)
                    psf_subtracted_file = os.path.join(extracted_sources_dir,
                                          '{}_psfsubtracted_image.pdf'.format(os.path.basename(f).split('.')[0]))
                    diff = photometry.get_residual_image()
                    plt.figure()
                    ax1 = plt.subplot(1,2,1)
                    plt.xlabel("X [pix]")
                    plt.ylabel("Y [pix]")
                    ax1.imshow(data_cps, norm=norm, cmap='Greys')
                    ax2 = plt.subplot(1,2,2)
                    plt.xlabel("X [pix]")
                    plt.ylabel("Y [pix]")
                    ax2.imshow(diff, norm=norm, cmap='Greys')
                    plt.title('PSF subtracted image for {}'.format(os.path.basename(f)))
                    plt.savefig(psf_subtracted_file)

            else:

                extracted_sources = iraf_extracted_sources
                extracted_sources.write(extracted_sources_file, overwrite=True)

            positions = np.transpose((extracted_sources['xcentroid'], extracted_sources['ycentroid']))
            apertures = CircularAperture(positions, r=10)
            norm = simple_norm(data_cps, 'sqrt', percent=99.)
            extracted_plot_file = os.path.join(extracted_sources_dir,
                                  '{}_extracted_sources.pdf'.format(os.path.basename(f).split('.')[0]))
            plt.figure(figsize=(12,12))
            plt.xlabel("X [pix]")
            plt.ylabel("Y [pix]")
            plt.imshow(data_cps, norm=norm, cmap='Greys', origin='lower')
            apertures.plot(color='blue', lw=1.5, alpha=0.5)
            title_string = '{}: {} selected sources'.format(os.path.basename(f),
                                                             len(extracted_sources))
            plt.title(title_string)
            plt.tight_layout()
            plt.savefig(extracted_plot_file)
            if parameters['show_extracted_sources']: plt.show()
            plt.close()

        else:
            extracted_sources = Table.read(extracted_sources_file)

        print('Extracted {} sources from {}'.format(len(extracted_sources), f))
        impose_positive_flux = True
        if impose_positive_flux and parameters['use_epsf']:
            extracted_sources.remove_rows(np.where(extracted_sources['flux_fit']<0)[0])
            print('Only {} sources have positve flux'.format(len(extracted_sources)))

        astrometry_uncertainty_mas = 5

        if len(extracted_sources) > 0:
            # Cal images are in DMS coordinates which correspond to the SIAF Science (SCI) frame
            extracted_sources['x_SCI'], extracted_sources['y_SCI'] = extracted_sources['xcentroid'], extracted_sources['ycentroid']

            # For now, astrometric uncertainty defaults to 5 mas for each source.
            extracted_sources['sigma_x_mas'] = np.ones(len(extracted_sources)) * astrometry_uncertainty_mas
            extracted_sources['sigma_y_mas'] = np.ones(len(extracted_sources)) * astrometry_uncertainty_mas

        # transfer info to astropy table header
        for key, value in header_info.items():
            extracted_sources.meta[key] = value

        extracted_sources.meta['DATAFILE'] = os.path.basename(f)
        extracted_sources.meta['DATAPATH'] = os.path.dirname(f)
        extracted_sources.meta['EPOCH'] = header_info['epoch_isot']

        if 'FGS' in extracted_sources.meta['instrument_name']:
            out_file = os.path.join(standardized_data_dir,'FPA_data_{}_{}_{}.fits'.format(
#                                    extracted_sources.meta['telescope'],
                                    extracted_sources.meta['instrument_name'],
                                    extracted_sources.meta['subarray_name'],
#                                    extracted_sources.meta['EPOCH'].replace(':','-').replace('.','-'),
                                    extracted_sources.meta['DATAFILE'].split('.')[0]).replace('/',''))
        else:
            out_file = os.path.join(standardized_data_dir,'FPA_data_{}_{}_{}_{}_{}.fits'.format(
#                                    extracted_sources.meta['telescope'],
                                    extracted_sources.meta['instrument_name'],
                                    extracted_sources.meta['subarray_name'],
                                    extracted_sources.meta['instrument_filter'],
                                    extracted_sources.meta['instrument_pupil'],
#                                    extracted_sources.meta['EPOCH'].replace(':','-').replace('.','-'),
                                    extracted_sources.meta['DATAFILE'].split('.')[0]).replace('/',''))

        print('Writing {}'.format(out_file))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning, append=True)
            extracted_sources.write(out_file, overwrite=True)

    return im


def crossmatch_fpa_data(parameters):
    """

    Parameters
    ----------
    parameters

    Returns
    -------

    """
    print('\nCROSSMATCH OF FPA DATA WITH REFERENCE CATALOG')
    if (not os.path.isfile(parameters['pickle_file']) or parameters['overwrite']):

        fpa_data_files = glob.glob(os.path.join(parameters['standardized_data_dir'], '*.fits'))
        fpa_data_files.sort()
        verbose_figures = parameters['verbose_figures']
        if parameters['save_plot']:
            save_plot = 1
        else:
            save_plot = 0
        #save_plot = parameters['save_plot']
        plot_dir = parameters['plot_dir']
        siaf = parameters['siaf']
        verbose = parameters['verbose']
        idl_tel_method = parameters['idl_tel_method']
        reference_catalog = parameters['reference_catalog']
        xmatch_radius = parameters['xmatch_radius']
        rejection_level_sigma = parameters['rejection_level_sigma']
        restrict_analysis_to_these_apertures = parameters['restrict_analysis_to_these_apertures']

        observations = []
        # Work on ALL FPA_data.....fits files together - should I make them seprate and have these called multiple times from main script?
        for j, f in enumerate(fpa_data_files):

            print('=' * 40)
            fpa_data = Table.read(f)

            plt.close('all')
            print('Loading FPA observations in %s' % f)
            fpa_name_seed = os.path.basename(f).split('.')[0]

            instrument_name = fpa_data.meta['INSTRUME']
            aperture_name   = fpa_data.meta['SIAFAPER']
            data_name       = fpa_data.meta['DATAFILE'].split('.')[0]

            if (restrict_analysis_to_these_apertures is not None):
                if (aperture_name not in restrict_analysis_to_these_apertures):
                    continue

            aperture = copy.deepcopy(siaf[aperture_name])
            reference_aperture = copy.deepcopy(aperture)

            print('Detector: %s' % (aperture.InstrName))
            print('Aperture: %s' % (aperture.AperName))

            # Generate alignment observation
            obs = AlignmentObservation(aperture.observatory, aperture.InstrName)
            obs.aperture = aperture

            # Compute (v2, v3) coordinates of referece catalog (Hawk-I for JWST LMC field) stars using reference aperture (using boresight)
            attitude_ref = pysiaf.utils.rotations.attitude(0., 0.,
                                                           fpa_data.meta['pointing_ra_v1'],
                                                           fpa_data.meta['pointing_dec_v1'],
                                                           fpa_data.meta['pointing_pa_v3'])
            reference_catalog['v2_spherical_arcsec'], reference_catalog['v3_spherical_arcsec'] = \
                pysiaf.utils.rotations.getv2v3(attitude_ref, np.array(reference_catalog['ra']), np.array(reference_catalog['dec']))

            ### Why convert to RA, Dec space???
            reference_cat = SkyCoord(ra =np.array(reference_catalog['v2_spherical_arcsec']) * u.arcsec,
                                     dec=np.array(reference_catalog['v3_spherical_arcsec']) * u.arcsec)


            # define Hawk-I catalog specific to every aperture to allow for local tangent-plane projection
#            v2v3_reference = SkyCoord(ra =reference_aperture.V2Ref * u.arcsec,
#                                      dec=reference_aperture.V3Ref * u.arcsec)
#            selection_index = np.where(reference_cat.separation(v2v3_reference) < 3 * u.arcmin)[0]
#            selection_index = np.where((reference_cat.separation(v2v3_reference) < 3 * u.arcmin) &
#                                       (reference_catalog['j_2mass_extrapolated'] < 21))[0]

            # Extract reference catalog stars in and around the calibration aperture.
            corners = aperture.corners('tel')
            corners_v2 = corners[0]
            corners_v3 = corners[1]
            selection_index = np.where( (reference_catalog['v2_spherical_arcsec']>(np.min(corners_v2)-10)) &
                                        (reference_catalog['v2_spherical_arcsec']<(np.max(corners_v2)+10)) &
                                        (reference_catalog['v3_spherical_arcsec']>(np.min(corners_v3)-10)) &
                                        (reference_catalog['v3_spherical_arcsec']<(np.max(corners_v3)+10)) &
                                        (reference_catalog['j_2mass_extrapolated'] < 20.5) )[0]

            obs.reference_catalog = reference_catalog[selection_index]

            # Determine which reference stars fall into the aperture
            path_tel = aperture.path('tel')
            mask = path_tel.contains_points(np.array(obs.reference_catalog['v2_spherical_arcsec', 'v3_spherical_arcsec'].to_pandas()))

            # Now deal with the observed catalog
            star_catalog = fpa_data
            star_catalog['star_id'] = star_catalog['id']
            obs.star_catalog = star_catalog

            # SCI science frame (in pixels) -> IDL(=distortion-corrected) frame (in arcsec)
            obs.star_catalog['x_idl_arcsec'], obs.star_catalog['y_idl_arcsec'] = \
                aperture.sci_to_idl(np.array(obs.star_catalog['x_SCI']), np.array(obs.star_catalog['y_SCI']))
            # compute V2/V3: # IDL frame in degrees ->  V2/V3_tangent_plane in arcsec
            obs.star_catalog = compute_idl_to_tel_in_table(obs.star_catalog, aperture, method=idl_tel_method)


            ######################################################################################
            #
            # In order to do the cross-matching between reference and observed catalog positions,
            # we first need to figure out the rough offset. If this step is not performed,
            # the cross-match using pystortion.distortion.xmatch fails because there are
            # way too many sources in both of the catalogs to ensure a reliable match. [STS]
            #
            bright_index = np.where( (reference_catalog['v2_spherical_arcsec']>(np.min(corners_v2)-10)) &
                                     (reference_catalog['v2_spherical_arcsec']<(np.max(corners_v2)+10)) &
                                     (reference_catalog['v3_spherical_arcsec']>(np.min(corners_v3)-10)) &
                                     (reference_catalog['v3_spherical_arcsec']<(np.max(corners_v3)+10)) &
                                     (reference_catalog['j_2mass_extrapolated'] < 18.5) )[0] ### 18.5 works for FGS1 and 2, but may need to use 19 for NIRCam?

            print('Number of reference catalog stars used for initial offset match:',len(bright_index))

            ref_cat = reference_catalog[bright_index]

            ref_cat['TPx'], ref_cat['TPy'] = ref_cat['v2_spherical_arcsec'], ref_cat['v3_spherical_arcsec']

            obs_cat = obs.star_catalog
            obs_cat['TPx'], obs_cat['TPy'] = obs_cat['v2_spherical_arcsec'], obs_cat['v3_spherical_arcsec']
            obs_cat.sort('flux',reverse=True)
            # Adjust the number of sources to be used for initial matching below
            obs_cat_bright = obs_cat[:149] ##### If I get a "exceeded" error, try changing this number


#            offset_match = matchutils.TPMatch(searchrad=3, separation=1, use2dhist=True, tolerance=0.5) --> tolerance=0.5 causes an error for nrcb2
            offset_match = matchutils.TPMatch(searchrad=3, separation=1, use2dhist=True, tolerance=0.7)
            idx_ref, idx_obs = offset_match(ref_cat, obs_cat_bright)

            #
            # NOTE: Do NOT use median below. For some reason, median does not work in some cases
            #       and returns an empty array or value. [STS]
            #
            dv2 = ref_cat['TPx'][idx_ref] - obs_cat_bright['TPx'][idx_obs]
            offset_v2 = sigma_clip(dv2, sigma=3, maxiters=5, cenfunc='mean')
            avg_offset_v2 = np.mean(offset_v2)

            dv3 = ref_cat['TPy'][idx_ref] - obs_cat_bright['TPy'][idx_obs]
            offset_v3 = sigma_clip(dv3, sigma=3, maxiters=5, cenfunc='mean')
            avg_offset_v3 = np.mean(offset_v3)

            if 1:
                plt.figure(figsize=(10,6))
                plt.suptitle('Initial offsets between measured vs. catalog-based positions - {}'.format(data_name))
                ax1 = plt.subplot(1,2,1)
                ax1.set_xlabel('V2 offset [arcsec]')
                ax1.set_ylabel('N')
                ax1.hist(offset_v2, alpha=0.5, histtype='stepfilled')
                ax1.axvline(x=avg_offset_v2, color='r', linewidth=2)
                ax2 = plt.subplot(1,2,2)
                ax2.set_xlabel('V3 offset [arcsec]')
                ax2.set_ylabel('N')
                ax2.hist(offset_v3, alpha=0.5, histtype='stepfilled')
                ax2.axvline(x=avg_offset_v3, color='r', linewidth=2)
                plt.tight_layout()
                plt.show()
                plt.close()
                #1/0
            #
            # Inpsect figure above, and if you see something weird, the bright mag cutoff for reference catalog
            # and/or the number of sources used in the observed catalog can be changed.
            #
            ######################################################################################

            # if verbose_figures:
            if 1:
                plt.figure(figsize=(10,10))
                plt.plot(obs.star_catalog['v2_spherical_arcsec']+avg_offset_v2,
                         obs.star_catalog['v3_spherical_arcsec']+avg_offset_v3, 'ko', mfc='w', mew=1,
                         label = 'Measured positions')
                plt.plot(obs.reference_catalog['v2_spherical_arcsec'],
                         obs.reference_catalog['v3_spherical_arcsec'], 'b.',
                         label = 'Reference catalog positions')
                corners = aperture.corners('tel')
                corners_x = corners[0]
                corners_y = corners[1]
                buffer = 10
                plt.xlim(np.min(corners_x)-buffer, np.max(corners_x)+buffer)
                plt.ylim(np.min(corners_y)-buffer, np.max(corners_y)+buffer)
                plt.axis('equal')
                plt.title(fpa_name_seed)
                aperture.plot()
                plt.legend(loc='upper right')
                ax = plt.gca()
                # ax.invert_yaxis()
                if save_plot == 1:
                    figure_name = os.path.join(plot_dir, '%s_v2v3.pdf' % data_name)
                    plt.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)

                if verbose_figures: plt.show()
                plt.close()

                #1/0

            remove_multiple_matches = True
            retain_best_match = True

            # Offset (v2, v3) positions of observed catalog as found from the process above
            obs.star_catalog['v2_spherical_arcsec'] += avg_offset_v2
            obs.star_catalog['v3_spherical_arcsec'] += avg_offset_v3

            star_cat = SkyCoord(ra =np.array(obs.star_catalog['v2_spherical_arcsec']) * u.arcsec,
                                dec=np.array(obs.star_catalog['v3_spherical_arcsec']) * u.arcsec)

            reference_cat = SkyCoord(ra =np.array(obs.reference_catalog['v2_spherical_arcsec']) * u.arcsec,
                                     dec=np.array(obs.reference_catalog['v3_spherical_arcsec']) * u.arcsec)

#            plt.close(all)

            # # tackle wrapping or RA coordinates
            # if np.ptp(star_cat.ra).value > 350:
            #     star_cat.ra[np.where(star_cat.ra > 180 * u.deg)[0]] -= 360 * u.deg
            # if np.ptp(reference_cat.ra).value > 350:
            #     reference_cat.ra[np.where(reference_cat.ra > 180 * u.deg)[0]] -= 360 * u.deg

            #xmatch_radius = copy.deepcopy(xmatch_radius)

            ###
            ### TBD: replace routine below with second run of matchutils.TPMatch
            ###
            #
            # *** TBD: If I use below, make sure to remove the +=avg_offset_v2, v3 above and -= below ***
            # obs.star_catalog['TPx'], obs.star_catalog['TPy'] = obs.star_catalog['v2_spherical_arcsec'], obs.star_catalog['v3_spherical_arcsec']
            # obs.reference_catalog['TPx'], obs.reference_catalog['TPy'] = obs.reference_catalog['v2_spherical_arcsec'], obs.reference_catalog['v3_spherical_arcsec']
            #
            #match = matchutils.TPMatch(searchrad=0.5, separation=1, tolerance=0.5,
            #                           xoffset=avg_offset_v2, yoffset=avg_offset_v3)
            #idx_refcat, idx_obscat = match(obs.reference_catalog, obs.star_catalog)
            #
            # TBD: Add plots here
            #
            # DON'T NEED PLOT BELOW - ALREADY ABOVE
            #if 1:
            #    plt.fits(figsize=(7,7), facecolor='w', edgecolor='k')
            #    plt.plot(reference_cat.ra, reference_cat.dec, 'bo', label='Reference Catalog', mfc=None, zorder=-40)
            #    plt.plot(star_cat.ra, star_cat.dec, 'r.', label='Reference Catalog', mfc=None, zorder=-40)
            #
            #


            idx_reference_cat, idx_star_cat, d2d, d3d, diff_ra, diff_dec = crossmatch.xmatch(
                reference_cat, star_cat, xmatch_radius, rejection_level_sigma, verbose=verbose,
                verbose_figures=verbose_figures, saveplot=save_plot, out_dir=plot_dir,
                name_seed=fpa_name_seed, retain_best_match=retain_best_match,
                remove_multiple_matches=remove_multiple_matches)

            print(
                '{:d} measured stars, {:d} reference catalog stars in the aperture, {:d} matches.'.format(
                    len(obs.star_catalog), np.sum(mask), len(idx_reference_cat)))


            obs.number_of_measured_stars = len(obs.star_catalog)
            obs.number_of_reference_stars = np.sum(mask)
            obs.number_of_matched_stars = len(idx_reference_cat)

            obs.reference_catalog_matched = obs.reference_catalog[idx_reference_cat]

            # Subtract back the offset applied above
            obs.star_catalog['v2_spherical_arcsec'] -= avg_offset_v2
            obs.star_catalog['v3_spherical_arcsec'] -= avg_offset_v3

            obs.star_catalog_matched = obs.star_catalog[idx_star_cat]
            obs.reference_catalog_matched['star_id'] = obs.star_catalog_matched['star_id']

            # save space in pickle, speed up
            obs.reference_catalog = []
            obs.star_catalog = []
            #obs.reference_reference_catalog = [] # NOT USED!

            obs.siaf_aperture_name = aperture_name
            obs.fpa_data = fpa_data
            obs.fpa_name_seed = fpa_name_seed

            # dictionary that defines the names of columns in the star/reference_catalog for use later on
            fieldname_dict = {}
            fieldname_dict['star_catalog'] = {}  # observed
            fieldname_dict['reference_catalog'] = {}  # Gaia

            if idl_tel_method == 'spherical':
                fieldname_dict['reference_catalog']['position_1'] = 'v2_spherical_arcsec'
                fieldname_dict['reference_catalog']['position_2'] = 'v3_spherical_arcsec'
            else:
                fieldname_dict['reference_catalog']['position_1'] = 'v2_tangent_arcsec'
                fieldname_dict['reference_catalog']['position_2'] = 'v3_tangent_arcsec'

            if 'Name: J/A+A/563/A80/jwstcf' in reference_catalog.meta['comments']:
                reference_catalog_identifier = 'ID'  # HAWK-I
                fieldname_dict['reference_catalog']['sigma_position_1'] = 'ra_error_mas'
                fieldname_dict['reference_catalog']['sigma_position_2'] = 'dec_error_mas'
            else:
                reference_catalog_identifier = 'source_id'  # Gaia
                fieldname_dict['reference_catalog']['sigma_position_1'] = 'ra_error'
                fieldname_dict['reference_catalog']['sigma_position_2'] = 'dec_error'
            fieldname_dict['reference_catalog']['identifier'] = reference_catalog_identifier
            fieldname_dict['reference_catalog']['position_unit'] = u.arcsecond
            fieldname_dict['reference_catalog']['sigma_position_unit'] = u.milliarcsecond

            if idl_tel_method == 'spherical':
                fieldname_dict['star_catalog']['position_1'] = 'v2_spherical_arcsec'
                fieldname_dict['star_catalog']['position_2'] = 'v3_spherical_arcsec'
            else:
                fieldname_dict['star_catalog']['position_1'] = 'v2_tangent_arcsec'
                fieldname_dict['star_catalog']['position_2'] = 'v3_tangent_arcsec'

            fieldname_dict['star_catalog']['sigma_position_1'] = 'sigma_x_mas'
            fieldname_dict['star_catalog']['sigma_position_2'] = 'sigma_y_mas'
            fieldname_dict['star_catalog']['identifier'] = 'star_id'
            fieldname_dict['star_catalog']['position_unit'] = u.arcsecond
            fieldname_dict['star_catalog']['sigma_position_unit'] = u.milliarcsecond

            obs.fieldname_dict = fieldname_dict
            observations.append(obs)

        pickle.dump((observations), open(parameters['pickle_file'], "wb"))
    else:
        observations = pickle.load(open(parameters['pickle_file'], "rb"))
        print('Loaded pickled file {}'.format(parameters['pickle_file']))

    return observations



def correct_dva(obs_collection, parameters):
    """Correct for effects of differential velocity aberration. This routine provides the necessary input to the DVA
    calculations (as attributes to the aperture object). DVA corrections are performed within the aperture methods when
    necessary.

    Parameters
    ----------
    obs_collection

    Returns
    -------

    """
    print('\nCORRECT FOR DIFFERENTIAL VELOCITY ABERRATION')

    dva_dir = parameters['dva_dir']
    dva_source_dir = parameters['dva_source_dir']
    verbose = parameters['verbose']

    for group_id in np.unique(obs_collection.T['group_id']):
        obs_indexes = np.where((obs_collection.T['group_id'] == group_id))[0]
        # obs_collection.T[obs_indexes].pprint()

        superfgs_observation_index = \
            np.where((obs_collection.T['group_id'] == group_id) & (
            obs_collection.T['INSTRUME'] == 'SUPERFGS'))[0]
        superfgs_obs = obs_collection.observations[superfgs_observation_index][0]

        camera_observation_index = \
            np.where((obs_collection.T['group_id'] == group_id) & (
            obs_collection.T['INSTRUME'] != 'SUPERFGS'))[0]

        fgs_exposure_midtimes = np.mean(np.vstack(
            (superfgs_obs.fpa_data['EXPSTART'].data, superfgs_obs.fpa_data['EXPEND'].data)), axis=0)

        # exclude HST FGS because it has already been corrected
        for i in camera_observation_index:

            camera_obs = obs_collection.observations[i]
            camera_obs_name_seed = '{}_{}_{}_{}_{}'.format(camera_obs.fpa_data.meta['TELESCOP'],
                                                           camera_obs.fpa_data.meta['INSTRUME'],
                                                           camera_obs.fpa_data.meta['APERTURE'],
                                                           camera_obs.fpa_data.meta['EPOCH'],
                                                           camera_obs.fpa_data.meta['DATAFILE'].split('.')[0]).replace(':', '-')
            dva_filename = os.path.join(dva_dir, 'DVA_data_{}.txt'.format(camera_obs_name_seed))
            dva_file = open(dva_filename, 'w')
            # dva_file = sys.stdout

            camera_exposure_midtime = np.mean(
                [camera_obs.fpa_data.meta['EXPSTART'], camera_obs.fpa_data.meta['EXPEND']])
            matching_fgs_exposure_index = np.argmin(
                np.abs(camera_exposure_midtime - fgs_exposure_midtimes))
            if verbose:
                print('Camera-FGS Match found with delta_time = {:2.3f} min'.format(np.abs(
                    camera_exposure_midtime - fgs_exposure_midtimes[
                        matching_fgs_exposure_index]) * 24 * 60.))
                print('Writing parameter file for DVA code')
            for key in 'PRIMESI V2Ref V3Ref FGSOFFV2 FGSOFFV3 RA_V1 DEC_V1 PA_V3 POSTNSTX POSTNSTY POSTNSTZ VELOCSTX VELOCSTY VELOCSTZ EXPSTART'.split():
                if key in 'V2Ref V3Ref'.split():
                    value = getattr(superfgs_obs.aperture, key)
                elif key == 'PRIMESI':
                    value = np.int(superfgs_obs.fpa_data[key][matching_fgs_exposure_index][-1])
                elif key in 'FGSOFFV2 FGSOFFV3'.split():
                    value = superfgs_obs.fpa_data[key][matching_fgs_exposure_index]
                elif key == 'EXPSTART':
                    # Scale of EXPSTART seems to be UTC, see http://www.stsci.edu/ftp/documents/calibration/podps.dict
                    fgs_time = Time(camera_obs.fpa_data.meta[key], format='mjd', scale='utc')
                    value = fgs_time.yday.replace(':', ' ')
                else:
                    value = camera_obs.fpa_data.meta[key]

                print('{:<30} {}'.format(value, key), file=dva_file)
            dva_file.close()
            camera_obs.aperture._correct_dva = True
            camera_obs.aperture._dva_parameters = {'parameter_file': dva_filename,
                                                   'dva_source_dir': dva_source_dir}

            obs_collection.observations[i] = camera_obs
            inspect_ref_vs_obs_positions = True

            if inspect_ref_vs_obs_positions:
                v2, v3 = camera_obs.aperture.correct_for_dva(
                    np.array(camera_obs.star_catalog_matched['v2_spherical_arcsec']),
                    np.array(camera_obs.star_catalog_matched['v3_spherical_arcsec']))

                v2v3_data_file = os.path.join(dva_dir, 'v2v3_data_{}_measured.txt'.format(
                    camera_obs_name_seed))
                camera_obs.star_catalog_matched['v2_spherical_arcsec', 'v3_spherical_arcsec'].write(
                    v2v3_data_file,
                    format='ascii.fixed_width_no_header',
                    delimiter=' ',
                    bookend=False,
                    overwrite=True)

                v2v3_corrected_file = v2v3_data_file.replace('_measured', '_corrected')
                system_command = '{} {} {} {}'.format(os.path.join(dva_source_dir, 'compute-DVA.e'),
                                                      dva_filename,
                                                      v2v3_data_file, v2v3_corrected_file)
                print('Running system command \n{}'.format(system_command))
                subprocess.call(system_command, shell=True)

                v2v3_corrected = Table.read(v2v3_corrected_file, format='ascii.no_header',
                                            names=('v2_original', 'v3_original', 'v2_corrected',
                                                   'v3_corrected'))
                1 / 0

                # interpolate ephmeris
                # fgs_time = Time(superfgs_obs.fpa_data[key][matching_fgs_exposure_index], format='mjd', scale='tdb')
                # start_time = Time(superfgs_obs.fpa_data[key][matching_fgs_exposure_index]-2001, format='mjd')
                # stop_time  = Time(superfgs_obs.fpa_data[key][matching_fgs_exposure_index]-1999, format='mjd')
                start_time = fgs_time - TimeDelta(1, format='jd')
                stop_time = fgs_time + TimeDelta(1, format='jd')

                # center = 'g@399'# Earth
                # center = '500@399'
                # target = '0' # SSB
                center = '500@0'  # SSB
                target = '399'  # Earth
                # target = '500@399'
                e = pystrometry.get_ephemeris(center=center, target=target, start_time=start_time,
                                              stop_time=stop_time,
                                              step_size='1h', verbose=True, out_dir=dva_dir,
                                              vector_table_output_type=2,
                                              output_units='KM-S', overwrite=True,
                                              reference_plane='FRAME')

                ip_values = []
                for colname in ['X', 'Y', 'Z', 'VX', 'VY', 'VZ']:
                    ip_fun = scipy.interpolate.interp1d(np.array(e['JDTDB']), np.array(e[colname]),
                                                        kind='linear', copy=True, bounds_error=True,
                                                        fill_value=np.nan)
                    # http://docs.astropy.org/en/stable/time/#id6
                    ip_val = ip_fun(fgs_time.tdb.jd)
                    ip_values.append(ip_val)

                e.add_row(np.hstack(([fgs_time.tdb.jd, 'N/A'], np.array(ip_values))))
                e[[-1]].pprint()
    return obs_collection
