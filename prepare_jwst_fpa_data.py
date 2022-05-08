import os
import copy
import glob
import numpy as np
import pickle
import sys
import time
import warnings
import subprocess
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import astropy.units as u
import astropy.constants as const

from collections import OrderedDict
from astropy.table import Table, vstack
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyWarning
from astropy.stats import sigma_clip
from astropy.visualization import simple_norm
from astropy.nddata import NDData
from astropy.modeling.fitting import LevMarLSQFitter

from jwst import datamodels
from photutils import IRAFStarFinder, DAOStarFinder
from photutils import CircularAperture, EPSFBuilder
from photutils.psf import DAOGroup, extract_stars, IterativelySubtractedPSFPhotometry, DAOPhotPSFPhotometry
from photutils.background import MMMBackground, MADStdBackgroundRMS, SExtractorBackground
from photutils.centroids import centroid_sources, centroid_2dg, centroid_com

import pysiaf
import pysiaf.utils.rotations as rotations
from pystortion import crossmatch
from scipy.spatial import cKDTree

from alignment import AlignmentObservation, compute_idl_to_tel_in_table

# Below was added to figure out the initial offset between reference and observed catalog positions. [STS]
from tweakwcs import matchutils


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

    """Generate standardized focal plane alignment (fpa) data
       based on JWST camera image.
    """

    save_plot = parameters['save_plot']

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

        #for attribute in ''

        header_info['INSTRUME'] = header_info['instrument_name']
        header_info['SIAFAPER'] = header_info['aperture_name']

        instrument_name = getattr(im.meta.instrument, 'name')
        instrument_detector = getattr(im.meta.instrument, 'detector')
        instrument_filter = getattr(im.meta.instrument, 'filter')
        instrument_pupil = getattr(im.meta.instrument, 'pupil')

        # DVA correction related
        va_scale   = getattr(im.meta.velocity_aberration, 'scale_factor')
        va_ra_ref  = getattr(im.meta.velocity_aberration, 'va_ra_ref')
        va_dec_ref = getattr(im.meta.velocity_aberration, 'va_dec_ref')
        scvel_x = getattr(im.meta.ephemeris, 'velocity_x_bary')
        scvel_y = getattr(im.meta.ephemeris, 'velocity_y_bary')
        scvel_z = getattr(im.meta.ephemeris, 'velocity_z_bary')

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

        # TBD: Need to remove making yet another directory
        #extracted_sources_dir = os.path.join(standardized_data_dir, 'extraction')
        #if os.path.isdir(extracted_sources_dir) is False:
        #    os.makedirs(extracted_sources_dir)
        extracted_sources_file = os.path.join(standardized_data_dir, #extracted_sources_dir,
                                              '{}_extracted_sources.fits'
                                              .format(os.path.basename(f).split('.')[0]))

        mask_extreme_slope_values = False
        parameters['maximum_slope_value'] = 1000.



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


            # Detection parameters
            # sharp_lo, sharp_hi = parameters['sharp_lo'], parameters['sharp_hi']
            # round_lo, round_hi = parameters['round_lo'], parameters['round_hi']
            # sigma_factor = parameters['sigma_factor']
            # fwhm = parameters['fwhm']
            # fwhm_lo, fwhm_hi = parameters['fwhm_lo'], parameters['fwhm_hi']
            # flux_percent_lo, flux_percent_hi = \
            #       parameters['flux_percent_lo'], parameters['flux_percent_hi']
            # minsep_fwhm = parameters['minsep_fwhm']



            # Default parameters that generally works for NIRCam/NIRISS images
            bkg_factor = 1
            sigma_factor = 10
            round_lo, round_hi = 0.0, 0.6
            sharp_lo, sharp_hi = 0.3, 1.4
            fwhm_lo, fwhm_hi   = 1.0, 20.0
            fwhm = 2.0
            minsep_fwhm = 7 # NOTE: minsep_fwhm>5 to reject artifacts around saturated stars
            flux_percent_lo, flux_percent_hi = 2, 99.5
            flux_min = 50

            # if 'sharp_lo' in parameters:
            #    sharp_lo = parameters['sharp_lo']

            ###
            ### TBD1: Relocate params below to config parts/files
            ###
            # Use different criteria for selecting good stars
            if parameters['nominalpsf']:

################################################################################
### FOR NIS-11 and NIS-13
################################################################################
                if instrument_name == 'NIRISS':

                    sigma_factor = 3
                    sharp_lo, sharp_hi = 0.7, 1.3
                    round_lo, round_hi = 0.0, 0.4

                    fff = instrument_filter+instrument_pupil
                    if fff.find('F090W' or 'F115W')>-1:
                        fwhm = 1.4
                    elif fff.find('F140M' or 'F150W' or 'F158M' or 'F200W' or 'F277W')>-1:
                        fwhm = 1.5
                    elif fff.find('F356W')>-1:
                        fwhm = 1.6
                    elif fff.find('F380M')>-1:
                        fwhm = 1.7
                    elif fff.find('F430M' or 'F444W' or 'F480M')>-1:
                        fwhm = 1.8
                    else:
                        fwhm = 1.55
################################################################################

################################################################################
### FOR FGS-11
################################################################################

                elif instrument_name == 'FGS':
                    sigma_factor = 3
                    fwhm = 1.55
                    sharp_lo, sharp_hi = 0.7, 1.4
                    round_lo, round_hi = 0.0, 0.4
                    flux_min = 100

################################################################################

                elif instrument_name == 'NIRCAM':
                    sharp_lo, sharp_hi = 0.6, 1.4

                elif instrument_name == 'MIRI':
                    if 'F770W' in instrument_filter:
                        fwhm               = 1.8
                        sharp_lo, sharp_hi = 0.9, 1.2
                        fwhm_lo, fwhm_hi   = 1.5, 2.2
                        bkg_factor         = 0.2
                        sigma_factor       = 3
                        flux_percent_lo, flux_percent_hi = 1, 99.9
                    if 'F560W' in instrument_filter:
                        fwhm               = 1.6
                        sharp_lo, sharp_hi = 0.85, 1.2
                        fwhm_lo, fwhm_hi   = 1.3, 2.0
                        bkg_factor         = 0.5
                        sigma_factor       = 3

                elif instrument_name == 'NIRSPEC':
                    sharp_lo, sharp_hi = 0.6, 0.8
                    round_lo, round_hi = 0.0, 0.3
                    fwhm_lo, fwhm_hi   = 1.0, 1.75
            else:
                ###
                ### For OTE commissioning, tweak the params below after finding
                ### the correct ranges by runnin the photometry notebook.
                ###

                # If using Commissioning (non-phased) PSF models
                if instrument_name == 'NIRISS':
                    sharp_lo, sharp_hi = 0.6, 1.4
                    fwhm_lo, fwhm_hi   = 1.4, 2.4

                elif instrument_name == 'FGS':
                    sigma_factor = 100
                    minsep_fwhm = 7
                    sharp_lo, sharp_hi = 0.3, 1.4
                    round_lo, round_hi = 0.0, 0.6
                    flux_percent_lo, flux_percent_hi = 10, 99
                    fwhm = 1.5

                # Below is for the un-phased images (obs1, 2) in OTE-10

                elif instrument_name == 'NIRCAM':
                    sigma_factor = 3
                    minsep_fwhm = 2.5
                    sharp_lo, sharp_hi = 0.5, 0.7
                    round_lo, round_hi = 0.0, 0.2
                    flux_percent_lo, flux_percent_hi = 2, 99
                    if 'F200W' in instrument_filter:
                        fwhm = 10
                    elif 'F356W' in instrument_filter:
                        fwhm = 8
                    elif 'F090W' in instrument_filter:
                        fwhm = 5.5
                    elif 'F277W' in instrument_filter:
                        fwhm = 6.5
                    else:
                        fwhm = 3

                elif instrument_name == 'MIRI':
                    sharl_lo, sharp_hi = 0.5, 1.0
                    fwhm_lo, fwhm_hi   = 1.5, 2.2
                    sigma_factor       = 3

                elif instrument_name == 'NIRSPEC':
                    sharp_lo, sharp_hi = 0.5, 0.8
                    round_lo, round_hi = 0.0, 0.3
                    fwhm_lo, fwhm_hi   = 1.0, 1.75


            # Use IRAFStarFinder for source detection

            bkgrms = MADStdBackgroundRMS()
            if instrument_detector == 'MIRIMAGE':
                mmm_bkg = SExtractorBackground()
            else:
                mmm_bkg = MMMBackground()
            bgrms = bkgrms(data_cps)
            bgavg = mmm_bkg(data_cps)

            iraffind = IRAFStarFinder(threshold=sigma_factor*bgrms+bkg_factor*bgavg,
                                      fwhm=fwhm, minsep_fwhm=minsep_fwhm,
                                      roundlo=round_lo, roundhi=round_hi,
                                      sharplo=sharp_lo, sharphi=sharp_hi)

            # Create default mask with all False values
            datamask = np.zeros(data_cps.shape, dtype=bool) # This creates an array with all False

            # Mask the left (for NRS1) and right regions (for NRS2) for NIRSpec
            if instrument_detector == 'NRS1':
                datamask[:, :1023] = True   # Mask everything on the left side
            elif instrument_detector == 'NRS2':
                datamask[:, 1024:] = True   # Mask everything on the right side

            # Mask the left for MIRI
            if instrument_detector == "MIRIMAGE":
                datamask[:, :350] = True # Mask everyhing in the range x=1-350

            iraf_extracted_sources = iraffind(data_cps, mask=datamask)

            # Perform some basic filtering

            # Remove sources based on flux percentile
            #flux_min = np.percentile(iraf_extracted_sources['flux'], flux_percent_lo)
            flux_max = np.percentile(iraf_extracted_sources['flux'], flux_percent_hi)
            iraf_extracted_sources.remove_rows(np.where(iraf_extracted_sources['flux']<flux_min))
            iraf_extracted_sources.remove_rows(np.where(iraf_extracted_sources['flux']>flux_max))

            #
            # Use a FWHM-based filtering if you get LOTS of bad sources not filtered by above
            #
            #iraf_extracted_sources.remove_rows(np.where(iraf_extracted_sources['fwhm']<fwhm_lo))
            #iraf_extracted_sources.remove_rows(np.where(iraf_extracted_sources['fwhm']>fwhm_hi))

            # Now improve the positions by re-running centroiding algorithm if necessary.

            if parameters['use_centroid_2dg']:
                print("=======================================================")
                print(" Re-centroiding sources using centroid_2dg in progress ")
                print("=======================================================")
                intx = np.round(iraf_extracted_sources['xcentroid'].data)
                inty = np.round(iraf_extracted_sources['ycentroid'].data)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="The fit may be unsuccessful")
                    newx, newy = centroid_sources(data_cps, intx, inty, box_size=9,
                                                  centroid_func=centroid_2dg)
                iraf_extracted_sources['xcentroid'] = newx
                iraf_extracted_sources['ycentroid'] = newy

            print('Number of extracted sources after filtering: {} sources'.format(len(iraf_extracted_sources)))

            #
            # Below is for the ePSF positional measurement.
            # But, don't use this unless for special test case.
            # There's a separate epsf_astrometry.py code for doing this part,
            # that outputs a _FPA_data.fits files. That file should be copied
            # to the fpa_data directory, and then the following flags should be set:
            # generate_standardized_fpa_data = False
            # overwrite_source_extraction    = False
            #
            if parameters['use_epsf'] is True:
                psf_stars = iraf_extracted_sources
                flux_min = np.percentile(psf_stars['flux'], 25)
                flux_max = np.percentile(psf_stars['flux'], 95)
                psf_stars.remove_rows(np.where(psf_stars['flux']<flux_min))
                psf_stars.remove_rows(np.where(psf_stars['flux']>flux_max))

                size = 25
                hsize = (size-1)/2
                x = psf_stars['xcentroid']
                y = psf_stars['ycentroid']
                mask = ((x>hsize) & (x<(data_cps.shape[1]-1-hsize)) &
                        (y>hsize) & (y<(data_cps.shape[0]-1-hsize)))
                stars_tbl = Table()
                stars_tbl['x'] = x[mask]
                stars_tbl['y'] = y[mask]

                print('Using {} stars to build epsf'.format(len(stars_tbl)))

                data_cps_bkgsub = data_cps.copy()
                data_cps_bkgsub -= bgavg
                nddata = NDData(data=data_cps_bkgsub)
                stars = extract_stars(nddata, stars_tbl, size=size)

                #
                # Figure - PSF stars
                #
                plt.rc('font', family='serif')
                nrows = 10
                ncols = 10
                fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20), squeeze=True)
                ax = ax.ravel()
                for i in range(nrows * ncols):
                    if i <= len(stars)-1:
                        norm = simple_norm(stars[i], 'log', percent=99.)
                        ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
                plt.title('{} sample stars for epsf'.format(header_info['APERTURE']))
                plt.tight_layout()
                if save_plot:
                    figname = os.path.join(extracted_sources_dir,
                              '{}_sample_psfs.pdf'.format(os.path.basename(f).split('.')[0]))
                    plt.savefig(figname)
                if parameters['show_extracted_sources']:
                    plt.show()

                #
                # Timer for ePSF construction
                #
                tic = time.perf_counter()
                epsf_builder = EPSFBuilder(oversampling=4, maxiters=3, progress_bar=False)
                print("Building ePSF ...")
                epsf, fitted_stars = epsf_builder(stars)
                toc = time.perf_counter()
                print("Time elapsed for building ePSF:", toc-tic)
                print()

                #
                # Figure - ePSF plot
                #
                plt.figure(figsize=(10,10))
                norm_epsf = simple_norm(epsf.data, 'log', percent=99.)
                plt.rc('font', family='serif')
                plt.imshow(epsf.data, norm=norm_epsf, origin='lower', cmap='viridis')
                plt.colorbar()
                plt.title('{} epsf using {} stars'.format(header_info['APERTURE'], len(stars_tbl)))
                plt.tight_layout()
                if save_plot:
                    figname = os.path.join(extracted_sources_dir,
                              '{}_epsf.pdf'.format(os.path.basename(f).split('.')[0]))
                    plt.savefig(figname)
                if parameters['show_extracted_sources']:
                    plt.show()

                fitter = LevMarLSQFitter()
                psf_model = epsf.copy()
                psf_phot = DAOPhotPSFPhotometry(crit_separation=1.0, threshold=4*std+bkg,
                                                fwhm=fwhm, psf_model=psf_model, fitshape=5,
                                                fitter=fitter, niters=3, aperture_radius=2,
                                                extra_output_cols=('sharpness','roundness'))

                #photometry = IterativelySubtractedPSFPhotometry(finder=iraffind, group_maker=daogroup,
                #                                                bkg_estimator=mmm_bkg, psf_model=psf_model,
                #                                                fitter=LevMarLSQFitter(), niters=1,
                #                                                fitshape=(11,11), aperture_radius=5)

                print("Performing DAOPhotPSFPhotometry using ePSF")
                tic = time.perf_counter()
                psf_extracted_sources = psf_phot(data_cps)
                toc = time.perf_counter()
                print("Time elapsed for PSF photometry:", toc - tic)
                print('Final source extraction with epsf: {} sources'.format(len(epsf_extracted_sources)))
                print()

                psf_extracted_sources['xcentroid'] = psf_extracted_sources['x_fit']
                psf_extracted_sources['ycentroid'] = psf_extracted_sources['y_fit']
                extracted_sources = psf_extracted_sources
                extracted_sources.write(extracted_sources_file, overwrite=True)

                plt.rc('font', family='serif')
                plt.figure(figsize=(40,20))
                norm = simple_norm(data_cps, 'sqrt', percent=95.)
                diff = psf_phot.get_residual_image()
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
                plt.tight_layout()
                if save_plot:
                    figname = os.path.join(extracted_sources_dir,
                                           '{}_psfsubtracted_image.pdf'
                                           .format(os.path.basename(f).split('.')[0]))
                    plt.savefig(figname)
                if parameters['show_psfsubtracted_image']:
                    plt.show()

            else:

                extracted_sources = iraf_extracted_sources
                extracted_sources.write(extracted_sources_file, overwrite=True)

            positions = np.transpose((extracted_sources['xcentroid'], extracted_sources['ycentroid']))
            apertures = CircularAperture(positions, r=5)
            norm = simple_norm(data_cps, 'sqrt', percent=99.)

            plt.rc('font', family='serif')
            plt.figure(figsize=(12,12))
            plt.xlabel("X [pix]")
            plt.ylabel("Y [pix]")
            plt.imshow(data_cps, norm=norm, cmap='Greys', origin='lower')
            apertures.plot(color='blue', lw=1.5, alpha=0.5)
            title_string = '{}: {} selected sources'.format(os.path.basename(f),
                                                            len(extracted_sources))
            plt.title(title_string)
            plt.tight_layout()
            if save_plot:
                figname = os.path.join(standardized_data_dir,
                                       '{}_extracted_sources.pdf'
                                       .format(os.path.basename(f).split('.')[0]))
                plt.savefig(figname)
            if parameters['show_extracted_sources']:
                plt.show()
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

        out_file = os.path.join(standardized_data_dir, '{}_FPA_data.fits'.format(
                                extracted_sources.meta['DATAFILE'].split('.')[0]))

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

        if parameters['fpa_file_name'] is None:
            fpa_file_name = '*_FPA_data.fits'
        else:
            if not os.path.isfile(os.path.join(parameters['standardized_data_dir'],
                                               parameters['fpa_file_name'])):
                sys.exit('Standardized FPA file not found.')
            fpa_file_name = parameters['fpa_file_name']

        fpa_data_files = glob.glob(os.path.join(parameters['standardized_data_dir'], fpa_file_name))
        fpa_data_files.sort()

        verbose_figures = parameters['verbose_figures']
        if parameters['save_plot']:
            save_plot = 1
        else:
            save_plot = 0
        #save_plot = parameters['save_plot']
        data_dir = parameters['data_dir']
        plot_dir = parameters['plot_dir']
        siaf = parameters['siaf']
        verbose = parameters['verbose']
        idl_tel_method = parameters['idl_tel_method']
        reference_catalog = parameters['reference_catalog']
        xmatch_radius = parameters['xmatch_radius']
        rejection_level_sigma = parameters['rejection_level_sigma']
        restrict_analysis_to_these_apertures = parameters['restrict_analysis_to_these_apertures']
        use_default_siaf_distortion = parameters['use_default_siaf_distortion']
        correct_dva = parameters['correct_dva']
        # TBD: REPLACE BELOW WITH TAKING INPUT FROM HEADER (METADATA)
        #if correct_dva:
        #    sc_velocity = parameters['sc_velocity']

        # Reference catalog magnitude range (for improved crossmatching)
        xmatch_refcat_mag_range = parameters['xmatch_refcat_mag_range']

        # Set up return parameter
        observations = []

        # Work on ALL FPA_data.....fits files together - should I make them separate and have these called multiple times from main script?
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
############
############ GLORIOUS ADDITION OF USING CUSTOM DISTORTION FILE

            # Use the default SIAF if use_default_siaf_distortion is True
            aperture = copy.deepcopy(siaf[aperture_name])

            # But if using a custom distortion file, override coeffs in aperture
            if not parameters['use_default_siaf_distortion']:
                file_names = os.listdir(data_dir)
                try:
                    distortion_coefficients_file = \
                        [s for s in file_names if aperture_name.lower() in s][0]
                    try:
                        aperture.set_distortion_coefficients_from_file(os.path.join(data_dir, distortion_coefficients_file))
                        print()
                        print('*'*70)
                        print('Custom distortion coefficients file loaded: ',distortion_coefficients_file)
                        print('*'*70)
                        print()
                    except:
                        sys.exit('Distortion coefficients file is not in correct format!')
                except:
                    print()
                    print('*'*70)
                    print('WARNING!!! Did not find matching distortion coefficient file for', aperture_name.lower())
                    print('WARNING!!! Using default distortion coefficients.')
                    print('*'*70)
                    print()
                    #sys.exit('WARNING!! distortion coefficients file must be in the same directory as the data file!')
############
############

            reference_aperture = copy.deepcopy(aperture) ### ---> This does NOT get used anywhere. Why is it here?

            print('Detector: %s' % (aperture.InstrName))
            print('Aperture: %s' % (aperture.AperName))

            # Generate alignment observation
            obs = AlignmentObservation(aperture.InstrName)
            obs.aperture = aperture

            # Compute (v2, v3) coordinates of referece catalog stars using reference aperture (using boresight)
            attitude_ref = rotations.attitude(0., 0.,
                                             fpa_data.meta['pointing_ra_v1'],
                                             fpa_data.meta['pointing_dec_v1'],
                                             fpa_data.meta['pointing_pa_v3'])
            ###!!!
            ###!!! If dva correction is required, below would be the place to apply it.
            ###!!! Simplest way to do this is to apply dva correction to RA, DEC
            ###!!!
            #if correct_dva:
            #    ra_corrected, dec_corrected = \
            #             va_correct_radec(reference_catalog['ra'].data,
            #                              reference_catalog['dec'].data,
            #                              sc_velocity[0],
            #                              sc_velocity[1],
            #                              sc_velocity[2])
            #
            #    reference_catalog['v2_spherical_arcsec'], \
            #    reference_catalog['v3_spherical_arcsec'] = \
            #                    rotations.getv2v3(attitude_ref,
            #                                      ra_corrected, dec_corrected)
            #else:
            reference_catalog['v2_spherical_arcsec'], \
            reference_catalog['v3_spherical_arcsec'] = \
                        rotations.getv2v3(attitude_ref,
                                          np.array(reference_catalog['ra']),
                                          np.array(reference_catalog['dec']))

            reference_cat = SkyCoord(ra =np.array(reference_catalog['v2_spherical_arcsec']) * u.arcsec,
                                     dec=np.array(reference_catalog['v3_spherical_arcsec']) * u.arcsec)


            # Extract reference catalog stars in and around the calibration aperture.
            corners = aperture.corners('tel')
            corners_v2 = corners[0]
            corners_v3 = corners[1]
            selection_index = np.where( (reference_catalog['v2_spherical_arcsec']>(np.min(corners_v2)-30)) &
                                        (reference_catalog['v2_spherical_arcsec']<(np.max(corners_v2)+30)) &
                                        (reference_catalog['v3_spherical_arcsec']>(np.min(corners_v3)-30)) &
                                        (reference_catalog['v3_spherical_arcsec']<(np.max(corners_v3)+30)) &
                                        (reference_catalog['j_magnitude']>np.min(xmatch_refcat_mag_range)) &
                                        (reference_catalog['j_magnitude']<np.max(xmatch_refcat_mag_range)) )[0]

            obs.reference_catalog = reference_catalog[selection_index]

            # Determine which reference stars fall into the aperture
            path_tel = aperture.path('tel')
            mask = path_tel.contains_points(np.array(obs.reference_catalog['v2_spherical_arcsec', 'v3_spherical_arcsec'].to_pandas()))

            # Now deal with the observed catalog
            ##
            ## NOTES: This is where sci_to_idl happens, so...
            ##
            star_catalog = fpa_data
            star_catalog['star_id'] = star_catalog['id']
            obs.star_catalog = star_catalog






            # Convert from sci frame (in pixels) to idl frame (in arcsec)
            obs.star_catalog['x_idl_arcsec'], obs.star_catalog['y_idl_arcsec'] = \
                aperture.sci_to_idl(np.array(obs.star_catalog['x_SCI']),
                                    np.array(obs.star_catalog['y_SCI']))











            # compute V2/V3: # idl frame in degrees ->  V2/V3_tangent_plane in arcsec
            obs.star_catalog = compute_idl_to_tel_in_table(obs.star_catalog,
                                                           aperture, method=idl_tel_method)

            ######################################################################################
            #
            # In order to do the cross-matching between reference and observed catalog positions,
            # we first need to figure out the rough offset. If this step is not performed,
            # the cross-match using pystortion.distortion.xmatch fails because there are
            # way too many sources in both of the catalogs to ensure a reliable match. [STS]
            #
            bright_index = np.where( (reference_catalog['v2_spherical_arcsec']>(np.min(corners_v2)-20)) &
                                     (reference_catalog['v2_spherical_arcsec']<(np.max(corners_v2)+20)) &
                                     (reference_catalog['v3_spherical_arcsec']>(np.min(corners_v3)-20)) &
                                     (reference_catalog['v3_spherical_arcsec']<(np.max(corners_v3)+20)) &
                                     (reference_catalog['j_magnitude'] < 19.0) )[0] ### 19.0 works for FGS1 and 2, but may need to use 19 for NIRCam?

            print('Number of reference catalog stars used for initial offset match:',len(bright_index))

            ref_cat = reference_catalog[bright_index]

            ref_cat['TPx'], ref_cat['TPy'] = ref_cat['v2_spherical_arcsec'], ref_cat['v3_spherical_arcsec']

            obs_cat = obs.star_catalog
            obs_cat['TPx'], obs_cat['TPy'] = obs_cat['v2_spherical_arcsec'], obs_cat['v3_spherical_arcsec']
            try:
                obs_cat.sort('flux',reverse=True)
            except ValueError:
                obs_cat.sort('flux_fit', reverse=True)
            # Adjust the number of sources to be used for initial matching below
            obs_cat_bright = obs_cat[:299] ##### If I get a "exceeded" error, try changing this number

### THIS WAS REQUIRED FOR THE HUGE OFFSET IN FLIGHT OTE-10 OBS16 (WRONG GUIDE STAR TRACKED)
#            offset_match = matchutils.TPMatch(searchrad=30, separation=1, use2dhist=True, tolerance=0.6) # --> try tolerance=0.5
###
            offset_match = matchutils.TPMatch(searchrad=5, separation=0.7, use2dhist=True, tolerance=0.6) # --> try tolerance=0.5
            idx_ref, idx_obs = offset_match(ref_cat, obs_cat_bright)
            dv2 = ref_cat['TPx'][idx_ref] - obs_cat_bright['TPx'][idx_obs]
            offset_v2 = sigma_clip(dv2.data, sigma=2.5, maxiters=5, cenfunc='median')
            avg_offset_v2 = 3*np.ma.median(offset_v2) - 2*np.ma.mean(offset_v2)

            dv3 = ref_cat['TPy'][idx_ref] - obs_cat_bright['TPy'][idx_obs]
            offset_v3 = sigma_clip(dv3.data, sigma=2.5, maxiters=5, cenfunc='median')
            avg_offset_v3 = 3*np.ma.median(offset_v3) - 2*np.ma.mean(offset_v3)

            print('*************************')
            print(avg_offset_v2, avg_offset_v3)
            print('*************************')

            #
            # Figure - Initial Offsets
            #
            plt.rc('font', family='serif')
            plt.figure(figsize=(10,6))
            plt.suptitle('Initial offsets between measured vs. catalog-based positions - {}'.format(data_name))
            ax1 = plt.subplot(1,2,1)
            ax1.set_xlabel('V2 offset [arcsec]')
            ax1.set_ylabel('N')
            ax1.hist(offset_v2, alpha=0.5, histtype='bar', edgecolor='black', linewidth=1.2)
            ax1.axvline(x=avg_offset_v2, color='r', linewidth=2)
            ax2 = plt.subplot(1,2,2)
            ax2.set_xlabel('V3 offset [arcsec]')
            ax2.set_ylabel('N')
            ax2.hist(offset_v3, alpha=0.5, histtype='bar', edgecolor='black', linewidth=1.2)
            ax2.axvline(x=avg_offset_v3, color='r', linewidth=2)
            plt.tight_layout()
            if save_plot:
                figname = os.path.join(plot_dir,'%s_initial_xmatch.pdf' % data_name)
                plt.savefig(figname, transparent=True, bbox_inches='tight', pad_inches=0)
            if verbose_figures:
                plt.show()
            plt.close()
            #
            # Inpsect figure above, and if you see something weird, the bright mag cutoff for reference catalog
            # and/or the number of sources used in the observed catalog can be changed.
            #
            ######################################################################################

            #
            # Figure - v2v3 crossmatch
            #
            plt.rc('font', family='serif')
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
            plt.tight_layout()
            if save_plot:
                figure_name = os.path.join(plot_dir, '%s_v2v3_xmatch.pdf' % data_name)
                plt.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)
            if verbose_figures:
                plt.show()
            plt.close()

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


            ###
            ### TBD: Try replacing routine below with second run of matchutils.TPMatch instead of relying on pystortion
            ###      This is to remove dependencies on external packages that won't be guaranteed to stay up to date.
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



            idx_reference_cat, idx_star_cat, d2d, d3d, diff_ra, diff_dec = \
                crossmatch.xmatch(reference_cat, star_cat, xmatch_radius,
                                  rejection_level_sigma, verbose=verbose,
                                  verbose_figures=verbose_figures,
                                  saveplot=save_plot, out_dir=plot_dir,
                                  name_seed=fpa_name_seed,
                                  retain_best_match=retain_best_match,
                                  remove_multiple_matches=remove_multiple_matches)

            print('{:d} measured stars, \
                   {:d} reference catalog stars in the aperture, \
                   {:d} matches.'
                   .format(len(obs.star_catalog), np.sum(mask),
                           len(idx_reference_cat)))


            obs.number_of_measured_stars = len(obs.star_catalog)
            obs.number_of_reference_stars = np.sum(mask)
            obs.number_of_matched_stars = len(idx_reference_cat)

            obs.reference_catalog_matched = obs.reference_catalog[idx_reference_cat]

            #***
            #*** Subtract back the offset applied above
            #***
            obs.star_catalog['v2_spherical_arcsec'] -= avg_offset_v2
            obs.star_catalog['v3_spherical_arcsec'] -= avg_offset_v3

            obs.star_catalog_matched = obs.star_catalog[idx_star_cat]
            obs.reference_catalog_matched['star_id'] = obs.star_catalog_matched['star_id']

            # Save space in pickle, speed up
            obs.reference_catalog = []
            obs.star_catalog = []

            obs.siaf_aperture_name = aperture_name
            obs.fpa_data = fpa_data
            obs.fpa_name_seed = fpa_name_seed

            # dictionary that defines the names of columns in the star/reference_catalog for use later on
            fieldname_dict = {}
            fieldname_dict['star_catalog'] = {}       # observed
            fieldname_dict['reference_catalog'] = {}  # reference

            if idl_tel_method == 'spherical':
                fieldname_dict['reference_catalog']['position_1'] = 'v2_spherical_arcsec'
                fieldname_dict['reference_catalog']['position_2'] = 'v3_spherical_arcsec'
            else:
                fieldname_dict['reference_catalog']['position_1'] = 'v2_tangent_arcsec'
                fieldname_dict['reference_catalog']['position_2'] = 'v3_tangent_arcsec'

            # For HawkI and HST catalogs
            reference_catalog_identifier = 'ID'
            fieldname_dict['reference_catalog']['sigma_position_1'] = 'ra_error_mas'
            fieldname_dict['reference_catalog']['sigma_position_2'] = 'dec_error_mas'

            # For Gaia catalog (which is not used for JWST at this point)
            #reference_catalog_identifier = 'source_id'  # Gaia
            #fieldname_dict['reference_catalog']['sigma_position_1'] = 'ra_error'
            #fieldname_dict['reference_catalog']['sigma_position_2'] = 'dec_error'

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

def dva_correct_radec(ra, dec, vx, vy, vz):

    beta_vector = np.array([vx, vy, vz])/const.c.to('km/s').value
    beta2 = np.dot(beta_vector, beta_vector)
    igamma = np.sqrt(1.0 - beta2)
    star_vector = rotations.unit(ra, dec)
    dot_product = np.dot(beta_vector, star_vector)
    scale = igamma/(1.0+dot_product)
    mean_scale = scale.mean()
    star_vector_corrected =  (igamma*star_vector + \
                              (1.0 + (1.0 - igamma)*dot_product/beta2)*beta_vector)/ \
                             (1.0 + dot_product)
    ra_corrected  = np.degrees(np.arctan2(star_vector_corrected[1],
                                          star_vector_corrected[0]))
    dec_corrected = np.degrees(np.arcsin(star_vector_corrected[2]))

    return ra_corrected, dec_corrected #, mean_scale

def dva_correct_v2v3(v2, v3, vx, vy, vz, attitude_matrix):

    ra, dec = rotations.pointing(attitude_matrix, v2, v3)
    ra_corrected, dec_corrected = dva_correct_radec(ra, dec, vx, vy, vz)
    v2_corrected, v3_corrected = rotations.getv2v3(attitude_matrix,
                                                   ra_corrected,
                                                   dec_corrected)

    return v2_corrected, v3_corrected
