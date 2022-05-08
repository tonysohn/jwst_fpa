import os
import copy
import glob
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits

from collections import OrderedDict
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from astropy.visualization import simple_norm
from astropy.nddata import NDData
from astropy.modeling.fitting import LevMarLSQFitter

from jwst import datamodels
from photutils import IRAFStarFinder
from photutils import CircularAperture, EPSFBuilder
from photutils.psf import extract_stars, DAOPhotPSFPhotometry
from photutils.psf import DAOGroup, BasicPSFPhotometry, IterativelySubtractedPSFPhotometry
from photutils.background import MMMBackground, MADStdBackgroundRMS

def epsf_extract_sources(image_file, flux_min=50):

    save_plot = True

    f = image_file

    plt.close('all')
    print()
    print('Image being processed: {}'.format(f))

    im = datamodels.open(f)
    if hasattr(im, 'data') is False:
        im.data = fits.getdata(f)

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
    instrument_pupil  = getattr(im.meta.instrument, 'pupil')

    # DVA correction related
    va_scale   = getattr(im.meta.velocity_aberration, 'scale_factor')
    va_ra_ref  = getattr(im.meta.velocity_aberration, 'va_ra_ref')
    va_dec_ref = getattr(im.meta.velocity_aberration, 'va_dec_ref')
    scvel_x = getattr(im.meta.ephemeris, 'velocity_x_bary')
    scvel_y = getattr(im.meta.ephemeris, 'velocity_y_bary')
    scvel_z = getattr(im.meta.ephemeris, 'velocity_z_bary')

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

    data = copy.deepcopy(im.data)
    #dq = copy.deepcopy(im.dq)

    # Convert image data to counts per second
    photmjsr = getattr(im.meta.photometry,'conversion_megajanskys')
    data_cps = data/photmjsr

################################################################################
### FOR NIS-11 and NIS-13
################################################################################
    bkg_factor   = 1
    minsep_fwhm  = 5
    sharp_lo, sharp_hi = 0.3, 1.4
    round_lo, round_hi = 0.0, 0.6

    if instrument_name == 'NIRISS':

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
            fwhm = 1.5
################################################################################
    
    bkgrms = MADStdBackgroundRMS()
    mmm_bkg = MMMBackground()
    bgrms = bkgrms(data_cps)
    bgavg = mmm_bkg(data_cps)

    iraffind = IRAFStarFinder(threshold=4*bgrms+bgavg,
                              fwhm=fwhm, minsep_fwhm=5,
                              roundlo=round_lo, roundhi=round_hi,
                              sharplo=sharp_lo, sharphi=sharp_hi)

    iraf_extracted_sources = iraffind(data_cps)

    psf_stars = iraf_extracted_sources
    flux_max = np.percentile(psf_stars['flux'], 99.9)
    psf_stars.remove_rows(np.where(psf_stars['flux']<1000))
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
        figname = '{}_sample_psfs.pdf'.format(os.path.basename(f).split('.')[0])
        plt.savefig(figname)

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
        figname = '{}_epsf.pdf'.format(os.path.basename(f).split('.')[0])
        plt.savefig(figname)

    #
    # Now do the ePSF photometry/astrometry using the ePSF built above.
    #
    psf_model = epsf.copy()
    fitter = LevMarLSQFitter()
    daogroup = DAOGroup(2*fwhm)
    psf_phot = BasicPSFPhotometry(finder=iraffind, 
                                  group_maker=daogroup, 
                                  bkg_estimator=mmm_bkg,
                                  psf_model=psf_model, 
                                  fitter=fitter,
                                  fitshape=5,
                                  extra_output_cols=['fwhm','sharpness','roundness','pa','npix','sky','peak'])

    #psf_phot = DAOPhotPSFPhotometry(crit_separation=1.0, threshold=5*bgrms+bgavg,
    #                                fwhm=fwhm, psf_model=psf_model, fitshape=5,
    #                                fitter=fitter, niters=1, aperture_radius=2,
    #                                extra_output_cols=('sharpness','roundness2','sky','peak','npix'))

    print("Performing BasicPSFPhotometry using ePSF")
    tic = time.perf_counter()
    psf_extracted_sources = psf_phot(data_cps)
    toc = time.perf_counter()
    print("Time elapsed for PSF photometry:", toc - tic)
    print('Final source extraction with epsf: {} sources'.format(len(psf_extracted_sources)))
    print()

    extracted_sources = Table()
    extracted_sources['id']        = psf_extracted_sources['id']
    extracted_sources['xcentroid'] = psf_extracted_sources['x_fit']
    extracted_sources['ycentroid'] = psf_extracted_sources['y_fit']
    extracted_sources['fwhm']      = psf_extracted_sources['fwhm']
    extracted_sources['sharpness'] = psf_extracted_sources['sharpness']
    extracted_sources['roundness'] = psf_extracted_sources['roundness']
    extracted_sources['pa']        = psf_extracted_sources['pa']
    extracted_sources['npix']      = psf_extracted_sources['npix']
    extracted_sources['sky']       = psf_extracted_sources['sky']
    extracted_sources['peak']      = psf_extracted_sources['peak']
    extracted_sources['flux']      = psf_extracted_sources['flux_fit']
    extracted_sources['mag']       = -2.5*np.log10(psf_extracted_sources['flux_fit'])
    extracted_sources['mag_err']   = 1.0857*psf_extracted_sources['flux_unc']/psf_extracted_sources['flux_fit']


    print('Number of all extracted sources from {}: {}'.format(f, len(extracted_sources)))
    extracted_sources.remove_rows(np.where(extracted_sources['flux']<flux_min)[0])
    print('Number of filtered extracted sources from {}: {}'.format(f, len(extracted_sources)))
    print()

    extracted_sources_file = '{}_extracted_sources.fits'.format(os.path.basename(f).split('.')[0])
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
        figname = '{}_psfsubtracted_image.pdf'.format(os.path.basename(f).split('.')[0])
        plt.savefig(figname)

    positions = np.transpose((extracted_sources['xcentroid'], extracted_sources['ycentroid']))
    apertures = CircularAperture(positions, r=5)
    norm = simple_norm(data_cps, 'sqrt', percent=99.)

    plt.rc('font', family='serif')
    plt.figure(figsize=(12,12))
    plt.xlabel("X [pix]")
    plt.ylabel("Y [pix]")
    plt.imshow(data_cps, norm=norm, cmap='Greys', origin='lower')
    apertures.plot(color='blue', lw=1.5, alpha=0.5)
    title_string = '{}: {} selected sources'.format(os.path.basename(f), len(extracted_sources))
    plt.title(title_string)
    plt.tight_layout()
    if save_plot:
        figname = '{}_extracted_sources.pdf'.format(os.path.basename(f).split('.')[0])
        plt.savefig(figname)


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

    out_file = '{}_FPA_data.fits'.format(extracted_sources.meta['DATAFILE'].split('.')[0])

    print('Writing {}'.format(out_file))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AstropyWarning, append=True)
        extracted_sources.write(out_file, overwrite=True)

    return
