#######################
#######################
#######################
### Routine below is for HST case. TBD: Rewrite for JWST in a much simplfied way
#######################
#######################
#######################
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

