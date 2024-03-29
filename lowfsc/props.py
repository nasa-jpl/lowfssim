"""A fresh start, hoping for a better API."""
from functools import partial
from lowfsc.constants import BEAM_DIA_AT_DM1

from prysm import (
    fttools,
    propagation,
    polynomials,
    detector
)

from prysm.mathops import np

LOCAM_PIXEL_PITCH = 13
LOCAM_IMAGE_SIZE = 50

WF = propagation.Wavefront


def forward_model_debug(wvl, data, zernikes=None, locam_misfocus=0, locam_shear=(0, 0), hlc_crop=True):
    """Forward model of LOWFS with a few knobs to turn.

    Parameters
    ----------
    wvl : float
        wavelength of light, microns
    data : data.DesignData
        object with properties of:
        roman_pupil - amplitude map of roman pupil
        dm1_wfe - phase map in nm from DM1
        dm2_wfe - phase map in nm from DM2
        fpm - callable with a single argument of wavelength (in um) that produces
            a complex reflection map of the FPM
        pupil_mask - amplitude map of a pupil mask (i.e., for SPC)
    zernikes : ndarray
        1D array of Zernike coefficients, must be same length as len(data.zern309)
    locam_misfocus : float, optional
        misfocus or misconjugation of the LOWFS camera, millimeters
    locam_shear : tuple of int, int
        shear in x and y of the image on locam, in oversampled space.
        The model has 8x oversampling at locam, so locam_shear=(1,0) shears by
        1/8px on the output grid
    hlc_crop : bool, optional
        if True, crop the FoV at OAP6 to the LOCAM AoI
        else leave at natural resolution (FoV determined by data.nmodel)



    Returns
    -------
    dict
        keys of optics, the actual optical planes
                fields, the fields before and after interaction with each plane

    """
    ############################################################################
    # ETL

    roman_pupil = data.roman_pupil
    dm1_wfe = data.dm1_wfe
    dm2_wfe = data.dm2_wfe
    dNpup = data.dNpup
    cmplx_fpm = data.fpm(wvl)

    if zernikes is not None:
        phs = polynomials.sum_of_2d_modes(data.zernNpup, zernikes)
    else:
        phs = data.zNpup

    if dNpup is not None:
        if np.isreal(dNpup.dtype):
            phs = phs + dNpup
            disturbance = WF.from_amp_and_phase(roman_pupil, phs, wvl, data.dx_pup)
        else:
            disturbance = WF.from_amp_and_phase(roman_pupil, phs, wvl, data.dx_pup)
            disturbance *= dNpup
    else:
        disturbance = WF.from_amp_and_phase(roman_pupil, phs, wvl, data.dx_pup)

    # astf = Angular Spectrum Tansfer Function
    if wvl not in data.astf_dm1_to_dm2:
        tf1to2 = propagation.angular_spectrum_transfer_function(data.nmodel, wvl, data.dx_pup, 1000)
        tf2to1 = propagation.angular_spectrum_transfer_function(data.nmodel, wvl, data.dx_pup, -1000)
        data.astf_dm1_to_dm2[wvl] = tf1to2
        data.astf_dm2_to_dm1[wvl] = tf2to1
    else:
        tf1to2 = data.astf_dm1_to_dm2[wvl]
        tf2to1 = data.astf_dm2_to_dm1[wvl]

    if data.prop_dms:
        dm1 = WF.from_amp_and_phase(data.oNpup, dm1_wfe, wvl, data.dx_pup)
        dm2 = WF.from_amp_and_phase(data.oNpup, dm2_wfe, wvl, data.dx_pup)
        dm2.data = fttools.pad2d(dm2.data,
                                 Q=data.nmodel/data.npup,
                                 mode='constant',
                                 value=dm2.data[0, 0])

    # ETL
    ############################################################################

    ############################################################################
    # PROPS

    if data.prop_dms:
        field_at_dm1 = disturbance
        field_after_dm1 = disturbance * dm1

        field_after_dm1.data = fttools.pad2d(field_after_dm1.data,
                                             Q=data.nmodel/data.npup,
                                             mode='constant',
                                             value=field_after_dm1.data[0, 0])

        # propagate from DM1 to DM2 and apply the phase error
        field_at_dm2 = field_after_dm1.free_space(tf=tf1to2)
        field_after_dm2 = field_at_dm2 * dm2

        # now return to the pupil
        field_at_pupil = field_after_dm2.free_space(tf=tf2to1)
    else:
        field_at_dm1 = None
        field_after_dm1 = None
        field_at_dm2 = None
        field_after_dm2 = None
        field_at_pupil = disturbance
        field_at_pupil.data = fttools.pad2d(field_at_pupil.data,
                                            Q=data.nmodel/data.npup,
                                            mode='constant',
                                            value=field_at_pupil.data[0, 0])

    if data.pupil_mask is None:
        field_after_pupil = field_at_pupil
    else:
        field_after_pupil = field_at_pupil * data.pupil_mask

    fno = 32.5676970504222  # TODO: error?  see constants.py, ~32.2?
    efl = BEAM_DIA_AT_DM1*fno
    field_at_oap6, field_at_fpm, field_after_fpm = \
        field_after_pupil.to_fpm_and_back(efl, cmplx_fpm, data.dx_fpm, return_more=True)

    # eps = 1 - abs(cmplx_fpm[0, 0])**2
    # field_at_oap6, field_at_fpm, field_after_fpm, _ = \
        # field_after_pupil.babinet(efl, None, 1 - eps*cmplx_fpm, data.dx_fpm, return_more=True)

    samples_inter_out = field_at_pupil.data.shape[0]
    if locam_misfocus != 0:
        field_after_oap6 = field_at_oap6.free_space(locam_misfocus, Q=2)
        low = samples_inter_out//4
        high = samples_inter_out-low
        field_after_oap6.data = field_after_oap6.data[low:high, low:high]
    else:
        field_after_oap6 = field_at_oap6

    if locam_shear[0] != 0:
        field_after_oap6.data = np.roll(field_after_oap6.data, locam_shear)

    if hlc_crop:
        c = samples_inter_out//2
        w = 200
        field_after_oap6.data = field_after_oap6.data[c-w:c+w, c-w:c+w]

    field_after_oap6.dx = 0.013 / data.bin_factor

    # PROPS
    ############################################################################
    return {
        'optics': {
            'disturbance': disturbance,
            'dm1': dm1_wfe,
            'dm2': dm2_wfe,
            'fpm': cmplx_fpm,
        },
        'fields': {
            'dm1': (field_at_dm1, field_after_dm1),
            'dm2': (field_at_dm2, field_after_dm2),
            'pupil': (field_at_pupil, field_after_pupil),
            'fpm': (field_at_fpm, field_after_fpm),
            'oap6': (field_at_oap6, field_after_oap6)
        }
    }


def forward_model(wvl, data, zernikes=None, locam_misfocus=0, locam_shear=(0, 0)):
    """Forward model of LOWFS with a few knobs to turn.

    Parameters
    ----------
    wvl : float
        wavelength of light, microns
    data : data.DesignData
        object with properties of:
        roman_pupil - amplitude map of roman pupil
        dm1_wfe - phase map in nm from DM1
        dm2_wfe - phase map in nm from DM2
        fpm - callable with a single argument of wavelength (in um) that produces
            a complex reflection map of the FPM
        pupil_mask - amplitude map of a pupil mask (i.e., for SPC)
    zernikes : ndarray
        1D array of Zernike coefficients, must be same length as len(data.zern309)
    locam_misfocus : float, optional
        misfocus or misconjugation of the LOWFS camera, millimeters
    locam_shear : tuple of (int, int)
        shear in x and y of the image on locam, in oversampled space.
        The model has 8x oversampling at locam, so locam_shear=(1,0) shears by
        1/8px on the output grid


    Returns
    -------
    dict
        keys of optics, the actual optical planes
                fields, the fields before and after interaction with each plane

    """
    binfctr = data.bin_factor
    intermediate = forward_model_debug(wvl=wvl, data=data, zernikes=zernikes,
                                       locam_misfocus=locam_misfocus, locam_shear=locam_shear)

    data = intermediate['fields']['oap6'][1].intensity.data
    return detector.bindown(data, binfctr, mode='sum')


def polychromatic(wvls, weights, data, zernikes=None, locam_misfocus=0, locam_shear=(0, 0), pool=None):
    """Polychromatic forward model of LOWFS.

    Parameters
    ----------
    wvls : iterable of float
        wavelengths of light, microns
    weights : iterable of float
        spectral weights to apply to the data
    data : data.DesignData
        object with properties of:
        roman_pupil - amplitude map of roman pupil
        dm1_wfe - phase map in nm from DM1
        dm2_wfe - phase map in nm from DM2
        fpm - callable with a single argument of wavelength (in um) that produces
            a complex reflection map of the FPM
        pupil_mask - amplitude map of a pupil mask (i.e., for SPC)
    zernikes : ndarray
        vector of Zernike coefficients, same length as dd.seed_zernikes
    locam_misfocus : float, optional
        misfocus or misconjugation of the LOWFS camera, millimeters
    locam_shear : tuple of (int, int)
        shear in x and y of the image on locam, in oversampled space.
        The model has 8x oversampling at locam, so locam_shear=(1,0) shears by
        1/8px on the output grid
    pool : mapper
        a type which has a map method, multiprocessing thread and process pools
        work, as do concurrent futures equivalents

    """
    f = partial(forward_model, data=data, zernikes=zernikes, locam_misfocus=locam_misfocus, locam_shear=locam_shear)
    if pool is None:
        out = [f(wvl) for wvl in wvls]
    else:
        out = list(pool.map(f, wvls))  # list exhausts the generator

    out = np.asarray(out)

    # looks weird, just better wrapper for tensordot
    return polynomials.sum_of_2d_modes(out, weights)


def smear(inp, cam):
    smearperline = (cam.frame_time - cam.exposure_time) / \
        cam.exposure_time / (inp.shape[0]-1)
    # for 50 rows, there are 49 "in-between" states;
    # when multiplied by exp_time, real max exposure is frame_time
    out = inp.copy()
    for irow in range(1, inp.shape[0]):  # don't smear at 0 shift
        out[:-irow] += inp[irow:] * smearperline
    return out


def polysmear(wvls, weights, data, cam, **kwargs):
    imnosmear = polychromatic(wvls, weights, data, **kwargs)
    return smear(imnosmear, cam)
