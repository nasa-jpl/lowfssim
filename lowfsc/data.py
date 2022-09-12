"""ETL routines for Roman CGI LOWFS diffraction model."""

# this file contains many lines which have np.array(<something that is an array>)
# on them.
#
# they are not a bug.  import engine as np below imports something that quacks
# like numpy as np.  np.array may actually be cupy.array, which does host=>device
# transfer.
#
# removing them will break the GPU compatibility of this code.
from pathlib import Path
from functools import partial

import numpy as truenp

from scipy.interpolate import interp1d

from scipy.io import loadmat

from astropy.io import fits

from prysm import (
    coordinates,
    geometry,
    polynomials,
    thinfilm,
    refractive,
    detector,
    fttools,
)

from prysm.mathops import np
from prysm.conf import config

from lowfsc.constants import (
    INHERITED_SAMPLING_PITCH_PUPIL_HLC,
    INHERITED_SAMPLING_PITCH_PUPIL_SPC,
    OAP5_Fs,
    BEAM_DIA_AT_OAP5,
    SPC_PIMPLE_HEIGHT,
    AL_BASELINE_THICKNESS,
    BEAM_DIA_AT_DM1,
    FPM_FUSED_SILICA_THICKNESS,
)

BAND1_HLC_DM1_SOLUTION = 'band1/run461_dm1wfe.fits'
BAND1_HLC_DM2_SOLUTION = 'band1/run461_dm2wfe.fits'

PMGI_DESIGN_BAND1 = 'band1/run461_theta6.69imthicks_PMGIfnum32.5676970504_lam5.75e-07_.fits'
PMGI_MDL_SPIN1_ALL_DOF = 'band1/RHLCSN3_R5C2_band1_PMGI_dropin-translation-rotation-scale-registered.fits'
PMGI_MDL_SPIN2_ALL_DOF = 'band1/RHLCSN4_R5C2_Band1_PMGI_stitched_dropin-translation-rotation-scale-registered.fits'
PMGI_MDL_SPIN3_ALL_DOF = 'band1/RHLCSN8_R4C3_Band1_Stitched_PMGI_dropin-translation-rotation-scale-registered.fits'
PMGI_MDL_SPIN4_ALL_DOF = 'band1/RHLC12_SN19_R3C2_PMGIonly_dropin-translation-rotation-scale-registered.fits'

MDL_SN8_BAND3_HR = 'band3/RHLC34_SN8_Band3GS_PMGI-fullres-translation-scale-registered.fits'
MDL_SN8_BAND3_LR = 'band3/RHLC34_SN8_Band3GS_PMGI-dropin-translation-scale-registered.fits'

MDL_SN8_BAND4_HR = 'band4/RHLC34_SN8_Band4GS_PMGI-fullres-translation-scale-registered.fits'
MDL_SN8_BAND4_LR = 'band4/RHLC34_SN8_Band4GS_PMGI-dropin-translation-scale-registered.fits'

thickness_database_band1 = {
    'Ni':   'band1/run461_theta6.69imthicks_nifnum32.5676970504_lam5.75e-07_.fits',
    'Ti':   'band1/run461_theta6.69imthicks_tifnum32.5676970504_lam5.75e-07_.fits',
}

PMGI_DESIGN_BAND2 = 'band2/run249_theta5.5imthicks_PMGIfnum32.22_lam6.6e-07_requestdx2e-07_rot180.fits'

thickness_database_band2 = {
    'Ni': 'band2/run249_theta5.5imthicks_nifnum32.22_lam6.6e-07_requestdx2e-07_.fits',
    'Ti': 'band2/run249_theta5.5imthicks_tifnum32.22_lam6.6e-07_requestdx2e-07_.fits',
}

PMGI_DESIGN_BAND3 = 'band3/run195_theta5.5imthicks_PMGIfnum32.22_lam7.3e-07_requestdx2e-07_rot180.fits'
PMGI_DESIGN_BAND3_LR = 'band3/run195_theta5.5imthicks_PMGIfnum32.22_lam7.3e-07_requestdx2.825422643278202e-06_rot180.fits'

thickness_database_band3 = {
    'Ni':   'band3/run195_theta5.5imthicks_nifnum32.22_lam7.3e-07_requestdx2e-07_.fits',
    'Ti':   'band3/run195_theta5.5imthicks_tifnum32.22_lam7.3e-07_requestdx2e-07_.fits',
}

PMGI_DESIGN_BAND4 = 'band4/run209_theta5.5imthicks_PMGIfnum32.22_lam8.25e-07_requestdx2e-07_rot180.fits'

thickness_database_band4 = {
    'Ni':   'band4/run209_theta5.5imthicks_nifnum32.22_lam8.25e-07_requestdx2e-07_.fits',
    'Ti':   'band4/run209_theta5.5imthicks_tifnum32.22_lam8.25e-07_requestdx2e-07_.fits',
}


def tabular_index(matl, wavelength, data_root):
    """Index of refraction of a material.

    Parameters
    ----------
    matl : str
        a material
    wavelength : float
        wavelength of light, microns

    Returns
    -------
    float or complex
        index associated with the material at this wavelength

    """
    if matl == 'Ni':
        fname = 'Johnson_Ni.csv'
    elif matl == 'Ti':
        fname = 'Johnson_Ti.csv'
    elif matl == 'Cr':
        fname = 'Johnson_Cr.csv'
    elif matl == 'Al':
        fname = 'Mathewson_Al.csv'
    elif matl == 'MgF2':
        fname = 'refractive-index-info_mgf2.csv'
    else:
        raise ValueError('unknown material')

    ary = truenp.loadtxt(data_root / fname, skiprows=1, delimiter=',')
    if ary.shape[1] > 2:
        wvl, n, k = ary.swapaxes(0, 1)
        interpf_n = interp1d(wvl, n)
        interpf_k = interp1d(wvl, k)
        n = interpf_n(wavelength)
        k = interpf_k(wavelength)
        return n + 1j*k
    else:
        wvl, n = ary.swapaxes(0, 1)
        interpf = interp1d(wvl, n)
        return interpf(wavelength)


def refractive_data(material, wavelength, data_root):
    """Refractive index of a material, real or complex.

    Parameters
    ----------
    material : str, {'PMGI', 'Ni', "Ti', 'C7980'}
        a material used in this problem
    wavelength : float or numpy.ndarray
        a wavelength (float) or array of them, microns
    data_root : Path, str, or path_like
        location containing refractive index data

    Returns
    -------
    float or numpy.ndarray
        refractive index, of the same type (or complex version of type)

    """
    if material == 'COMP':
        return 1
    if material == 'C7980':
        # data from Corning data sheet,
        # search Google for "Corning C-7980 refractive index"
        # -> Corning® HPFS® 7979, 7980, 8655 Fused Silica
        # at https://www.corning.com/media/worldwide/csm/documents/5bf092438c5546dfa9b08e423348317b.pdf
        # URL may change...
        # data for 22 Celsius
        A = [  # NOQA
            0.68374049400,
            0.42032361300,
            0.58502748000
        ]
        B = [  # NOQA
            0.00460352869,
            0.01339688560,
            64.49327320000
        ]
        return refractive.sellmeier(wavelength, A, B)
    if material == 'PMGI':
        coefs = [
            1.524,
            5.176e-3,
            2.105e-4
        ]
        return refractive.cauchy(wavelength, *coefs)
    else:
        return tabular_index(material, wavelength, data_root)


def thickness_data(material, data_root, pmgi_fn, tdb=thickness_database_band1):
    """Spatially-varying thickness of a material.

    Parameters
    ----------
    material : str, {'PMGI', 'Ni', 'Ti', 'MgF2', 'C7980'}
        a material used in the FPM stack.
    data_root : Path, str, or path_like
        location containing the thickness data files
    pmgi_fn : str
        filename to use for PMGI data; see lowfssim.data.
            PMGI_DESIGN
            PMGI_OMC
            PMGI_MDL_SPIN1_ALL_DOF
    tdb : dict
        thickness database

    Returns
    -------
    numpy.ndarray
        an array of the material's thickness.  52x52 unless the FITS files change.

    """
    td = partial(thickness_data, data_root=data_root, pmgi_fn=pmgi_fn, tdb=tdb)
    if material == 'COMP':
        return -1 * (td('PMGI') + td('Ni') + td('Ti'))
    if material == 'C7980':
        return truenp.ones_like(td('Ni')) * FPM_FUSED_SILICA_THICKNESS  # 6.5 mm in um
    else:
        if material == 'PMGI':
            fn = pmgi_fn
        else:
            fn = tdb[material]
        with fits.open(data_root / fn) as hdu:
            dat = hdu[0].data.astype(config.precision) * 1e6  # m -> um
            # dat = truenp.roll(dat, -1, axis=0)  # roll keeps data centered on pimple
            # return truenp.flipud(dat)
            return dat


def hlc_fpm_complex_reflection(wavelength, data_root, pmgi_fn, tdb, substrate_R=0.001, polarization='avg', aoi=5.5, samples=512):
    """Complex reflection of the FPM for a given wavelength.

    This is not optimized/vectorized but it is cached

    Parameters
    ----------
    wavelength : float
        wavelength of light, microns
    data_root : Path, str, or path_like
        location containing the thickness and refractive index data files
    pmgi_fn : str
        filename to use for PMGI data; see lowfssim.data
            PMGI_DESIGN
            PMGI_OMC
            PMGI_MDL_SPIN1_ALL_DOF
    tdb : dict
        thickness database
    substrate_R : float
        reflectance value used to paint the region outside the nickel with
    polarization : str, {'r', 's', 'avg'}
        which polarization state to use.  If avg, (r-s)/2, which accounts for phase
    aoi : float
        angle of incidence, degrees
    samples : int
        number of samples in the padded output

    Returns
    -------
    numpy.ndarray
        complex reflection associated with the FPM at that wavelength

    """
    # these two lines and the one before the return ensure that prysm tries
    # to do refractive calculations on CPU instead of GPU
    old = np._srcmodule
    np._srcmodule = truenp
    polarization = polarization.lower()
    matls = [
        'COMP',
        'PMGI',
        'Ni',
        'Ti',
        'C7980',
    ]
    thicknesses = [thickness_data(m, data_root, pmgi_fn, tdb) for m in matls]
    indices = [refractive_data(m, wavelength, data_root.parent) for m in matls]
    shp = thicknesses[0].shape
    # now prepare the stack matrix
    stack = np.empty((5, 2, thicknesses[0].size), dtype=config.precision_complex)
    for ilayer in range(5):
        stack[ilayer, 0, :] = indices[ilayer]
        t_layer = thicknesses[ilayer]
        if hasattr(t_layer, 'ravel'):
            # spatially varying
            stack[ilayer, 1, :] = thicknesses[ilayer].ravel()
        else:
            # constant
            stack[ilayer, 1, :] = thicknesses[ilayer]

    out = truenp.zeros(shp, dtype=config.precision_complex)
    if polarization == 's':
        r, _ = thinfilm.multilayer_stack_rt(stack=stack, wavelength=wavelength, polarization='s', aoi=aoi)
    elif polarization == 'p':
        r, _ = thinfilm.multilayer_stack_rt(stack=stack, wavelength=wavelength, polarization='p', aoi=aoi)
    else:
        rs, _ = thinfilm.multilayer_stack_rt(stack=stack, wavelength=wavelength, polarization='s', aoi=aoi)
        rp, _ = thinfilm.multilayer_stack_rt(stack=stack, wavelength=wavelength, polarization='p', aoi=aoi)
        r = (rs + rp) / 2

    out[:] = r.reshape(out.shape)
    out[thicknesses[1] == 0] = truenp.sqrt(substrate_R)
    denom = out.shape[0]
    out = fttools.pad2d(out, Q=samples/denom, mode='constant', value=out[0, 0])
    np._srcmodule = old
    return np.array(out)


def spc_spec_complex_reflection(wavelength, data_root, substrate_R=0.001, polarization='avg', aoi=5.5, samples=512):
    """Complex reflection of the FPM for a given wavelength.

    Parameters
    ----------
    wavelength : float
        wavelength of light, microns
    data_root : Path, str, or path_like
        location containing the thickness and refractive index data files
    substrate_R : float
        reflectance value used to paint the region outside the nickel with
    polarization : str, {'r', 's', 'avg'}
        which polarization state to use.  If avg, (r-s)/2, which accounts for phase
    aoi : float
        angle of incidence, degrees
    samples : int
        number of samples in the padded output

    Returns
    -------
    numpy.ndarray
        complex reflection associated with the FPM at that wavelength

    """
    # this function works basically as follows:
    # 1. load mask design file, resample as appropriate.
    #    In a separate array, paint the (boolean) mask for where the pimple is
    # 2. compute the complex reflectance via thin films for both places
    # 3. paint those values into the output array
    old = np._srcmodule
    np._srcmodule = truenp

    bowtie = fits.getdata(data_root/'spc-spec'/'fpm_0.05lamD.fits')
    bowtie = detector.bindown(bowtie, 3, mode='avg')  # Q=20 to Q=6.66
    s = bowtie.shape[0]
    sb2 = s//2
    # magic numbers:
    # 0.15 = oversampling
    # 0.575 = design wavelength
    ss = OAP5_Fs * 0.575 / BEAM_DIA_AT_OAP5 * 0.15
    width_minor = 18.54
    width_major = 18.54*1.75
    width_minor /= ss
    width_major /= ss
    major_axis_angle = -90

    x = (np.arange(s)-sb2)
    y = x
    xv, yv = np.meshgrid(x, y)
    A = np.radians(-major_axis_angle)
    a, b = width_major, width_minor
    major_axis_term = ((xv * np.cos(A) + yv * np.sin(A)) ** 2) / a ** 2
    minor_axis_term = ((xv * np.sin(A) - yv * np.cos(A)) ** 2) / b ** 2
    pimple = major_axis_term + minor_axis_term > 1
    stack_bg = [
        (refractive_data('Al', wavelength, data_root), AL_BASELINE_THICKNESS),
        (refractive_data('C7980', wavelength, data_root), FPM_FUSED_SILICA_THICKNESS),
    ]
    stack_lowfs = [
        (refractive_data('Al', wavelength, data_root), AL_BASELINE_THICKNESS+SPC_PIMPLE_HEIGHT),
        (refractive_data('C7980', wavelength, data_root), FPM_FUSED_SILICA_THICKNESS),
    ]
    if polarization == 's':
        r_bg, _ = thinfilm.multilayer_stack_rt(stack=stack_bg, wavelength=wavelength, polarization='s', aoi=aoi)
        r_lwf, _ = thinfilm.multilayer_stack_rt(stack=stack_lowfs, wavelength=wavelength, polarization='s', aoi=aoi)
    elif polarization == 'p':
        r_bg, _ = thinfilm.multilayer_stack_rt(stack=stack_bg, wavelength=wavelength, polarization='p', aoi=aoi)
        r_lwf, _ = thinfilm.multilayer_stack_rt(stack=stack_lowfs, wavelength=wavelength, polarization='p', aoi=aoi)
    else:
        rs_bg, _ = thinfilm.multilayer_stack_rt(stack=stack_bg, wavelength=wavelength, polarization='s', aoi=aoi)
        rp_bg, _ = thinfilm.multilayer_stack_rt(stack=stack_bg, wavelength=wavelength, polarization='p', aoi=aoi)
        r_bg = (rs_bg + rp_bg) / 2
        rs_lwf, _ = thinfilm.multilayer_stack_rt(stack=stack_lowfs, wavelength=wavelength, polarization='s', aoi=aoi)
        rp_lwf, _ = thinfilm.multilayer_stack_rt(stack=stack_lowfs, wavelength=wavelength, polarization='p', aoi=aoi)
        r_lwf = (rs_lwf + rp_lwf) / 2

    r_bg = r_bg.real    # explicitly zero phase, going to add in phase for _lwf
    r_lwf = r_lwf.real  # but not _bg
    r_lwf = r_lwf * np.exp(-1j * 2 * np.pi / wavelength * -2*SPC_PIMPLE_HEIGHT)

    arr = np.ones(bowtie.shape, config.precision_complex) * (substrate_R**2)
    arr[bowtie == 0] = r_bg
    arr[pimple == 0] = r_lwf
    arr = fttools.pad2d(arr, Q=samples/arr.shape[0], mode='constant', value=arr[0, 0])
    np._srcmodule = old
    return np.array(arr)


def spc_wfov_complex_reflection(wavelength, data_root, substrate_R=0.001, polarization='avg', aoi=5.5, samples=512):
    """Complex reflection of the FPM for a given wavelength.

    Parameters
    ----------
    wavelength : float
        wavelength of light, microns
    data_root : Path, str, or path_like
        location containing the thickness and refractive index data files
    substrate_R : float
        reflectance value used to paint the region outside the nickel with
    polarization : str, {'r', 's', 'avg'}
        which polarization state to use.  If avg, (r-s)/2, which accounts for phase
    aoi : float
        angle of incidence, degrees
    samples : int
        number of samples in the padded output

    Returns
    -------
    numpy.ndarray
        complex reflection associated with the FPM at that wavelength

    """
    # this function works basically as follows:
    # 1. load mask design file, resample as appropriate.
    #    In a separate array, paint the (boolean) mask for where the pimple is
    # 2. compute the complex reflectance via thin films for both places
    # 3. paint those values into the output array
    old = np._srcmodule
    np._srcmodule = truenp

    ring = fits.getdata(data_root/'spc-wfov'/'FPM_SPC-20200610_res6.fits')
    s = ring.shape[0]
    sb2 = s//2
    x = (np.arange(s)-sb2)
    y = x
    xv, yv = np.meshgrid(x, y)
    rv = np.sqrt(xv**2 + yv**2)

    pimple = rv < (0.65*6)
    stack_bg = [
        (refractive_data('Al', wavelength, data_root), AL_BASELINE_THICKNESS),
        (refractive_data('C7980', wavelength, data_root), FPM_FUSED_SILICA_THICKNESS),
    ]
    stack_lowfs = [
        (refractive_data('Al', wavelength, data_root), AL_BASELINE_THICKNESS+SPC_PIMPLE_HEIGHT),
        (refractive_data('C7980', wavelength, data_root), FPM_FUSED_SILICA_THICKNESS),
    ]

    if polarization == 's':
        r_bg, _ = thinfilm.multilayer_stack_rt(stack=stack_bg, wavelength=wavelength, polarization='s', aoi=aoi)
        r_lwf, _ = thinfilm.multilayer_stack_rt(stack=stack_lowfs, wavelength=wavelength, polarization='s', aoi=aoi)
    elif polarization == 'p':
        r_bg, _ = thinfilm.multilayer_stack_rt(stack=stack_bg, wavelength=wavelength, polarization='p', aoi=aoi)
        r_lwf, _ = thinfilm.multilayer_stack_rt(stack=stack_lowfs, wavelength=wavelength, polarization='p', aoi=aoi)
    else:
        rs_bg, _ = thinfilm.multilayer_stack_rt(stack=stack_bg, wavelength=wavelength, polarization='s', aoi=aoi)
        rp_bg, _ = thinfilm.multilayer_stack_rt(stack=stack_bg, wavelength=wavelength, polarization='p', aoi=aoi)
        r_bg = (rs_bg + rp_bg) / 2
        rs_lwf, _ = thinfilm.multilayer_stack_rt(stack=stack_lowfs, wavelength=wavelength, polarization='s', aoi=aoi)
        rp_lwf, _ = thinfilm.multilayer_stack_rt(stack=stack_lowfs, wavelength=wavelength, polarization='p', aoi=aoi)
        r_lwf = (rs_lwf + rp_lwf) / 2

    r_bg = r_bg.real    # explicitly zero phase, going to add in phase for _lwf
    r_lwf = r_lwf.real  # but not _bg
    r_lwf = r_lwf * np.exp(-1j * 2 * np.pi / wavelength * -2*SPC_PIMPLE_HEIGHT)
    arr = np.ones(ring.shape, config.precision_complex) * (substrate_R**2)
    arr[ring == 0] = r_bg
    arr[pimple] = r_lwf
    arr = fttools.pad2d(arr, Q=samples/arr.shape[0], mode='constant', value=arr[0, 0])
    np._srcmodule = old
    return np.array(arr)


def parametrized_spc_wfov_complex_reflection(wavelength, data_root, substrate_R=0.001, polarization='avg', aoi=5.5,
                                             pimple_radius=0.65, pimple_height=0.072):
    """Complex reflection of the FPM for a given wavelength.

    This is not optimized/vectorized but it is cached

    Parameters
    ----------
    wavelength : float
        wavelength of light, microns
    data_root : Path, str, or path_like
        location containing the thickness and refractive index data files
    substrate_R : float
        reflectance value used to paint the region outside the nickel with
    polarization : str, {'r', 's', 'avg'}
        which polarization state to use.  If avg, (r-s)/2, which accounts for phase
    aoi : float
        angle of incidence, degrees
    pimple_radius : float
        semidiameter of the pimple, units of fL/D
    pimple_height : float
        height of the pimple, um

    Returns
    -------
    numpy.ndarray
        complex reflection associated with the FPM at that wavelength

    """
    old = np._srcmodule
    np._srcmodule = truenp

    Q = 20
    ID = 5.6
    OD = 20.4
    samples = 850
    sb2 = samples//2
    # magic numbers:
    # 6 = oversampling,
    # .575 = design wavelength of FPM
    x = np.arange(-sb2, -sb2+samples, dtype=config.precision) / Q
    y = x
    xv, yv = np.meshgrid(x, y)
    rv = np.sqrt(xv**2 + yv**2)

    pimple = rv < (pimple_radius)
    ring1 = rv > ID
    ring2 = rv < OD
    ring = ring1 & ring2
    stack_bg = [
        (refractive_data('Al', wavelength, data_root), AL_BASELINE_THICKNESS),
        (refractive_data('C7980', wavelength, data_root), FPM_FUSED_SILICA_THICKNESS),
    ]
    stack_lowfs = [
        (refractive_data('Al', wavelength, data_root), AL_BASELINE_THICKNESS+pimple_height),
        (refractive_data('C7980', wavelength, data_root), FPM_FUSED_SILICA_THICKNESS),
    ]
    if polarization == 's':
        r_bg, _ = thinfilm.multilayer_stack_rt(stack=stack_bg, wavelength=wavelength, polarization='s', aoi=aoi)
        r_lwf, _ = thinfilm.multilayer_stack_rt(stack=stack_lowfs, wavelength=wavelength, polarization='s', aoi=aoi)
    elif polarization == 'p':
        r_bg, _ = thinfilm.multilayer_stack_rt(stack=stack_bg, wavelength=wavelength, polarization='p', aoi=aoi)
        r_lwf, _ = thinfilm.multilayer_stack_rt(stack=stack_lowfs, wavelength=wavelength, polarization='p', aoi=aoi)
    else:
        rs_bg, _ = thinfilm.multilayer_stack_rt(stack=stack_bg, wavelength=wavelength, polarization='s', aoi=aoi)
        rp_bg, _ = thinfilm.multilayer_stack_rt(stack=stack_bg, wavelength=wavelength, polarization='p', aoi=aoi)
        r_bg = (rs_bg + rp_bg) / 2
        rs_lwf, _ = thinfilm.multilayer_stack_rt(stack=stack_lowfs, wavelength=wavelength, polarization='s', aoi=aoi)
        rp_lwf, _ = thinfilm.multilayer_stack_rt(stack=stack_lowfs, wavelength=wavelength, polarization='p', aoi=aoi)
        r_lwf = (rs_lwf + rp_lwf) / 2

    r_bg = r_bg.real    # explicitly zero phase, going to add in phase for _lwf
    r_lwf = r_lwf.real  # but not _bg
    r_lwf = r_lwf * np.exp(-1j * 2 * np.pi / wavelength * (-2*pimple_height))
    arr = np.ones(ring.shape, config.precision_complex) * (substrate_R**2)
    arr[~ring] = r_bg
    arr[pimple] = r_lwf
    arr = fttools.pad2d(arr, Q=4096/arr.shape[0], mode='constant', value=arr[0, 0])
    np._srcmodule = old
    return np.array(arr)


EXEP_SIMPLE_OCCULTER_DIAMETER_UM = 376.33
EXEP_SIMPLE_OCCULTER_THICKNESS_TI_NM = 3
EXEP_SIMPLE_OCCULTER_THICKNESS_NI_NM = 100
EXEP_SIMPLE_OCCULTER_BASE_THICKNESS_PMGI_NM = 1130
EXEP_SIMPLE_OCCULTER_PCWFS_SPOT_HEIGHT_NM = -135  # + = outie, - = innie
EXEP_SIMPLE_OCCULTER_PCWFS_SPOT_DIAMETER_UM = 18.53

PARAMETRIZED_PCWFS_DEFAULT_SAMPLING = 18.53/16


def parametrized_hlc_like_pcwfs_spot(wavelength, data_root, substrate_R=0.001, polarization='avg', aoi=5.5, dx=PARAMETRIZED_PCWFS_DEFAULT_SAMPLING, samples=1024,
                                     stack_diameter=EXEP_SIMPLE_OCCULTER_DIAMETER_UM,
                                     l1_ti_thickness=EXEP_SIMPLE_OCCULTER_THICKNESS_TI_NM,
                                     l2_ni_thickness=EXEP_SIMPLE_OCCULTER_THICKNESS_NI_NM,
                                     l3_pmgi_thickness=EXEP_SIMPLE_OCCULTER_BASE_THICKNESS_PMGI_NM,
                                     l3_pmgi_pcwfs_offset=EXEP_SIMPLE_OCCULTER_PCWFS_SPOT_HEIGHT_NM,
                                     l3_pmgi_pcwfs_diameter=EXEP_SIMPLE_OCCULTER_PCWFS_SPOT_DIAMETER_UM):
    """Parametrized recipe for an Hybrid-Lyot like PCWFS-optimized occulter.

    Parameters
    ----------
    wavelength : float
        wavelength of light, microns
    data_root : Path, str, or path_like
        location containing the thickness and refractive index data files
    substrate_R : float
        reflectance value used to paint the region outside the nickel with
    polarization : str, {'r', 's', 'avg'}
        which polarization state to use.  If avg, (r-s)/2, which accounts for phase
    aoi : float
        angle of incidence, degrees
    dx : float
        inter-sample spacing, microns
    samples : int
        diameter of the output, in integer samples (times dx)
    stack_diameter : float
        diameter of the thin film stack, microns
        a circular occulter of this diameter will be drawn, with three layers:
        ti, ni, pmgi, having the same diameter and the given thicknesses
        there is a depression in the PMGI layer, which forms the phase contrast
        wavefront sensing "spot"
    l1_ti_thickness: float
        thickness of the first titanium layer, in nanometers
    l2_ni_thickness: float
        thickness of the second nickel layer, in nanometers
    l3_pmgi_thickness: float
        base or bulk thickness of trhe third PMGI layer, in nanometers
    l3_pmgi_pcwfs_offset : float
        offset of the "spot" portion of the PMGI layer, in nanometers
        positive = protrusion
        nagative = depression
    l3_pmgi_pcwfs_diameter : float
        diameter of the PCWFS spot, in microns

    Returns
    -------
    numpy.ndarray
        complex reflection coefficient associated with this thin film occulter
        at the specified AOI and polarization

    """
    old = np._srcmodule
    np._srcmodule = truenp
    polarization = polarization.lower()

    # in doing the thin film calculations, as for the SPC cases we fudge everything
    # compute only two cases, inside the bulk area or inside the PMGI depression
    # the process of packaging everything to vectorize the calculation over only
    # two elements is not worth it, so build two separate stacks

    # stack is layer, [n,t]
    # /1e3s -- nm -> um
    indices = {m: refractive_data(m, wavelength, data_root) for m in ['Ni', 'Ti', 'PMGI', 'C7980']}
    # BDD 2022-03-01: issues with failure in the calculation due to the etalon formed with the C7980
    # the Ni is thick enough to effectively render the C7980 moot, so comment it out... lesser of two
    # evils
    stack_bulk = [
        (1.0,              -l3_pmgi_pcwfs_offset/1e3),  # vacuum compensator so that phases are evaluated at the same plane
        (indices['PMGI'],  l3_pmgi_thickness/1e3),
        (indices['Ni'],    l2_ni_thickness/1e3),
        (indices['Ti'],    l1_ti_thickness/1e3),
        # (indices['C7980'], FPM_FUSED_SILICA_THICKNESS), # 6.35 mm thickness of fused silica
    ]
    stack_spot = [
        # no compensator
        (indices['PMGI'],  (l3_pmgi_thickness + l3_pmgi_pcwfs_offset)/1e3),
        (indices['Ni'],    l2_ni_thickness/1e3),
        (indices['Ti'],    l1_ti_thickness/1e3),
        # (indices['C7980'], FPM_FUSED_SILICA_THICKNESS), # 6.35 mm thickness of fused silica

    ]
    # lwf = lowfs, bg = background
    if polarization == 's':
        r_bg, _ = thinfilm.multilayer_stack_rt(stack=stack_bulk, wavelength=wavelength, polarization='s', aoi=aoi)
        r_lwf, _ = thinfilm.multilayer_stack_rt(stack=stack_spot, wavelength=wavelength, polarization='s', aoi=aoi)
    elif polarization == 'p':
        r_bg, _ = thinfilm.multilayer_stack_rt(stack=stack_bulk, wavelength=wavelength, polarization='p', aoi=aoi)
        r_lwf, _ = thinfilm.multilayer_stack_rt(stack=stack_spot, wavelength=wavelength, polarization='p', aoi=aoi)
    else:
        rs_bg, _ = thinfilm.multilayer_stack_rt(stack=stack_bulk, wavelength=wavelength, polarization='s', aoi=aoi)
        rp_bg, _ = thinfilm.multilayer_stack_rt(stack=stack_bulk, wavelength=wavelength, polarization='p', aoi=aoi)
        r_bg = (rs_bg + rp_bg) / 2
        rs_lwf, _ = thinfilm.multilayer_stack_rt(stack=stack_spot, wavelength=wavelength, polarization='s', aoi=aoi)
        rp_lwf, _ = thinfilm.multilayer_stack_rt(stack=stack_spot, wavelength=wavelength, polarization='p', aoi=aoi)
        r_lwf = (rs_lwf + rp_lwf) / 2

    # now build up the mask in three parts:
    reflectivity = np.sqrt(substrate_R)  # reflectance => reflectivity
    reflectivity = reflectivity * np.exp(1j*np.angle(r_bg))  # enforce zero phase delta between plateau
    crefl = np.full((samples, samples), reflectivity, dtype=config.precision_complex)

    # TODO: no phase = background has same phase angle as the bulk, or
    #       no phase = background has zero phase
    # ...?
    # A/B compared with nominal ExEp sized occulter, found next to no difference
    x, y = coordinates.make_xy_grid(samples, dx=dx)
    r, _ = coordinates.cart_to_polar(x, y)
    bulk_mask = geometry.circle(stack_diameter/2, r)
    spot_mask = geometry.circle(l3_pmgi_pcwfs_diameter/2, r)
    crefl[bulk_mask] = r_bg
    crefl[spot_mask] = r_lwf
    np._srcmodule = old
    return np.array(crefl)


def aj_contributed_dual_occulter(wavelength, data_root, sn, spot_depth, fpm_samples):
    if sn not in (4, 19):
        raise ValueError(f'sn must be 4 or 19, got {sn}')

    sn4_valid_depths = [174, 190, 204, 214, 231, 241, 244]
    sn19_valid_depths = [166, 175, 190, 204, 218, 229, 231]
    valid_wavelengths = truenp.array([575-64, 575-32, 575, 575+32, 575+64])*1e-3
    valid_depths = {
        4: sn4_valid_depths,
        19: sn19_valid_depths
    }
    if spot_depth not in valid_depths[sn]:
        raise ValueError(f'for sn {sn} spot_depth must be in {valid_depths[sn]}, got {spot_depth}')

    try:
        iwvl = truenp.around(valid_wavelengths, 3).tolist().index(wavelength)
    except ValueError as e:
        raise ValueError(f'wavelength must be a member of {valid_wavelengths}, got {wavelength}') from e

    idepth = valid_depths[sn].index(spot_depth)

    p = data_root / 'hlc' / f'aj_contributed_sn{sn}_5wvl.mat'
    d = loadmat(p)
    carr = d['rcube'][idepth, iwvl]
    # astype -> fp64 to fp32 | array -> cpu to gpu
    carr = np.array(carr.astype(config.precision_complex))
    carr = fttools.pad2d(carr, out_shape=fpm_samples, mode='edge')
    return np.array(carr)


class DesignData():
    """A class which prevents you from having to go to disk to get an array."""

    def __init__(self, roman_pupil, dm1_wfe, dm2_wfe, dx_pup, npup, nmodel, prop_dms, fpm, fpm_dx, fpm_samples, which, pupil_mask=None):
        """Create a new DesignData instance.

        Parameters
        ----------
        roman_pupil : numpy.ndarray
            ndarray of roman pupil amplitude, i.e. sqrt(Intensity)
        dm1_wfe : numpy.ndarray
            array of wavefront error (OPD) applied by DM1, with same array shape
            as roman_pupil.  Units of nm.
        dm2_wfe : numpy.ndarray
            as dm1_wfe, but for DM2.
        dx_pup : float
            sample spacing of the pupil plane, microns
        npup : int
            number of samples spanned by the Roman pupil
        nmodel : int
            number of samples spanning the array used in the model
        prop_dms : bool
            if True, propagates between and applies the DMs; if false, ignores
            the DMs
        fpm : callable
            focal plane mask generator.  Should accept a single argument of "wvl"
            which is the wavelength at which to compute the complex reflectivity
        fpm_dx : float
            inter-sample spacing at the FPM, microns
        fpm_samples : int
            number of samples across the FPM
        which : str
            either HLC or SPC, specifies which model to run
            The models have different sampling, so this flag is needed
        pupil_mask : numpy.ndarray
            ndarray of any pupil plane mask (excluding the roman pupil) to be
            applied before propagating to the FPM

        Notes
        -----
        Many variables created in this init and used later have shorthand names.
        The decoder ring is as follows:

        <L>Npup like xNpup is a variable on a NpupxNpup array.
        [x,y],[r,t] are the cartesian and polar coordinates.
        rNpupz is normalized for Zernikes and does not have max ~= 23, but ~= 1

        dNpup is the disturbance on a NpupxNpup.  If it is None, the model will do
        no computation.  If it is not None, the behavior changes depending on the type.
        If the array is of complex dtype, it is multiplied by the complex field.
        If it is of real dtype, it is added to the summed Zernike phase (if any).

        zNpup is an array of zeros of size NpupxNpup.
        oNpup is an array of ones of size NpupxNpup.

        zernNpup is the stack of Zernike terms on a NpupxNpup grid.  it is a
        length N list of NpupxNpup arrays.
        if zernNpup is None and the model is called with nonzero zernike coefficients
        an error will be generated.


        """
        self.dx_pup = dx_pup
        self.npup = npup
        self.nmodel = nmodel
        self.prop_dms = prop_dms
        self.which = which

        self.xNpup, self.yNpup = coordinates.make_xy_grid(npup, dx=dx_pup)
        self.rNpup, self.tNpup = coordinates.cart_to_polar(self.xNpup, self.yNpup)
        # rNpupz is for Zernikes and has a value of 1 where the spines of the array reach the edge
        self.rNpupz = self.rNpup / (BEAM_DIA_AT_DM1 / 2)

        self.zNpup = np.zeros_like(self.xNpup)
        self.oNpup = np.ones_like(self.xNpup)

        self.astf_dm1_to_dm2 = {}
        self.astf_dm2_to_dm1 = {}

        self.dNpup = None
        self.zernNpup = None
        self.roman_pupil = roman_pupil
        self.dm1_wfe = dm1_wfe
        self.dm2_wfe = dm2_wfe
        self.fpm = fpm
        self.dx_fpm = fpm_dx
        self.fpm_samples = fpm_samples
        self.pupil_mask = pupil_mask
        self.dm1 = False
        self.dm2 = False

    def seed_zernikes(self, jnoll=None, nms=None):
        """Compute and save Zernike polnyomials.

        Parameters
        ----------
        jnoll : iterable of int, optional
            [1,2,3,4,5] Noll indexed Zernike description (base 1;1=piston)
            a range object works
        nms : iterable of tuple, optional
            [(n1,m1),(n2,m2)] ANSI 2 term Zernike description

        Notes
        -----
        either nms or jnoll must be given.  If jnoll is not None, it clobbers nms

        """
        if nms is None and jnoll is None:
            raise ValueError('one of nms and jnoll must not be None')
        if jnoll is not None:
            nms = [polynomials.noll_to_nm(j) for j in jnoll]

        self.zernNpup = np.array(list(polynomials.zernike_nm_sequence(nms, self.rNpupz, self.tNpup)))
        return

    def update_dm(self, n, actuators):
        """Update one of the DMs.

        Parameters
        ----------
        n : int
            which DM
        actuators : numpy.ndarray
            48x48 array of actuator commands, in units of length and incorporating
            gain maps, etc, or any other desired processing

        Notes
        -----
        self.prop_dms will be set to True by this function call
        must call lowfssim.dm.setup_dms prior to using this function

        """
        # an alternative that permits "more code reuse" is to use strings to
        # specify which attribute to lookup, self.dm1 or self.dm2, and where
        # to put the final array, using setattr/getattr.  But that is slower
        # than the hard-coding here and basically the same number of LOC
        if n == 1:
            self.dm1.actuators = actuators
            dmmap = self.dm1.render(True)  # wfe, not sfe
            dmmap = fttools.crop_center(dmmap, self.npup)
            self.dm1_wfe = dmmap
        elif n == 2:
            self.dm2.actuators = actuators
            dmmap = self.dm2.render(True)
            dmmap = fttools.crop_center(dmmap, self.npup)
            self.dm2_wfe = dmmap

        self.prop_dms = True
        return

    @classmethod
    def _hlcfactory(cls, data_fldr, fpm_fov, pmgi_fn, tdb, band, override_fpm_dx=None):
        if band == 1:
            fpm_dx = 2.825422643278202
            dm1_fn = BAND1_HLC_DM1_SOLUTION
            dm2_fn = BAND1_HLC_DM2_SOLUTION
        elif band in (2, 3, 4):
            fpm_dx = 0.2
            dm1_fn = None
            dm2_fn = None

        if override_fpm_dx is not None:
            fpm_dx = override_fpm_dx

        data_fldr = Path(data_fldr) / 'hlc'
        roman_pupil = fits.getdata(data_fldr / 'run461_pupil_rotated.fits')
        if dm1_fn is not None:
            dm1_wfe = fits.getdata(data_fldr / dm1_fn) * 1e9
            dm2_wfe = fits.getdata(data_fldr / dm2_fn) * 1e9
            dm1_wfe = fttools.crop_center(dm1_wfe, 309)
            dm2_wfe = fttools.crop_center(dm2_wfe, 309)
        else:
            dm1_wfe = np.zeros((309, 309), dtype=config.precision)
            dm2_wfe = np.zeros((309, 309), dtype=config.precision)

        roman_pupil = fttools.crop_center(roman_pupil, 309)

        # double astype => do conversion on CPU to avoid bug in cupy,
        # then convert for dtype homogenaeity
        roman_pupil = np.array(roman_pupil.astype(truenp.float64)).astype(config.precision)

        # this normalization makes the whole diffraction propagation a "unitary"
        # operation.  In other words, diffraction x radiometry = photons
        # the sqrt is because we are normalizing in amplitude space and not
        # intensity space
        roman_pupil /= np.sqrt(roman_pupil.sum())

        dm1_wfe = np.array(dm1_wfe).astype(config.precision)
        dm2_wfe = np.array(dm2_wfe).astype(config.precision)
        hfcr = partial(hlc_fpm_complex_reflection, pmgi_fn=pmgi_fn, samples=fpm_fov, tdb=tdb)
        fpm = FPMCache(hfcr, data_fldr)
        pupil_mask = None

        return cls(roman_pupil=roman_pupil,
                   dm1_wfe=dm1_wfe,
                   dm2_wfe=dm2_wfe,
                   dx_pup=INHERITED_SAMPLING_PITCH_PUPIL_HLC,
                   npup=309,
                   nmodel=512,
                   prop_dms=True,
                   fpm=fpm,
                   # magic number - from FITS file provided by A.J.
                   fpm_dx=fpm_dx,
                   fpm_samples=fpm_fov,
                   which='HLC',
                   pupil_mask=pupil_mask)

    @classmethod
    def hlc_design(cls, data_fldr, fpm_fov=256):
        return cls._hlcfactory(data_fldr, fpm_fov, PMGI_DESIGN_BAND1, thickness_database_band1, 1)

    @classmethod
    def hlc_spin1(cls, data_fldr, fpm_fov=256):
        return cls._hlcfactory(data_fldr, fpm_fov, PMGI_MDL_SPIN1_ALL_DOF, thickness_database_band1, 1)

    @classmethod
    def hlc_spin2(cls, data_fldr, fpm_fov=256):
        return cls._hlcfactory(data_fldr, fpm_fov, PMGI_MDL_SPIN2_ALL_DOF, thickness_database_band1, 1)

    @classmethod
    def hlc_spin3(cls, data_fldr, fpm_fov=256):
        return cls._hlcfactory(data_fldr, fpm_fov, PMGI_MDL_SPIN3_ALL_DOF, thickness_database_band1, 1)

    @classmethod
    def hlc_spin4(cls, data_fldr, fpm_fov=256):
        return cls._hlcfactory(data_fldr, fpm_fov, PMGI_MDL_SPIN4_ALL_DOF, thickness_database_band1, 1)

    @classmethod
    def hlc_band2_design(cls, data_fldr, fpm_fov=4096):
        self = cls._hlcfactory(data_fldr, fpm_fov, PMGI_DESIGN_BAND2, thickness_database_band2, 2)
        self.which = 'HLC'
        # do not have DM design shapes for band 2,3,4
        self.dm1_wfe[:] = 0
        self.dm2_wfe[:] = 0
        return self

    @classmethod
    def hlc_band3_design(cls, data_fldr, fpm_fov=4096):
        self = cls._hlcfactory(data_fldr, fpm_fov, PMGI_DESIGN_BAND3, thickness_database_band3, 3)
        self.which = 'HLC'
        # do not have DM design shapes for band 2,3,4
        self.dm1_wfe[:] = 0
        self.dm2_wfe[:] = 0
        return self

    @classmethod
    def hlc_band4_design(cls, data_fldr, fpm_fov=4096):
        self = cls._hlcfactory(data_fldr, fpm_fov, PMGI_DESIGN_BAND4, thickness_database_band4, 4)
        self.which = 'HLC'
        # do not have DM design shapes for band 2,3,4
        self.dm1_wfe[:] = 0
        self.dm2_wfe[:] = 0
        return self

    @classmethod
    def hlc_band3_sn8_hr(cls, data_fldr, fpm_fov=4096):
        self = cls._hlcfactory(data_fldr, fpm_fov, MDL_SN8_BAND3_HR, thickness_database_band3, 3, override_fpm_dx=0.2)
        self.which = 'HLC'
        # do not have DM design shapes for band 2,3,4
        self.dm1_wfe[:] = 0
        self.dm2_wfe[:] = 0
        return self

    @classmethod
    def hlc_band3_sn8_lr(cls, data_fldr, fpm_fov=256):
        self = cls._hlcfactory(data_fldr, fpm_fov, MDL_SN8_BAND3_LR, thickness_database_band3, 3, override_fpm_dx=2.825422643278202)
        self.which = 'HLC'
        # do not have DM design shapes for band 2,3,4
        self.dm1_wfe[:] = 0
        self.dm2_wfe[:] = 0
        return self

    @classmethod
    def hlc_band4_sn8_hr(cls, data_fldr, fpm_fov=4096):
        self = cls._hlcfactory(data_fldr, fpm_fov, MDL_SN8_BAND4_HR, thickness_database_band4, 3, override_fpm_dx=0.2)
        self.which = 'HLC'
        # do not have DM design shapes for band 2,3,4
        self.dm1_wfe[:] = 0
        self.dm2_wfe[:] = 0
        return self

    @classmethod
    def hlc_band4_sn8_lr(cls, data_fldr, fpm_fov=256):
        self = cls._hlcfactory(data_fldr, fpm_fov, MDL_SN8_BAND4_LR, thickness_database_band4, 3, override_fpm_dx=2.825422643278202)
        self.which = 'HLC'
        # do not have DM design shapes for band 2,3,4
        self.dm1_wfe[:] = 0
        self.dm2_wfe[:] = 0
        return self

    @classmethod
    def contributed_pcwfsc_parametrized(cls, data_fldr, fpm_fov=1024, fpm_dx=PARAMETRIZED_PCWFS_DEFAULT_SAMPLING,
                                        stack_diameter=EXEP_SIMPLE_OCCULTER_DIAMETER_UM,
                                        l1_ti_thickness=EXEP_SIMPLE_OCCULTER_THICKNESS_TI_NM,
                                        l2_ni_thickness=EXEP_SIMPLE_OCCULTER_THICKNESS_NI_NM,
                                        l3_pmgi_thickness=EXEP_SIMPLE_OCCULTER_BASE_THICKNESS_PMGI_NM,
                                        l3_pmgi_pcwfs_offset=EXEP_SIMPLE_OCCULTER_PCWFS_SPOT_HEIGHT_NM,
                                        l3_pmgi_pcwfs_diameter=EXEP_SIMPLE_OCCULTER_PCWFS_SPOT_DIAMETER_UM):
        """Parametrized version of the contributed mask for the PCWFS.

        See parametrized_hlc_like_pcwfs_spot for additional documentation.

        """
        data_fldr = Path(data_fldr)
        roman_pupil = fits.getdata(data_fldr / 'hlc' / 'run461_pupil_rotated.fits')
        dm1_wfe = np.zeros((309, 309), dtype=config.precision)
        dm2_wfe = np.zeros((309, 309), dtype=config.precision)

        roman_pupil = fttools.crop_center(roman_pupil, 309)

        # double astype => do conversion on CPU to avoid bug in cupy,
        # then convert for dtype homogenaeity
        roman_pupil = np.array(roman_pupil.astype(truenp.float64)).astype(config.precision)

        # this normalization makes the whole diffraction propagation a "unitary"
        # operation.  In other words, diffraction x radiometry = photons
        # the sqrt is because we are normalizing in amplitude space and not
        # intensity space
        roman_pupil /= np.sqrt(roman_pupil.sum())

        dm1_wfe = np.array(dm1_wfe).astype(config.precision)
        dm2_wfe = np.array(dm2_wfe).astype(config.precision)
        hfcr = partial(parametrized_hlc_like_pcwfs_spot, dx=fpm_dx, samples=fpm_fov, polarization='avg', aoi=5.5,
                       stack_diameter=stack_diameter,
                       l1_ti_thickness=l1_ti_thickness,
                       l2_ni_thickness=l2_ni_thickness,
                       l3_pmgi_thickness=l3_pmgi_thickness,
                       l3_pmgi_pcwfs_offset=l3_pmgi_pcwfs_offset,
                       l3_pmgi_pcwfs_diameter=l3_pmgi_pcwfs_diameter)
        fpm = FPMCache(hfcr, data_fldr)
        pupil_mask = None

        return cls(roman_pupil=roman_pupil,
                   dm1_wfe=dm1_wfe,
                   dm2_wfe=dm2_wfe,
                   dx_pup=INHERITED_SAMPLING_PITCH_PUPIL_HLC,
                   npup=309,
                   nmodel=512,
                   prop_dms=True,
                   fpm=fpm,
                   # magic number - from FITS file provided by A.J.
                   fpm_dx=fpm_dx,
                   fpm_samples=fpm_fov,
                   which='HLC',
                   pupil_mask=pupil_mask)

    @classmethod
    def contributed_dual_pcwfs(cls, data_fldr, fpm_fov=1024, sn=4, spot_depth=204):
        """Contributed masks for the PCWFS, which operate in both transmissive and reflective modes.

        Parameters
        ----------
        data_fldr : Path, str, or path_like
            the lowfssim data folder
        fpm_fov : int
            number of samples across the FPM; extent=1024 * 0.5 in microns
        sn : int, {4, 19}
            which substrate serial number
        spot_depth : float
            the depth of the spot; the valid values depend on the SN
            sn4: {174, 190, 204, 214, 231, 241, 244}
            sn19: {166, 175, 190, 204, 218, 229, 231}

        Returns
        -------
        DesignData
            the loaded data for the model

        """
        # this is hacky, but better than lots of copy-paste
        dd = DesignData.hlc_band2_design(data_fldr)
        hfcr = partial(aj_contributed_dual_occulter, sn=sn, spot_depth=spot_depth, fpm_samples=fpm_fov)
        fpm = FPMCache(hfcr, data_fldr)
        dd.fpm = fpm
        dd.dx_fpm = 1
        dd.fpm_samples
        return dd

    @classmethod
    def spec_design(cls, data_fldr, fpm_fov=512):
        """Design simulation setup for SPC SPEC.

        Parameters
        --------
        data_fldr : Path, str, or path_like
            folder containing the SPC SPEC design files.  Must include
            pupil_SPC-20200610_1000.fits
            dm1_surface_m_spc_spec.fits
            dm2_surface_m_spc_spec.fits
            SPM_SPC-20200617_1000_rounded9.fits
        fpm_fov : int
            field of view at the FPM, in samples
            The FPM itself is 381x381 samples, and the resolution is (0.05 fL/D per pixel)
            samples per fL/D.

            The default value need not be changed unless extremely large tilt
            disturbances are used (in which case there will likely be aliasing
            in the phase anyway)

        """
        data_fldr = Path(data_fldr)

        # this file is 1000 sample wide pupil in 1024 array.  Pad up to 2k such
        # that there is no need to do anything but Q=1 when navigating the space
        # before the pupil conjugate before the FPM
        roman_pupil = fits.getdata(data_fldr / 'spc-spec' / 'pupil_SPC-20200610_1000.fits')

        # double astype => do conversion on CPU to avoid bug in cupy,
        # then convert for dtype homogenaeity
        roman_pupil = np.array(roman_pupil.astype(truenp.float32)).astype(config.precision)

        # this normalization makes the whole diffraction propagation a "unitary"
        # operation.  In other words, diffraction x radiometry = photons
        # the sqrt is because we are normalizing in amplitude space and not
        # intensity space
        roman_pupil /= np.sqrt(roman_pupil.sum())

        pupil_mask = fits.getdata(data_fldr / 'spc-spec' / 'SPM_SPC-20200617_1000_rounded9.fits')
        pupil_mask = np.array(pupil_mask.astype(truenp.float32)).astype(config.precision)
        pupil_mask = fttools.pad2d(pupil_mask, Q=1300/pupil_mask.shape[0])  # nominally 1024/1001

        fpm_cmplx_refl = partial(spc_spec_complex_reflection, samples=fpm_fov)
        fpm = FPMCache(fpm_cmplx_refl, data_fldr)
        ss = OAP5_Fs * 0.575 / BEAM_DIA_AT_OAP5 * 0.15
        return DesignData(roman_pupil=roman_pupil,
                          dm1_wfe=None,
                          dm2_wfe=None,
                          dx_pup=INHERITED_SAMPLING_PITCH_PUPIL_SPC,
                          npup=1001,
                          nmodel=1300,
                          prop_dms=False,
                          fpm=fpm,
                          fpm_dx=ss,
                          fpm_samples=fpm_fov,
                          which='SPEC',
                          pupil_mask=pupil_mask)

    @classmethod
    def wfov_design(cls, data_fldr, fpm_fov=512, nmodel=1300):
        """Design simulation setup for SPC WFOV.

        Parameters
        --------
        data_fldr : Path, str, or path_like
            folder containing the SPC SPEC design files.  Must include
            pupil_SPC-20200610_1000.fits
            dm1_surface_m_spc_spec.fits
            dm2_surface_m_spc_spec.fits
            SPM_SPC-20200610_1000_rounded9_gray
        fpm_fov : int
            field of view at the FPM, in samples
            The FPM itself is 245x256 samples, and the resolution is (1/6 fL/D per pixel)
            samples per fL/D.

            The default value need not be changed unless extremely large tilt
            disturbances are used (in which case there will likely be aliasing
            in the phase anyway)

        """
        data_fldr = Path(data_fldr)

        # this file is 1000 sample wide pupil in 1024 array.  Pad up to 2k such
        # that there is no need to do anything but Q=1 when navigating the space
        # before the pupil conjugate before the FPM
        roman_pupil = fits.getdata(data_fldr / 'spc-wfov' / 'pupil_SPC-20200610_1000.fits')

        # double astype => do conversion on CPU to avoid bug in cupy,
        # then convert for dtype homogenaeity
        roman_pupil = np.array(roman_pupil.astype(truenp.float32)).astype(config.precision)

        # this normalization makes the whole diffraction propagation a "unitary"
        # operation.  In other words, diffraction x radiometry = photons
        # the sqrt is because we are normalizing in amplitude space and not
        # intensity space
        roman_pupil /= truenp.sqrt(roman_pupil.sum())

        pupil_mask = fits.getdata(data_fldr / 'spc-wfov' / 'SPM_SPC-20200610_1000_rounded9_gray.fits')
        pupil_mask = np.array(pupil_mask.astype(truenp.float32)).astype(config.precision)
        pupil_mask = fttools.pad2d(pupil_mask, Q=1300/pupil_mask.shape[0])  # nominally 2048/1001

        fpm_cmplx_refl = partial(spc_wfov_complex_reflection, samples=fpm_fov)
        fpm = FPMCache(fpm_cmplx_refl, data_fldr)
        ss = OAP5_Fs * 0.575 / BEAM_DIA_AT_OAP5 / 6
        return DesignData(roman_pupil=roman_pupil,
                          dm1_wfe=None,
                          dm2_wfe=None,
                          dx_pup=INHERITED_SAMPLING_PITCH_PUPIL_SPC,
                          npup=1001,
                          nmodel=1300,
                          prop_dms=False,
                          fpm=fpm,
                          fpm_dx=ss,
                          fpm_samples=fpm_fov,
                          which='WFOV',
                          pupil_mask=pupil_mask)

    @classmethod
    def wfov_design_parametric(cls, data_fldr, pimple_radius, pimple_height):
        """Design simulation setup for SPC WFOV.

        Parameters
        --------
        data_fldr : Path, str, or path_like
            folder containing the SPC SPEC design files.  Must include
            pupil_SPC-20200610_1000.fits
            dm1_surface_m_spc_spec.fits
            dm2_surface_m_spc_spec.fits
            SPM_SPC-20200610_1000_rounded9_gray
        pimple_radius : float
            radius of the pimple, fL/D.  Only 0.05 fL/D steps are handle-able exactly
        pimple_height : float
            pimple height in um

        """
        data_fldr = Path(data_fldr)

        # this file is 1000 sample wide pupil in 1024 array.  Pad up to 2k such
        # that there is no need to do anything but Q=1 when navigating the space
        # before the pupil conjugate before the FPM
        roman_pupil = fits.getdata(data_fldr / 'spc-wfov' / 'pupil_SPC-20200610_1000.fits')

        # double astype => do conversion on CPU to avoid bug in cupy,
        # then convert for dtype homogenaeity
        roman_pupil = np.array(roman_pupil.astype(truenp.float32)).astype(config.precision)

        # this normalization makes the whole diffraction propagation a "unitary"
        # operation.  In other words, diffraction x radiometry = photons
        # the sqrt is because we are normalizing in amplitude space and not
        # intensity space
        roman_pupil /= truenp.sqrt(roman_pupil.sum())

        pupil_mask = fits.getdata(data_fldr / 'spc-wfov' / 'SPM_SPC-20200610_1000_rounded9_gray.fits')
        pupil_mask = np.array(pupil_mask.astype(truenp.float32)).astype(config.precision)
        pupil_mask = fttools.pad2d(pupil_mask, Q=2048/pupil_mask.shape[0])  # nominally 2048/1001

        fpm_fn = partial(parametrized_spc_wfov_complex_reflection,
                         pimple_height=pimple_height, pimple_radius=pimple_radius)
        fpm = FPMCache(fpm_fn, data_fldr)
        ss = OAP5_Fs * 0.575 / BEAM_DIA_AT_OAP5 / 20
        return DesignData(roman_pupil=roman_pupil,
                          dm1_wfe=None,
                          dm2_wfe=None,
                          fpm=fpm, fpm_dx=ss,
                          pupil_mask=pupil_mask)


# FPMCache; like @lru_cache, but more clearable
class FPMCache:
    """Container for FPM realizations at different wavelengths."""

    def __init__(self, func, data_root):
        """Create a new cache, which partial(func, data_root=data_root) if needed."""
        self._call = partial(func, data_root=data_root)
        self._data = {}

    def __call__(self, wvl):
        """FPM complex reflection profile for a given wavelength."""
        out = self._data.get(wvl, None)
        if out is None:
            out = self._call(wvl)
            self._data[wvl] = out

        return out

    def clear(self):
        """Dump cache data."""
        self._data = {}
