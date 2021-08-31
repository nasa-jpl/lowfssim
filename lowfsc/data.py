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

from astropy.io import fits

from prysm import (
    coordinates,
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
)


PMGI_DESIGN = 'run461_theta6.69imthicks_PMGIfnum32.5676970504_lam5.75e-07_.fits'

thickness_database = {
    'Ni':   'run461_theta6.69imthicks_nifnum32.5676970504_lam5.75e-07_.fits',
    'Ti':   'run461_theta6.69imthicks_tifnum32.5676970504_lam5.75e-07_.fits',
}

def cropcenter(img, out_shape):
    """Crop the central (out_shape) of an image, with FFT alignment.

    As an example, if img=512x512 and out_shape=200
    out_shape => 200x200 and the returned array is 200x200, 156 elements from the [0,0]th pixel

    This function is the adjoint of padcenter.

    Parameters
    ----------
    img : `numpy.ndarray`
        ndarray of shape (m, n)
    out_shape : `int` or `iterable` of int
        shape to crop out, either a scalar or pair of values

    """
    if isinstance(out_shape, int):
        out_shape = (out_shape, out_shape)

    padding = [i-o for i, o in zip(img.shape, out_shape)]
    left = [p//2 for p in padding]
    slcs = tuple((slice(l, l+o) for l, o in zip(left, out_shape)))  # NOQA -- l ambiguous
    return img[slcs]


def tabular_index(matl, wavelength, data_root):
    """Index of refraction of a material.

    Parameters
    ----------
    matl : `str`
        a material
    wavelength : `float`
        wavelength of light, microns

    Returns
    -------
    `float` or `complex`
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
    material : `str`, {'PMGI', 'Ni', "Ti', 'C7980'}
        a material used in this problem
    wavelength : `float` or `numpy.ndarray`
        a wavelength (float) or array of them, microns
    data_root : `Path`, `str`, or path_like
        location containing refractive index data

    Returns
    -------
    `float` or `numpy.ndarray`
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


def thickness_data(material, data_root, pmgi_fn):
    """Spatially-varying thickness of a material.

    Parameters
    ----------
    material : `str`, {'PMGI', 'Ni', 'Ti', 'C7980'}
        a material used in the FPM stack.
    data_root : `Path`, `str`, or path_like
        location containing the thickness data files
    pmgi_fn : `str`
        filename to use for PMGI data; see lowfssim.data.
            PMGI_DESIGN
            PMGI_OMC
            PMGI_MDL_SPIN1_ALL_DOF

    Returns
    -------
    `numpy.ndarray`
        an array of the material's thickness.  52x52 unless the FITS files change.

    """
    td = partial(thickness_data, data_root=data_root, pmgi_fn=pmgi_fn)
    if material == 'COMP':
        return -1 * (td('PMGI') + td('Ni') + td('Ti'))
    if material == 'C7980':
        return truenp.ones_like(td('Ni')) * np.Inf
    else:
        if material == 'PMGI':
            fn = pmgi_fn
        else:
            fn = thickness_database[material]
        with fits.open(data_root / fn) as hdu:
            dat = hdu[0].data.astype(config.precision) * 1e6  # m -> um
            dat = truenp.roll(dat, -1, axis=0)  # roll keeps data centered on pimple
            return truenp.flipud(dat)


def hlc_fpm_complex_reflection(wavelength, data_root, pmgi_fn, substrate_R=0.001, polarization='avg', aoi=5.5, samples=512):
    """Complex reflection of the FPM for a given wavelength.

    This is not optimized/vectorized but it is cached

    Parameters
    ----------
    wavelength : `float`
        wavelength of light, microns
    data_root : `Path`, `str`, or path_like
        location containing the thickness and refractive index data files
    pmgi_fn : `str`
        filename to use for PMGI data; see lowfssim.data.
            PMGI_DESIGN
            PMGI_OMC
            PMGI_MDL_SPIN1_ALL_DOF
    substrate_R : `float`
        reflectance value used to paint the region outside the nickel with
    polarization : `str`, {'r', 's', 'avg'}
        which polarization state to use.  If avg, (r-s)/2, which accounts for phase
    aoi : `float`
        angle of incidence, degrees
    samples : `int`
        number of samples in the padded output

    Returns
    -------
    `numpy.ndarray`
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
    thicknesses = [thickness_data(m, data_root, pmgi_fn) for m in matls]
    indices = [refractive_data(m, wavelength, data_root.parent) for m in matls]
    # return thicknesses, indices
    shp = thicknesses[0].shape
    out = truenp.zeros(shp, dtype=config.precision_complex)
    for i in range(shp[0]):
        for j in range(shp[1]):
            thk = [a[i, j] for a in thicknesses]
            stack = [[i, t] for i, t in zip(indices, thk)]
            if polarization in {'p', 's'}:
                rs, _ = thinfilm.multilayer_stack_rt(
                    polarization=polarization,
                    stack=stack,
                    wavelength=wavelength,
                    aoi=aoi)
                out[i, j] = rs
            else:
                rp, _ = thinfilm.multilayer_stack_rt(
                    polarization='p',
                    stack=stack,
                    wavelength=wavelength,
                    aoi=aoi)
                rs, _ = thinfilm.multilayer_stack_rt(
                    polarization='s',
                    stack=stack,
                    wavelength=wavelength,
                    aoi=aoi)
                out[i, j] = (rp + rs) / 2

    out[thicknesses[1] == 0] = truenp.sqrt(substrate_R)
    out = fttools.pad2d(out, Q=samples/52, mode='constant', value=out[0, 0])
    np._srcmodule = old
    return np.array(out)


def spc_spec_complex_reflection(wavelength, data_root, substrate_R=0.001, polarization='avg', aoi=5.5, samples=512):
    """Complex reflection of the FPM for a given wavelength.

    Parameters
    ----------
    wavelength : `float`
        wavelength of light, microns
    data_root : `Path`, `str`, or path_like
        location containing the thickness and refractive index data files
    substrate_R : `float`
        reflectance value used to paint the region outside the nickel with
    polarization : `str`, {'r', 's', 'avg'}
        which polarization state to use.  If avg, (r-s)/2, which accounts for phase
    aoi : `float`
        angle of incidence, degrees
    samples : `int`
        number of samples in the padded output

    Returns
    -------
    `numpy.ndarray`
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
        (refractive_data('C7980', wavelength, data_root), 6350),
    ]
    stack_lowfs = [
        (refractive_data('Al', wavelength, data_root), AL_BASELINE_THICKNESS+SPC_PIMPLE_HEIGHT),
        (refractive_data('C7980', wavelength, data_root), 6350),
    ]
    if polarization in {'p', 's'}:
        r_bg, _ = thinfilm.multilayer_stack_rt(polarization, wavelength, stack_bg, aoi=aoi)
        r_lwf, _ = thinfilm.multilayer_stack_rt(polarization, wavelength, stack_lowfs, aoi=aoi)
    else:
        r_bg_p, _ = thinfilm.multilayer_stack_rt('p', wavelength, stack_bg, aoi=aoi)
        r_lwf_p, _ = thinfilm.multilayer_stack_rt('p', wavelength, stack_lowfs, aoi=aoi)
        r_bg_s, _ = thinfilm.multilayer_stack_rt('s', wavelength, stack_bg, aoi=aoi)
        r_lwf_s, _ = thinfilm.multilayer_stack_rt('s', wavelength, stack_lowfs, aoi=aoi)
        r_bg = (r_bg_p + r_bg_s) / 2
        r_lwf = (r_lwf_p + r_lwf_s) / 2

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
    wavelength : `float`
        wavelength of light, microns
    data_root : `Path`, `str`, or path_like
        location containing the thickness and refractive index data files
    substrate_R : `float`
        reflectance value used to paint the region outside the nickel with
    polarization : `str`, {'r', 's', 'avg'}
        which polarization state to use.  If avg, (r-s)/2, which accounts for phase
    aoi : `float`
        angle of incidence, degrees
    samples : `int`
        number of samples in the padded output

    Returns
    -------
    `numpy.ndarray`
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
        (refractive_data('C7980', wavelength, data_root), 6350),
    ]
    stack_lowfs = [
        (refractive_data('Al', wavelength, data_root), AL_BASELINE_THICKNESS+SPC_PIMPLE_HEIGHT),
        (refractive_data('C7980', wavelength, data_root), 6350),
    ]
    if polarization in {'p', 's'}:
        r_bg, _ = thinfilm.multilayer_stack_rt(polarization, wavelength, stack_bg, aoi=aoi)
        r_lwf, _ = thinfilm.multilayer_stack_rt(polarization, wavelength, stack_lowfs, aoi=aoi)
    else:
        r_bg_p, _ = thinfilm.multilayer_stack_rt('p', wavelength, stack_bg, aoi=aoi)
        r_lwf_p, _ = thinfilm.multilayer_stack_rt('p', wavelength, stack_lowfs, aoi=aoi)
        r_bg_s, _ = thinfilm.multilayer_stack_rt('s', wavelength, stack_bg, aoi=aoi)
        r_lwf_s, _ = thinfilm.multilayer_stack_rt('s', wavelength, stack_lowfs, aoi=aoi)
        r_bg = (r_bg_p + r_bg_s) / 2
        r_lwf = (r_lwf_p + r_lwf_s) / 2

    r_bg = r_bg.real    # explicitly zero phase, going to add in phase for _lwf
    r_lwf = r_lwf.real  # but not _bg
    r_lwf = r_lwf * np.exp(-1j * 2 * np.pi / wavelength * -2*SPC_PIMPLE_HEIGHT)
    arr = np.ones(ring.shape, config.precision_complex) * (substrate_R**2)
    arr[ring == 0] = r_bg
    arr[pimple] = r_lwf
    arr = fttools.pad2d(arr, Q=samples/arr.shape[0], mode='constant', value=arr[0, 0])
    np._srcmodule = old
    return np.array(arr)


class DesignData():
    """A class which prevents you from having to go to disk to get an array."""

    def __init__(self, roman_pupil, dm1_wfe, dm2_wfe, dx_pup, npup, nmodel, prop_dms, fpm, fpm_dx, fpm_samples, which, pupil_mask=None):
        """Create a new DesignData instance.

        Parameters
        ----------
        roman_pupil : `numpy.ndarray`
            ndarray of roman pupil amplitude, i.e. sqrt(Intensity)
        dm1_wfe : `numpy.ndarray`
            array of wavefront error (OPD) applied by DM1, with same array shape
            as roman_pupil.  Units of nm.
        dm2_wfe : `numpy.ndarray`
            as dm1_wfe, but for DM2.
        dx_pup : `float`
            sample spacing of the pupil plane, microns
        npup : `int`
            number of samples spanned by the Roman pupil
        nmodel : `int`
            number of samples spanning the array used in the model
        prop_dms : `bool`
            if True, propagates between and applies the DMs; if false, ignores
            the DMs
        fpm : `callable`
            focal plane mask generator.  Should accept a single argument of "wvl"
            which is the wavelength at which to compute the complex reflectivity
        fpm_dx : `float`
            inter-sample spacing at the FPM, microns
        fpm_samples : `int`
            number of samples across the FPM
        which : `str`
            either HLC or SPC, specifies which model to run
            The models have different sampling, so this flag is needed
        pupil_mask : `numpy.ndarray`
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

    def seed_zernikes(self, jnoll=None, nms=None):
        """Compute and save Zernike polnyomials.

        Parameters
        ----------
        jnoll : iterable of `int`, optional
            [1,2,3,4,5] Noll indexed Zernike description (base 1;1=piston)
            a range object works
        nms : iterable of `tuple`, optional
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

    @classmethod
    def hlc_design(cls, data_fldr, fpm_fov=256):
        """Design simulation setup for HLC.

        Parameters
        ----------
        data_fldr : `Path`, `str`, or `path_like`
            folder containing the HLC design files.  Must include
            run461_pupil_rotated.fits
            run461_dm1wfe.fits
            run461_dm2wfe.fits
            run461_theta6.69imthicks_nifnum32.5676970504_lam5.75e-07_.fits
            run461_theta6.69imthicks_tifnum32.5676970504_lam5.75e-07_.fits
            run461_theta6.69imthicks_PMGIfnum32.5676970504_lam5.75e-07_.fits
        fpm_fov : `int`
            field of view at the FPM, in samples
            The FPM itself is 54x54 samples, and the resolution is (2048/309)
            samples per fL/D.

            The default value need not be changed unless extremely large tilt
            disturbances are used (in which case there will likely be aliasing
            in the phase anyway)

        """
        data_fldr = Path(data_fldr) / 'hlc'
        roman_pupil = fits.getdata(data_fldr / 'run461_pupil_rotated.fits')
        dm1_wfe = fits.getdata(data_fldr / 'run461_dm1wfe.fits') * 1e9
        dm2_wfe = fits.getdata(data_fldr / 'run461_dm2wfe.fits') * 1e9
        roman_pupil = cropcenter(roman_pupil, 309)
        dm1_wfe = cropcenter(dm1_wfe, 309)
        dm2_wfe = cropcenter(dm2_wfe, 309)
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
        hfcr = partial(hlc_fpm_complex_reflection, pmgi_fn=PMGI_DESIGN, samples=fpm_fov)
        fpm = FPMCache(hfcr, data_fldr)
        pupil_mask = None
        return DesignData(roman_pupil=roman_pupil,
                          dm1_wfe=dm1_wfe,
                          dm2_wfe=dm2_wfe,
                          dx_pup=INHERITED_SAMPLING_PITCH_PUPIL_HLC,
                          npup=309,
                          nmodel=512,
                          prop_dms=True,
                          fpm=fpm,
                          # magic number - from FITS file provided by A.J.
                          fpm_dx=2.825422643278202,
                          fpm_samples=fpm_fov,
                          which='HLC',
                          pupil_mask=pupil_mask)

    @classmethod
    def spec_design(cls, data_fldr, fpm_fov=512):
        """Design simulation setup for SPC SPEC.

        Parameters
        --------
        data_fldr : `Path`, `str`, or `path_like`
            folder containing the SPC SPEC design files.  Must include
            pupil_SPC-20200610_1000.fits
            dm1_surface_m_spc_spec.fits
            dm2_surface_m_spc_spec.fits
            SPM_SPC-20200617_1000_rounded9.fits
        fpm_fov : `int`
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
        data_fldr : `Path`, `str`, or `path_like`
            folder containing the SPC SPEC design files.  Must include
            pupil_SPC-20200610_1000.fits
            dm1_surface_m_spc_spec.fits
            dm2_surface_m_spc_spec.fits
            SPM_SPC-20200610_1000_rounded9_gray
        fpm_fov : `int`
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
