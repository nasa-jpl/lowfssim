"""Reconstruction routines."""

import numpy as truenp

from prysm.mathops import np


# some utilities here, declaration order is relevant in python
def prepare_Zmm(zernike_chops, flux_mode=None, shear_chops=None, mask=None):
    """Assemble Zmm from chop images.

    Parameters
    ----------
    zernike_chops : `iterable` of numpy.ndarray
        set of zernike chops.  Assumed to begin at Z2.  Need not span as far as Z11.
        Each image need not be raveled.  "Gain" should be applied beforehand,
        such that these differential images are scaled to represent 1 nm RMS
    flux_mode : `numpy.ndarray`
        dark-subtracted reference frame; internally rescaled so that
        the output ranges from [-100, +inf], but realistically stopping
        long before +100.
    shear_chops : `iterable` of numpy.ndarray
        a lenght 2 iterable of pupil shear images.  "Gain" should be applied
        beforehand, such that these differential images are scaled to represent
        1 px of shear
    mask : `numpy.ndarray`
        mask to apply to the modes

    Returns
    -------
    numpy.ndarray
        array aligned to the flight sequence ready to be passed to a reconstructor
        init func.

    """
    if len(zernike_chops) > 10:
        raise ValueError('must have 10 or fewer Zernike modes')

    if shear_chops is not None:
        if len(shear_chops) != 2:
            raise ValueError('shear_chops must be length 2')
    else:
        s = np.zeros_like(zernike_chops[0])
        shear_chops = (s, s)

    if flux_mode is None:
        flux_mode = np.zeros_like(zernike_chops[0])

    if mask is None:
        mask = np.ones_like(zernike_chops[0])

    mask = mask.ravel()
    shear_chops = [s.ravel() * mask for s in shear_chops]
    flux_mode = flux_mode.ravel() / 100 * mask
    out = np.zeros((zernike_chops[0].size, 17), zernike_chops[0].dtype)
    for i, mode in enumerate(zernike_chops):
        # i starts at 0
        # slot 0 is unused
        out[:, i+1] = (mode.ravel() * mask)

    out[:, 0] = 1  # bias mode
    out[:, 11] = flux_mode
    out[:, 12] = shear_chops[0].ravel()
    out[:, 13] = shear_chops[1].ravel()
    return out


def reconstruct(A, P, Q):
    """Estimate (reconstruct) modes from an image.

    Assumes that c has the alignment of the flight estimator,
    [na Z2 .. Z11 F1 Sx Sy na na na]
    and populates the final element with the sum of A

    Parameters
    ----------
    A : `numpy.ndarray`
        the image, with masked pixels removed and raveled
    P : `numpy.ndarray`
        P from Eq. (40)
    Q : `numpy.ndarray`
        Q from Eq. (41)


    Returns
    -------
    `numpy.ndarray`
        array of shape (n modes)

    """
    c = P @ A + Q
    return c


def vmag_normalize(Zmm_old, old_ref, new_ref, new_sx, new_sy, mask):
    """Perform vmag normalization to accomodate a mode matrix for change in flux.

    Does not apply the mask to the old Zernike modes

    Parameters
    ----------
    Zmm_old : `numpy.ndarray`
        prior Zmm matrix, with I0 (flux mode) included
    old_ref : `numpy.ndarray`
        old reference frame, dark subtracted
    new_ref : `numpy.ndarray`
        new reference frame, dark subtracted
    new_sx : `numpy.ndarray`
        new shear x frame
    new_sy : `numpy.ndarray`
        new shear y frame
    mask : `numpy.ndarray`
        mask array

    Returns
    -------
    `numpy.ndarray`
        new Zmm, with no common memory or elements to Zmm_old

    """
    mask = mask.ravel()
    s_old = old_ref.sum()
    s_new = new_ref.sum()
    change_in_flux = s_new / s_old
    Zmm2 = Zmm_old.copy() * change_in_flux
    Zmm2[:, 11] = new_ref.ravel() / 100 * mask
    Zmm2[:, 12] = new_sx.ravel() * mask
    Zmm2[:, 13] = new_sy.ravel() * mask
    return Zmm2


class ReconstructorV3pt5:
    """Holds the needed metadata arrays and performs reconstruction of Zernike modes."""

    def __init__(self, Zmm, ref, dark=None):
        """Create a new reconstructor.  This is just a container for the state.

        Make a new Reconstructor if you wish to update any of the arguments.
        They do not store any history, unless you consider the chops history.  A
        hot swap would have no ill effects.

        Parameters
        ----------
        Zmm : `numpy.ndarray`
            2500x17 array with modes as columns, flight "alignment"
        ref : `numpy.ndarray`
            reference image
        dark : `numpy.ndarray`
            dark frame, not used in this generation of estimator, but included
            as a parameter for API compatibility

        """
        self.ref = ref
        self.dark = dark
        self.Zmm = Zmm
        self.P, self.Q = assemble_reconstructor(Zmm, ref.ravel())

    def estimate_raw(self, img):
        """Perform reconstruction (estimation) on an image.

        You should only call this function if you are interested in the flux mode.

        The difference between it and estimate is that it returns the flux mode
        coefficient and has no other options.

        """
        return reconstruct(img.ravel(), self.P, self.Q)

    def estimate(self, img):
        """Perform reconstruction (estimation) on an image.

        Parameters
        ----------
        img : `numpy.ndarray`
            ndarray of shape (MxN), right out of the camera

        Returns
        -------
        `numpy.ndarray`
            a vector of coefficients

        """
        coefs = reconstruct(img.ravel(), self.P, self.Q)
        espilon_by_f = coefs[11]
        # epsilon_by_f is the flux mode
        coefs[:11] /= (1 + espilon_by_f)
        return coefs

    def vmag_normalize(self, new_ref):
        """Perform vmag normalization.

        Does not expose options for shear chop size, etc.  These could be added later.

        Parameters
        ----------
        new_ref : `numpy.ndarray`
            new reference frame

        """
        new_sy, new_sx = (synthesize_pupil_shear(new_ref, 0.038, a) for a in [0, 1])
        mask = self.Zmm[:, 11] == 0
        Zmm2 = vmag_normalize(self.Zmm, self.ref-self.dark, new_ref-self.dark, new_sx, new_sy, ~mask)
        return self.__class__(Zmm2, new_ref, self.dark)


class ReconstructorV2:
    """Second generation reconstructor."""

    def __init__(self, Zmm, ref, dark=None):
        """Create a new reconstructor.  This is just a container for the state.

        Make a new Reconstructor if you wish to update any of the arguments.
        They do not store any history, unless you consider the chops history.  A
        hot swap would have no ill effects.

        Include the flux mode in the 12th (11th indexed) column, or don't.
        The array is copied and the 12th mode is zeroed before Zmm is used.

        Parameters
        ----------
        Zmm : `numpy.ndarray`
            2500x17 array with modes as columns, flight "alignment"
        ref : `numpy.ndarray`
            reference image
        dark : `numpy.ndarray`
            dark frame, not used in this generation of estimator, but included
            as a parameter for API compatibility

        """
        self.ref = ref
        if dark is None:
            dark = np.zeros_like(ref)

        # kill the flux mode if it exists
        Zmm = Zmm.copy()
        Zmm[:, 11] = 0
        self.Zmm = Zmm
        self.ref = ref
        self.dark = dark
        self.P, self.Q = assemble_reconstructor(Zmm, ref.ravel(), dark.ravel())

    def estimate(self, img):
        """Perform reconstruction (estimation) on an image."""
        return reconstruct(img.ravel(), self.P, self.Q)


class ReconstructorV2pt5:
    """Second generation reconstructor, with flux mode."""

    def __init__(self, Zmm, ref, dark=None):
        """Create a new reconstructor.  This is just a container for the state.

        Make a new Reconstructor if you wish to update any of the arguments.
        They do not store any history, unless you consider the chops history.  A
        hot swap would have no ill effects.

        Include the flux mode in the 12th (11th indexed) column.

        Parameters
        ----------
        Zmm : `numpy.ndarray`
            2500x17 array with modes as columns, flight "alignment"
        ref : `numpy.ndarray`
            reference image
        dark : `numpy.ndarray`
            dark frame, not used in this generation of estimator, but included
            as a parameter for API compatibility

        """
        self.ref = ref
        if dark is None:
            dark = np.zeros_like(ref)

        self.Zmm = Zmm
        self.ref = ref
        self.dark = dark
        self.P, self.Q = assemble_reconstructor(Zmm, ref.ravel(), dark.ravel())

    def estimate(self, img):
        """Perform reconstruction (estimation) on an image."""
        return reconstruct(img.ravel(), self.P, self.Q)

    def vmag_normalize(self, new_ref):
        """Perform vmag normalization.

        Does not expose options for shear chop size, etc.  These could be added later.

        Parameters
        ----------
        new_ref : `numpy.ndarray`
            new reference frame

        """
        new_sy, new_sx = (synthesize_pupil_shear(new_ref, 0.038, a) for a in [0, 1])
        mask = self.Zmm[:, 11] == 0
        Zmm2 = vmag_normalize(self.Zmm, self.ref-self.dark, new_ref-self.dark, new_sx, new_sy, ~mask)
        return self.__class__(Zmm2, new_ref, self.dark)


def build_mode(ref, chop):
    """Build a mode according to E. Cady's Eq. (24) from lowfs_recon_v2.pdf.

    Parameters
    ----------
    ref : `numpy.ndarray`
        raw reference image, without mask applied or dark frame subtracted
    chop : `numpy.ndarray`
        chop with positive stimulus.  Raw image.

    Returns
    -------
    `numpy.ndarray`
        raveled ndarray after applying mask, shape of NxM - (NxM-sum(mask))

    """
    return chop - ref


def assemble_reconstructor(Zmm, ref, dark):
    """Assemble a reconstruction P, Q from any number of modes.

    Computes P, Q from Eq. 3, 12 of lowfs-recon-v3.5.pdf from B. Dube

    Parameters
    ----------
    Zmm : `numpy.ndarray`
        2500x17 array with modes along columns
    ref : `numpy.ndarray`
        reference frame, can be 50x50 or flat.  Not dark subtracted.  masked.
    dark : `numpy.ndarray`
        dark frame, can be 50x50 or flat.  masked.

    Returns
    -------
    P : `numpy.ndarray`
        Pseudo-inverse of the modes, ndarray of shape (len(modes), MxN)
    Q : `numpy.ndarray`
        "offset" array that has the function of b in the equation y = Ax + b

    """
    ref = ref.ravel()
    dark = dark.ravel()

    oldtype = Zmm.dtype
    Zmm = Zmm.astype(np.float64)
    P = np.linalg.pinv(Zmm)
    P = P.astype(oldtype)
    Q = -(P @ ref)
    return P, Q


def synthesize_pupil_shear(ref, samples, axis=0, order=3):
    """Create a pupil shear mimic from the reference frame.

    Parameters
    ----------
    ref : `numpy.ndarray`
        2D array containing the reference frame
    samples : `float`
        number of samples as ref is given to shear by.  Notionally, ref has
        13 um sample spacing, and so samples=1 shears by 13 um.
    axis : `int`
        which numpy axis (0=column, 1=row) to shift along

    Returns
    -------
    `numpy.ndarray`
        2D array with shift applied

    """
    # there is a little gymnastics in this function to move a GPU array to CPU
    # and back for ndimage shift if needed
    from scipy import ndimage
    if type(ref) != truenp.ndarray:
        ref = ref.get()
    shifts = [0, 0]
    shifts[axis] = samples
    shifts2 = [0, 0]
    shifts2[axis] = -samples
    # https://github.jpl.nasa.gov/WFIRST-CGI/cgisim-wrapper/blob/master/lowfsparam/lowfsparam.py#L220
    shiftedup = ndimage.shift(ref, shifts, order=3)  # no other args, same as WFSC
    shifteddown = ndimage.shift(ref, shifts2, order=3)  # no other args, same as WFSC
    mode = (shiftedup - shifteddown)/(2*samples)
    return np.asarray(mode)


# alias newest to reconstructor
Reconstructor = ReconstructorV2pt5
