"""Rudimentary model of an EMCCD."""

from prysm.mathops import   np
# this file is used to supplant emccd_detect.  The motivation for its existence
# is that it would take 5 hours of compute time to synthesize a chopping dataset
# for LOWFSC, which is unbearably long.  This file is 500 to 500k times faster
# in exchange for a modest reduction in accuracy.
# benched emccd_detect 2.0.0a0 at 120ms/frame
# emccd.py at 248 usec/frame (single) or 145usec/frame (10,000 frames)
# GPU bench at 402 usec/frame (single) or 5.8usec/frame (10,000 frames)
# speedup = 5.8e-3/120 ~= 5e5 = 500,000x

# easter egg
np.random.seed(0x434749)


def render_columnar_fpn(img_size, ncols, height, dtype=None):
    """Render per-column fixed pattern noise.

    Parameters
    ----------
    img_size : `int` or `tuple` of `int`
        size of the image, (img_size,img_size) px, or img_size if img_size is already a tuple
        e.g., 50 => (50,50) or (25,50) => (25,50)
    ncols : `int`
        number of columns to make noisy
    height : `float`
        "height" of the FPN; same units as the image it will be added to (likely DN).
    dtype : numpy datatype, optional
        a dtype like np.float64 or np.float32.  Same default as numpy (float64) if not given.

    Returns
    -------
    `numpy.ndarray`
        ndarray containing fixed pattern noise.

    """

    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    columns = np.arange(img_size[0])
    chosen_cols = np.random.choice(columns, ncols, replace=False)
    heights = np.random.uniform(-height, height, ncols)
    out = np.zeros(img_size, dtype=dtype)
    # sorting the columns produces a superior memory access pattern.
    # As the array becomes large, this speeds up the function.
    chosen_cols = np.sort(chosen_cols)
    heights = heights[np.newaxis, :]
    out[:, chosen_cols] = heights
    return out


def render_rowwise_fpn(img_size, nrows, height, dtype=None):
    """Render per-row fixed pattern noise.

    Parameters
    ----------
    img_size : `int` or `tuple` of `int`
        size of the image, (img_size,img_size) px, or img_size if img_size is already a tuple
        e.g., 50 => (50,50) or (25,50) => (25,50)
    ncols : `int`
        number of columns to make noisy
    height : `float`
        "height" of the FPN; same units as the image it will be added to (likely DN).
    dtype : numpy datatype, optional
        a dtype like np.float64 or np.float32.  Same default as numpy (float64) if not given.

    Returns
    -------
    `numpy.ndarray`
        ndarray containing fixed pattern noise.

    """
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    rows = np.arange(img_size[1])
    chosen_rows = np.random.choice(rows, nrows, replace=False)
    heights = np.random.uniform(-height, height, nrows)
    out = np.zeros(img_size, dtype=dtype)
    # sorting the rows produces a superior memory access pattern.
    # As the array becomes large, this speeds up the function.
    chosen_rows = np.sort(chosen_rows)
    heights = heights[:, np.newaxis]
    out[chosen_rows, :] = heights
    return out


def apply_lut(img, lut):
    """Apply a lookup table to img.

    Parameters
    ----------
    img : `numpy.ndarray`
        n dimensional array (2D and 3D are both OK) of an unsigned integer dtype
    lut : `numpy.ndarray`
        1 dimensional array whose indices are input values and values are output values

    Returns
    -------
    `numpy.ndarray`
        ndarray of the same shape as img
        the output array must not be modified in place, or lut will be modified as well.

    """
    # take is faster than indexing into the lut on older numpy
    return np.take(lut, img)


class EMCCD:
    """Special EMCCD model."""

    def __init__(self, dark_current, cic, read_noise, bias, em_gain, fwc,
                 conversion_gain, bits, exposure_time, frame_time,
                 nonlinear_lut=None):
        """Create a new EMCCD instance.

        Parameters
        ----------
        dark_current : `float`
            dark current rate, e-/sec
        cic : `float`
            clock induced charge, e-/frame
        read_noise : `float`
            read noise, output e- (after EM gain)
        bias : `float`
            bias, output e- (after EM gain)
        em_gain : `float`
            em gain (x, multiplier)
        fwc : `float`
            full-well capacity, e- (after EM gain)
        conversion_gain : `float`
            e- per ADU or DN
        bits : `int`
            bit depth of the camera
        exposure_time : `float`
            integration time, sec
        frame_time : `float`
            total time for 1 frame, sec
        nonlinear_lut : `numpy.ndarray`, optional
            a 1D lookup table containing the camera nonlinear response
            indices of the array are the input pixel values and values
            are the output values.

            For example, the array [0, 3, 10] would transform the input data
            [0, 1, 2, 1, 2, 2, 2] => [0, 3, 10, 3, 10, 10, 10]

        """
        self.dark_current = dark_current
        self.cic = cic
        self.read_noise = read_noise
        self.bias = bias
        self.em_gain = em_gain
        self.fwc = fwc
        self.conversion_gain = conversion_gain
        self.bits = bits
        self.exposure_time = exposure_time
        self.frame_time = frame_time
        self.nonlinear_lut = nonlinear_lut

    def expose(self, flux_map, frames=1):
        """Expose a flux map.

        Parameters
        ----------
        flux_map : `numpy.ndarray`
            ndarray which has units of e-/sec, must include any desired QE of
            the camera
        frames : `int`
            number of frames of data to synthesize.  Beware that large numbers
            for a large flux map size may cause an out of memory error.  Consider
            batching if your output would exceed ~125M values.

        Returns
        -------
        `numpy.ndarray`
            ndarray of shape (frames, *flux_map.shape)
            if frames=1, the first dim is squeezed off
            dtype=uint16, if nbits <= 16, uint32 if nbits <= 32, else uint64.

        """
        # from the input flux, determine the expected number of photons and draw
        # random noise realizations
        inp = flux_map * self.exposure_time

        dark_expectation_value = self.dark_current * self.exposure_time + self.cic
        expectation_value = inp + dark_expectation_value

        expectation_value = expectation_value.ravel()
        with_shot_noise = np.random.poisson(expectation_value, (frames, expectation_value.size))
        # given input photon noise realizations, draw samples from the gamma
        # distribution per /stochastic model for EMCCD/ Hirsch et al 2013
        # with shape parameter of input photoelectrons and scale parameter
        # theta = g - 1 + (1 / input photoelectrons)
        # then xi = gamma(xi;k,theta)
        # for xi = output electrons - input electrons + 1

        # for k = 0, output = 0.  stick a safe value in k for zero, which would
        # cause div by zero for theta.  At output stage, insert known value
        k = with_shot_noise
        mask = k == 0
        k[mask] = 1
        theta = self.em_gain - 1 + 1 / k
        post_em_gain = np.random.gamma(k, theta) + k - 1
        post_em_gain[mask] = 0

        # read noise is zero mean
        rn = np.random.normal(0, self.read_noise, with_shot_noise.shape)

        output = post_em_gain + self.bias + rn
        output[output > self.fwc] = self.fwc
        output /= self.conversion_gain
        adc_cap = 2 ** self.bits
        output[output < 0] = 0
        output[output > adc_cap] = adc_cap
        if self.bits <= 16:
            output = output.astype(np.uint16)
        elif self.bits <= 32:
            output = output.astype(np.uint32)
        else:
            output = output.astype(np.uint64)

        output = output.reshape((frames, *inp.shape))
        if frames == 1:
            output = output[0, :, :]

        if self.nonlinear_lut is not None:
            output = apply_lut(output, self.nonlinear_lut)

        return output

    @classmethod
    def cgi_camera(cls):
        """Parameters are taken from emccd_detect, v2.0.0a0."""
        return EMCCD(
            dark_current=0.0028,
            cic=0.02,
            read_noise=200,
            bias=10_000,
            em_gain=1000,
            fwc=110_000,
            conversion_gain=7,
            bits=14,
            exposure_time=441e-6,
            frame_time=1e-3
        )


class sCMOS:
    """Ordinary sCMOS model."""

    def __init__(self, dark_current, read_noise, bias, fwc,
                 conversion_gain, bits, exposure_time, nonlinear_lut=None):
        """Create a new sCMOS instance.

        Parameters
        ----------
        dark_current : `float`
            dark current rate, e-/sec
        cic : `float`
            clock induced charge, e-/frame
        read_noise : `float`
            read noise, output e- (after EM gain)
        bias : `float`
            bias, output e- (after EM gain)
        em_gain : `float`
            em gain (x, multiplier)
        fwc : `float`
            full-well capacity, e- (after EM gain)
        conversion_gain : `float`
            e- per ADU or DN
        bits : `int`
            bit depth of the camera
        exposure_time : `float`
            integration time, sec
        nonlinear_lut : `numpy.ndarray`, optional
            a 1D lookup table containing the camera nonlinear response
            indices of the array are the input pixel values and values
            are the output values.

            For example, the array [0, 3, 10] would transform the input data
            [0, 1, 2, 1, 2, 2, 2] => [0, 3, 10, 3, 10, 10, 10]

        """
        self.dark_current = dark_current
        self.read_noise = read_noise
        self.bias = bias
        self.fwc = fwc
        self.conversion_gain = conversion_gain
        self.bits = bits
        self.exposure_time = exposure_time
        self.nonlinear_lut = nonlinear_lut

    def expose(self, flux_map, frames=1):
        """Expose a flux map.

        Parameters
        ----------
        flux_map : `numpy.ndarray`
            ndarray which has units of e-/sec, must include any desired QE of
            the camera
        frames : `int`
            number of frames of data to synthesize.  Beware that large numbers
            for a large flux map size may cause an out of memory error.  Consider
            batching if your output would exceed ~125M values.

        Returns
        -------
        `numpy.ndarray`
            ndarray of shape (frames, *flux_map.shape)
            if frames=1, the first dim is squeezed off
            dtype=uint16, if nbits <= 16, uint32 if nbits <= 32, else uint64.

        """
        # from the input flux, determine the expected number of photons and draw
        # random noise realizations
        inp = flux_map * self.exposure_time

        dark_expectation_value = self.dark_current * self.exposure_time
        expectation_value = inp + dark_expectation_value

        expectation_value = expectation_value.ravel()
        with_shot_noise = np.random.poisson(expectation_value, (frames, expectation_value.size))

        # read noise is zero mean
        rn = np.random.normal(0, self.read_noise, with_shot_noise.shape)

        output = with_shot_noise + self.bias + rn
        output[output > self.fwc] = self.fwc
        output /= self.conversion_gain
        adc_cap = 2 ** self.bits
        output[output < 0] = 0
        output[output > adc_cap] = adc_cap
        if self.bits <= 16:
            output = output.astype(np.uint16)
        elif self.bits <= 32:
            output = output.astype(np.uint32)
        else:
            output = output.astype(np.uint64)

        output = output.reshape((frames, *inp.shape))
        if frames == 1:
            output = output[0, :, :]

        if self.nonlinear_lut is not None:
            output = apply_lut(output, self.nonlinear_lut)

        return output
