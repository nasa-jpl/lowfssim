"""Automatic routines for performing certain operations."""


from prysm.conf import config
from prysm.mathops import np

from . import props
from .reconstruction import synthesize_pupil_shear, Reconstructor as RV25, prepare_Zmm

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def chop_bipolar(wvl, weights, design_data, ref_z, chop_zs, cam=None, nframes_avg=10_000):
    """Do bipolar chopping (up-down)/2 about a particular point.

    Parameters
    ----------
    wvl : numpy.ndarray
        wavelengths of light, um
    weights : numpy.ndarray
        weights associated with the wavelengths.  Should include stellar flux,
        QE, and throughput
    design_data : `data.DesignData` instance
        data with the roman pupil, etc
    cam : emccd.EMCCD
        camera
    ref_z : numpy.ndarray
        ndarray of reference zernike positions
    chop_zs : iterable of dict
        a list of dicts of the same form as ref_z.  The elements of the list are
        pertubations about ref_z to use as symmetric disturbances.  Concretely,
        if ref_z = {'Z2': 3} and chop_zs = [{'Z2': 5}], one model will be run
        at (up) Z2=8, and one will be run at (down) Z2=-2.
    cam : emccd.EMCCD
        camera object.  If not None, the model includes the camera; else chops
        are done in aerial intensities
    nframes_avg : int
        number of frames of camera data to average, if the camera is used.  Does
        nothing if cam=None

    Returns
    -------
    up, down, diff : `numpy.ndarray`
        array of up, down, and differential images.
        Diff does not include the "chop size," i.e. is (up-down)/2

    """
    dd = design_data
    ups = []
    downs = []
    cds = []  # central differences
    for c in chop_zs:
        up = props.polychromatic(wvl, weights, dd, zernikes=ref_z+c)
        down = props.polychromatic(wvl, weights, dd, zernikes=ref_z-c)
        if cam is not None:
            up = cam.expose(up, nframes_avg).mean(axis=0).astype(config.precision)
            down = cam.expose(down, nframes_avg).mean(axis=0).astype(config.precision)

        cd = (up - down) / 2
        ups.append(up)
        downs.append(down)
        cds.append(cd)

    return cds, ups, downs


def _avg_frame_respecting_memory_limits(im, cam, frames, framelim):
    """Produce an average image of (im) using (cam) made up of (frames) observations, subject to (framelim) simultaneous draws.
    """
    f = frames
    aggregrate = []
    at_once = framelim
    while f > 0:
        if f < framelim:
            at_once = f
        block = cam.expose(im, at_once).mean(axis=0).astype(im.dtype)
        aggregrate.append(block)
        f -= at_once

    return np.mean(np.asarray(aggregrate), axis=0)


def flt_chop_seq(wvl, design_data, ref_z, chop_zs, gains,
                 ref=(2.25, 'g0v', 20), targ=(5, 'g0v', 150),
                 nframes_chop=10_000, nframes_ref_ref=10_000, nframes_ref_targ=120_000,
                 cam=None,  style='APPROXTARGET', ref_z_target=None, framelim=20_000,
                 mode='hlc', td=None, sd=None):
    """The flight calibration sequence.

    Internally builds several pieces of information it needs, for example
    the spectral weight vectors.

    Performs Zmm buildup using the baseline masks, etc.

    Parameters
    ----------
    wvl : numpy.ndarray
        wavelength vector, microns
    design_data : data.DesignData
        data with the roman pupil, etc
    ref_z : numpy.ndarray
        reference point zernikes
    chop_zs : numpy.ndarray
        chop offsets
    gains : numpy.ndarray
        gain associated with the chopping.  If the chops are "diagonal"
        then gains == np.diag(chop_zs)
    ref : tuple of length 3
        (vmag, color, em_gain), can skip em_gain if cam=None
    targ : tuple of length 3
        (vmag, color, em_gain), can skip em_gain if cam=None
    nframes_chop : int
        number of frames to use for the chopping frames, i.e. at the top or
        bottom of the square wave used to calibrate the estimator
    nframes_ref_ref : int
        number of frames used to stare at the reference star reference frame
    nframes_ref_targ : int
        number of frames used to stare at the target star reference frame
        note that this should be ~= the ratio of the fluxes of the ref and target
        stars for an "efficient" or "balanced" noise contribution from each to
        the standard error of the mean, which is the primary cause of nonzero
        offsets between the two when style!=APPROXTARGET, after consideration
        of drift (ref_z_target != ref_z)
    cam : emccd.EMCCD
        camera, if None calculations do not utilize the camera or EM gain information
    style : str, {'APPROXTARGET', 'TARGET'}
        how to reference the estimator and construct the flux and shear modes
        APPROXTARGET scales the reference frame to the target star, and may have
        zero point errors due to chromaticity and other factors
        TARGET uses a reference frame taken at ref_z_target
    ref_z_target : numpy.ndarray, optional
        reference Zernikes (zero point) on the target star.  If None, ref_z
        is used.  This argument allows the model to drift between reference
        and target stars
    framelim : int
        maximum number of frames for parallel computation at once on a GPU
        this is used to overcome memory limits when performing extremely deep
        averages.  Averages are done of up to framelim frames at once, and those
        averages are averaged to produce the final average.  For a linear process
        there is no difference between this approach and averaging all at once
    mode : str, {'hlc', 'spec', 'wfov'}
        coronagraphic configuration
        mode is a legacy name for this
    td : spectral.ThroughputDatabase
        database of system throughput for various CGI configurations
        note: None default value is for signature compatibility with JPL internal
        copy of lowfssim, which has a path pre-coded for the default which does
        not exist in the public version
    sd : spectral.StellarDatabase
        database of stellar spectra

    Returns
    -------
    dict
        keys of wref, wtarg, R
        wref: numpy.ndarray
            reference star spectral weights
        wtarg : numpy.ndarray
            target star spectral weight
        R : reconstruction.Reconstructor
            LOWFS estimator calibrated for the target star

    Notes
    -----
    does not mutate cam if it exists

    """
    if ref_z_target is None:
        ref_z_target = ref_z

    # input unpacking
    dd = design_data
    if cam is None:
        if len(ref) != 3:
            raise ValueError('when cam is not None, ref/target gains must be provided')

        Vref, Cref, *_ = ref
        Vtarg, Ctarg, *_ = targ
    else:
        Vref, Cref, Gref = ref
        Vtarg, Ctarg, Gtarg = targ

        if hasattr(cam, 'em_gain'):  # support sCMOS
            oldgain = cam.em_gain
            cam.em_gain = Gref
        else:
            Gref, Gtarg = 1, 1  # avoid calcs below going awry based on bogus gain that doesn't exist


    # spectral weight vector computations
    throughput = td(mode, wvl)

    weights = sd(Cref, wvl)
    fudge = sd.sparsity_fudge_factor(Cref, wvl)
    v = 10 ** (-Vref/2.5)
    wref = weights * (fudge*v*throughput)

    weights = sd(Ctarg, wvl)
    fudge = sd.sparsity_fudge_factor(Ctarg, wvl)
    v = 10 ** (-Vtarg/2.5)
    wtarg = weights * (fudge*v*throughput)


    # reference star chopping
    diffs, *_ = chop_bipolar(wvl, wref, dd, ref_z, chop_zs, cam=cam, nframes_avg=nframes_chop)
    for (d, g) in zip(diffs, gains):
        d /= g

    # reference frames and flux mode computations
    # the "S" values have no knowledge error, which is slightly non flight like
    refref = props.polychromatic(wvl, wref, dd, ref_z)
    reftarg = props.polychromatic(wvl, wtarg, dd, ref_z_target)

    _dark = np.zeros_like(refref)
    Sref = float(refref.sum())
    Starg = float(reftarg.sum())
    Sratio = Starg/Sref

    if cam is not None:
        refref = _avg_frame_respecting_memory_limits(refref, cam, nframes_ref_ref, framelim)
        darkref = _avg_frame_respecting_memory_limits(_dark, cam, nframes_chop, framelim)

        if hasattr(cam, 'em_gain'):  # support sCMOS
            cam.em_gain = Gtarg

        reftarg = _avg_frame_respecting_memory_limits(reftarg, cam, nframes_ref_targ, framelim)
        darktarg = _avg_frame_respecting_memory_limits(_dark, cam, nframes_chop, framelim)

        # fm = flux mode; (I-D); I=image from camera, D=dark
        fmref = refref-darkref
        if style.upper() == 'APPROXTARGET':
            fmtarg = fmref * (Starg/Sref) * (Gtarg/Gref)

            # reftarg is scaled reference aerial intensity + target dark
            reftarg = fmtarg + darktarg
        else:
            fmtarg = reftarg - darktarg

        Gratio = Gtarg / Gref
        ratio = Sratio * Gratio
    else:
        if style.upper() == 'APPROXTARGET':
            fmtarg = refref
        else:
            fmtarg = reftarg

        darktarg = _dark
        ratio = Sratio

    for d in diffs:
        d *= ratio

    CHOP_SHEAR_PX = 0.038
    sy = synthesize_pupil_shear(fmtarg, CHOP_SHEAR_PX, 1)
    sx = synthesize_pupil_shear(fmtarg, CHOP_SHEAR_PX, 0)
    sy /= CHOP_SHEAR_PX
    sx /= CHOP_SHEAR_PX

    mask = np.ones_like(sy)
    mask[0, :] = 0
    mask[-1,:] = 0
    mask[:, 0] = 0
    mask[:,-1] = 0
    zmm = prepare_Zmm(diffs, fmtarg, (sx,sy), mask)
    R = RV25(zmm, reftarg, darktarg)

    if cam and hasattr(cam, 'em_gain'):
        cam.em_gain = oldgain

    return {
        'R': R,
        'wref': wref,
        'wtarg': wtarg,
        'zmm': zmm
    }


def plot_modes(chops, sx, sy, clim=10, cmap='RdBu', interpolation='nearest', cbar_location='right'):
    """Plot modes in the usual grid, returning the figure.

    Parameters
    ----------
    chops : Iterable of `numpy.ndarray`
        Zernike chops, assumed to be Z2..Z11, inclusive
    sx : `numpy.ndarray`
        x shear mode
    sy : `numpy.ndarray`
        y shear mode
    ref : `numpy.ndarray`
        reference frame
    clim : `float`
        symmetric colorbar limit
    cmap : `str`
        colormap, passed directly to matplotlib
    interpolation : `str`
        interpolation method, passed directly to matplotlib
    cbar_location : `str`,
        location for the colorbar, passed directly to matplotlib

    Returns
    -------
    `matplotlib.figure.Figure`
        figure instance

    """
    keys_modes = ['Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'Z10', 'Z11', 'Sx', 'Sy']
    fig = plt.figure(figsize=(12, 10))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(3, 4),  # creates 2x2 grid of axes
                     axes_pad=0.0,  # pad between axes in inch.
                     cbar_location=cbar_location,
                     cbar_mode="single",
                     )

    labels = keys_modes
    clim = (-clim, clim)

    for ax, mode, lab in zip(grid, [*chops, sx, sy], labels):
        # Iterating over the grid returns the Axes.
        if lab not in ['Sx', 'Sy']:
            d = mode
        else:
            d = mode / 20

        if hasattr(d, 'get'):
            d = d.get()
        im = ax.imshow(d, cmap=cmap, interpolation=interpolation, clim=clim)
        ax.text(2, 4, lab, fontsize=14)
        ax.grid(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    fig.colorbar(im, cax=grid.cbar_axes[0], label=r'$\Delta$ DN')  # NOQA
    return fig
