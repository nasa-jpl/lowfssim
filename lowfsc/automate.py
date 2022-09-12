"""Automatic routines for performing certain operations."""
from collections import defaultdict
from pathlib import Path

import numpy as truenp

from prysm.conf import config
from prysm.mathops import np

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from lowfsc.reconstruction import (
    synthesize_pupil_shear,
    prepare_Zmm,
    ReconstructorV2pt5 as RV25,
)
from lowfsc import props
from lowfsc.spectral import ThroughputDatabase, StellarDatabase
from lowfsc import props

from scipy.stats import pearsonr


keys_modes = ['Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'Z10', 'Z11', 'Sx', 'Sy']
keys = ['na', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'Z10', 'Z11', 'F1', 'Sx', 'Sy', 'na', 'N', r'$\sum I$']


DEFAULT_ROOT = (Path(__file__).parent/'data').expanduser().absolute()

default_sd = StellarDatabase.bijan_data(DEFAULT_ROOT)
default_td = ThroughputDatabase.bijan_data(DEFAULT_ROOT)


def chop_bipolar(wvl, weights, design_data, ref_z, chop_zs, cam=None, nframes_avg=10_000):
    """Do bipolar chopping (up-down)/2 about a particular point.

    Parameters
    ----------
    wvl : numpy.ndarray
        wavelengths of light, um
    weights : numpy.ndarray
        weights associated with the wavelengths.  Should include stellar flux,
        QE, and throughput
    design_data : data.DesignData instance
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
    up, down, diff : numpy.ndarray
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
                 mode='hlc', td=default_td, sd=default_sd):
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

    mask = np.ones_like(sy)
    mask[0,  :] = 0  # NOQA
    mask[-1, :] = 0
    mask[:,  0] = 0
    mask[:, -1] = 0
    zmm = prepare_Zmm(diffs, fmtarg, (sx, sy), mask)
    R = RV25(zmm, reftarg, darktarg)

    if cam and hasattr(cam, 'em_gain'):
        cam.em_gain = oldgain

    return {
        'R': R,
        'wref': wref,
        'wtarg': wtarg,
        'zmm': zmm
    }


def plot_modes(chops, sx, sy, clim=80, cmap='RdBu', interpolation='nearest', cbar_location='right'):
    """Plot modes in the usual grid, returning the figure.

    Parameters
    ----------
    chops : Iterable of numpy.ndarray
        Zernike chops, assumed to be Z2..Z11, inclusive
    sx : numpy.ndarray
        x shear mode
    sy : numpy.ndarray
        y shear mode
    ref : numpy.ndarray
        reference frame
    clim : float
        symmetric colorbar limit
    cmap : str
        colormap, passed directly to matplotlib
    interpolation : str
        interpolation method, passed directly to matplotlib
    cbar_location : str,
        location for the colorbar, passed directly to matplotlib

    Returns
    -------
    matplotlib.figure.Figure
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


def evaluate_linear_response_z2(wvl, weights, design_data, prop, eval_range=60, offset=(0, 0), chop_size=5, scan='Z2'):
    """Evaluate (both in an automated fashion, and visually) linear error.

    Parameters
    ----------
    wvl : numpy.ndarray
        wavelengths of light, um
    weights : numpy.ndarray
        weights associated with the wavelengths.  Should include stellar flux,
        QE, and throughput
    design_data : data.DesignData instance
        data with the roman pupil, etc
    prop : module
        propagation module, either lowfsc.props_hlc or lowfsc.props_spc
        e.g.
        >>> from lowfsc import props_hlc
        >>> build_modes_estimator(prop=props_hlc,...)
    eval_range : float
        symmetric range of z2, in mas, to evaluate on a 1 mas pitch
    offset : tuple of float
        offset from (0,0) (z2,z3) for evaluation.  Will be chopped at offset
        and evaluated about that point
    chop_size : float
        size of chop, in nanometers
    scan : str, {'Z2', 'Z3'}
        which term to scan

    Returns
    -------
    dict
        keys:
            plots - tuple, with (linear response plot, deviation from linear plot)
            input_mas - ndarray of input values, mas
            response_mas - response of estimator in mas
            peak response - scalar value, maximum (monotonic) value produced by estimator
            monotonic range - the x coordinate of peak response
            linear range - the x coordinate at which the estimator error crosses 5%

    """
    dd = design_data

    # ----------- build estimator
    ref = prop.polychromatic(wvl, weights, dd, zernikes={'Z2': offset[0], 'Z3': offset[1]})
    chop_cmds = [
            {'Z2': chop_size+offset[0], 'Z3': offset[1]},
            {'Z3': chop_size+offset[1], 'Z2': offset[0]},
            {'Z4': chop_size, 'Z2': offset[0], 'Z3': offset[1]},
            {'Z5': chop_size, 'Z2': offset[0], 'Z3': offset[1]},
            {'Z6': chop_size, 'Z2': offset[0], 'Z3': offset[1]},
            {'Z7': chop_size, 'Z2': offset[0], 'Z3': offset[1]},
            {'Z8': chop_size, 'Z2': offset[0], 'Z3': offset[1]},
            {'Z9': chop_size, 'Z2': offset[0], 'Z3': offset[1]},
            {'Z10': chop_size, 'Z2': offset[0], 'Z3': offset[1]},
            {'Z11': chop_size, 'Z2': offset[0], 'Z3': offset[1]},
    ]

    chops = prop.chop(wvl, weights, dd, ref, chop_cmds)
    chops = [c / chop_size for c in chops]

    mask = np.ones_like(ref)
    mask[0, :] = 0
    mask[-1, :] = 0
    mask[:, 0] = 0
    mask[:, -1] = 0

    sy, sx = [synthesize_pupil_shear(ref, 0.038, a) for a in [0, 1]]

    zmm = prepare_Zmm(chops, ref, (sx, sy), mask)
    R25 = RV25(zmm, ref)

    # ----------- build estimator
    if scan == 'Z2':
        idx = 1
    else:
        idx = 2

    pts = truenp.arange(-eval_range, eval_range+1) + offset[idx-1]
    ests = []
    for z in pts:
        i = prop.polychromatic(wvl, weights, dd, zernikes={scan: z*2.87})
        e25 = R25.estimate(i)
        ests.append(float(e25[idx]))

    a = np.array(ests)/2.87

    if hasattr(a, 'get'):
        a = a.get()

    if hasattr(pts, 'pts'):
        pts = pts.get()

    i = pts
    ii = i - (offset[idx-1]/2.87)
    r = a
    lower = ii > -3  # 3 nm ~= 1 mas
    upper = ii < 3

    lower = truenp.argmax(lower)
    upper = truenp.argmin(upper)
    ii = i[lower:upper]
    rr = r[lower:upper]
    fitcoef = truenp.polyfit(ii, rr, deg=1)
    proj = truenp.polyval(fitcoef, i)

    linearity_error = a/proj*100-100
    # mask = a == 0
    # linearity_error[mask] = 0

    fig, ax = plt.subplots()
    ax.plot(pts, a, zorder=3)
    ax.plot(pts, pts, ls='--', c='k', alpha=0.75)
    ax.set(ylim=(-35, 35), xlabel=f'input {scan}, mas ots', ylabel=f'output {scan}, mas ots')

    o2 = list(offset)
    o2[0] = o2[0] / 2.87
    o2[1] = o2[1] / 2.87
    o2[0] = f'{o2[0]:.2f}'
    o2[1] = f'{o2[1]:.2f}'
    o2 = tuple(o2)

    o = offset[idx-1] / 2.87
    fig2, ax2 = plt.subplots()
    ax2.plot(i-o, linearity_error, zorder=3)
    ax2.axhline(0, ls=':', c='k', alpha=0.5)
    ax2.axhline(-10, ls='dashdot', c='r', alpha=0.25)
    ax2.axhline(+10, ls='dashdot', c='r', alpha=0.25)
    ax2.axvline(-5, ls='dashdot', c='r', alpha=0.25)
    ax2.axvline(+5, ls='dashdot', c='r', alpha=0.25)
    ax2.set(xlabel=f'Disturbance, mas OTS about {o2}', xlim=(-20, 20), ylim=(-20, 20), ylabel='Linearity error, %')

    dy = truenp.diff(a)
    precursor = abs(a[1:]) < 50
    mask = dy >= 0
    mask &= precursor
    left_monoi = truenp.argmax(mask)
    right_monoi = len(mask)-truenp.argmax(mask[::-1])
    left = a[left_monoi]
    right = a[right_monoi]
    if -left < right:
        mono_range = pts[left_monoi]
        peak_response = -a[left_monoi]
    else:
        mono_range = pts[right_monoi]
        peak_response = a[right_monoi]

    abs_le = abs(linearity_error)
    mask1 = abs_le <= 10
    mask2 = abs(a) < 55
    i_linear_range = truenp.argmax(mask1 & mask2)
    linear_range = pts[i_linear_range]

    return {
        'plots': (fig, fig2),
        'input_mas': pts,
        'response_mas': a,
        'peak response': peak_response,
        'monotonic range': mono_range,
        'linear range': linear_range,
        'fit': proj,
    }


def evaluate_estimator_noise(wvl, weights, cam, design_data, R25, ymax=10):
    """Evaluate (both in an automated fashion, and visually) linear error.

    Parameters
    ----------
    wvl : numpy.ndarray
        wavelengths of light, um
    weights : numpy.ndarray
        weights associated with the wavelengths.  Should include stellar flux,
        QE, and throughput
    cam : emccd.EMCCD
        camera instance
    design_data : data.DesignData instance
        data with the roman pupil, etc
    R25 : Reconstructor
        reconstructor instance, but notionally V2.5
    ymax : float
        maximum value of y to plot

    Returns
    -------
    dict
        keys:
            plots - tuple, with (linear response plot, deviation from linear plot)
            input_mas - ndarray of input values, mas
            response_mas - response of estimator in mas
            peak response - scalar value, maximum (monotonic) value produced by estimator
            monotonic range - the x coordinate of peak response
            linear range - the x coordinate at which the estimator error crosses 5%

    """
    dd = design_data
    P = R25.P
    PT = P.T
    ref_aerial = props.polychromatic(wvl, weights, dd)
    var = np.diag(
                  cam.expose(ref_aerial, 10_000)
                  .var(axis=0)
                  .astype(config.precision)
                  .ravel()
                 )
    covmat = P @ var @ PT
    sigma = np.sqrt(np.diag(covmat))

    if hasattr(sigma, 'get'):
        sigma = sigma.get()

    x = truenp.arange(17)+1
    fig, ax = plt.subplots()
    ax.bar(x, sigma, zorder=3)
    ax.axhline(7.5, zorder=1, c='r', alpha=0.5, ls='--')
    plt.xticks(x, keys)
    ax.set(xlabel='estimator field', xlim=(0.5, 12.5),
           ylabel=r'noise $\sigma$, nm (Z) px (S), unitless (F)', ylim=(0, ymax))
    return fig, sigma


def evaluate_estimator_crosstalk(wvl, weights, design_data, R25, prop, n_pts=9, eval_range_z2z3=3, eval_range_z4up=0.3):
    """Produce plots with which to evaluate the estimator crosstalk.

    Parameters
    ----------
    wvl : numpy.ndarray
        wavelengths of light, um
    weights : numpy.ndarray
        weights associated with the wavelengths.  Should include stellar flux,
        QE, and throughput
    design_data : data.DesignData instance
        data with the roman pupil, etc
    R25 : Reconstructor
        reconstructor instance, but notionally V2.5
    prop : module
        propagation module, either lowfsc.props_hlc or lowfsc.props_spc
        e.g.
        >>> from lowfsc import props_hlc
        >>> build_modes_estimator(prop=props_hlc,...)
    n_pts : int
        number of points over which to evaluate the crosstalk
    eval_range_z2z3 : float
        symmetric range of values over which to evaluate the crosstalk due to z2/z3
    eval_range_z4up : float
        symmetric range of values over which to evaluate the crosstalk due to z4..z11

    Returns
    -------
    tuple of matplotlib.figure.Figure
        (Z4+ plot, Z2/Z2 plot)

    """
    dd = design_data
    center = int(np.ceil(n_pts/2))-1  # -1 => 1 to 0 indexing
    z2z3pts = truenp.linspace(-eval_range_z2z3, eval_range_z2z3, n_pts)
    z4uppts = truenp.linspace(-eval_range_z4up, eval_range_z4up, n_pts)

    def factory():
        return z4uppts
    ranges = defaultdict(factory)
    ranges['Z2'] = z2z3pts
    ranges['Z3'] = z2z3pts

    estimates = {}
    for key in keys_modes[:-2]:  # :-2 = exclude shears
        pts = ranges[key]
        ests = []
        for pt in pts:
            i = prop.polychromatic(wvl, weights, dd, zernikes={key: pt})
            e = R25.estimate(i)
            ests.append(e)

        ests = np.array(ests)
        if hasattr(ests, 'get'):
            ests = ests.get()

        estimates[key] = ests

    fig, axs = plt.subplots(2, 4, figsize=(16, 8), sharex=False, sharey=False, gridspec_kw={'hspace': 0, 'wspace': 0})
    axs = axs.ravel()
    cntr = 0
    for k in keys_modes[2:-2]:  # 2:-2 = Z4..Z11
        i = keys.index(k)
        data = estimates[k]
        x = ranges[k]
        y = data.copy()
        y[:, i] = 0
        y[:, 15] = 0  # always 1
        y[:, 16] = 0  # image sum

        ylabel = 'Crosstalk, %'
        xlabel = 'Disturbance, pm RMS'
        xlim = (-300, 300)
        ylim = (-5, 5)

        xx = x.copy()
        xx[center] = 1
        line = (y/xx[:, truenp.newaxis]) * 100
        line[center, :] = 0
        ax = axs[cntr]
        line = line[:, 1:1+len(keys_modes)]
        ax.plot(x*1e3, line, zorder=3)  # 1e3 nm => pm
        ax.axhline(0, ls=':', c='k', alpha=0.5)
        ax.axhline(-1, ls='dashdot', c='r', alpha=0.25)
        ax.axhline(+1, ls='dashdot', c='r', alpha=0.25)
        ax.axvline(-250, ls='dashdot', c='r', alpha=0.25)
        ax.axvline(+250, ls='dashdot', c='r', alpha=0.25)
        ax.set(xlim=xlim, ylim=ylim)
        ax.text(xlim[0]+0.075*(xlim[1]-xlim[0]), ylim[1]-0.3*ylim[1], k, fontsize=14)
        if cntr != 0 and cntr != 4:
            ax.set_yticklabels([])
        else:
            ax.set(ylabel=ylabel)

        ax.set(xlabel=xlabel)
        cntr += 1

    axs[3].legend(keys_modes, title='modes', ncol=3)

    fig2, axs2 = plt.subplots(2, 1, figsize=(4, 8), sharex=False, sharey=False, gridspec_kw={'hspace': 0, 'wspace': 0})
    axs2 = axs2.ravel()
    cntr = 0
    for k in ['Z2', 'Z3']:  # 2:-2 = Z4..Z11
        i = keys.index(k)
        data = estimates[k]
        x = ranges[k]
        y = data.copy()
        y[:, i] = 0
        y[:, 15] = 0  # always 1
        y[:, 16] = 0  # image sum

        ylabel = 'Crosstalk, %'
        xlabel = 'Disturbance, nm RMS'
        xlim = (-3, 3)
        ylim = (-5, 5)

        xx = x.copy()
        xx[center] = 1
        line = (y/xx[:, np.newaxis])*100
        line = line[:, 1:1+len(keys_modes)]
        line[center, :] = 0
        ax = axs2[cntr]
        ax.plot(x, line, zorder=3)
        ax.axhline(0, ls=':', c='k', alpha=0.5)
        ax.axhline(-1, ls='dashdot', c='r', alpha=0.25)
        ax.axhline(+1, ls='dashdot', c='r', alpha=0.25)
        ax.axvline(-1.63, ls='dashdot', c='r', alpha=0.25)
        ax.axvline(1.63, ls='dashdot', c='r', alpha=0.25)
        ax.set(xlim=xlim, ylim=ylim)
        ax.text(xlim[0]+0.075*(xlim[1]-xlim[0]), ylim[1]-0.3*ylim[1], k, fontsize=14)
        ax.set(ylabel=ylabel)

        ax.set(xlabel=xlabel)
        cntr += 1

    axs2[0].legend(keys_modes, title='modes', ncol=3)

    return fig, fig2


def evaluate_modal_correlation(Zmm):
    """Evaluate the correlation of the estimator modes.

    Parameters
    ----------
    Zmm : numpy.ndarray
        Array of (pixels x modes)

    Returns
    -------
    numpy.ndarray, matplotlib.figure.Figure
        correlation matrix (modes x modes) and plot

    """
    if hasattr(Zmm, 'get'):
        Zmm = Zmm.get()

    corrs = truenp.empty((17, 17))
    for i in range(17):
        for j in range(17):
            mode_i = Zmm[:, i]
            mode_j = Zmm[:, j]
            r, p = pearsonr(mode_i, mode_j)
            corrs[i, j] = r

    x = truenp.arange(17)+1
    fig, ax = plt.subplots(figsize=(5, 5))
    ii = ax.imshow(corrs, cmap='RdBu', clim=(-1, 1), zorder=3, alpha=0.75)
    plt.xticks(x - 1, keys)
    plt.yticks(x - 1, keys)
    fig.colorbar(ii, label='Pearson''s Correlation', fraction=0.046)

    return corrs, fig


def optimize_em_gain(wvl, weights, cam, dd, prop, target, percentile=98, iters=10, debug=False):
    """Optimize the EM gain of the camera.

    Parameters
    ----------
    wvl : numpy.ndarray
        wavelengths of light, um
    weights : numpy.ndarray
        weights associated with the wavelengths.  Should include stellar flux,
        QE, and throughput
    cam : emccd.EMCCD
        camera instance
    dd : data.DesignData instance
        data with the roman pupil, etc
    prop : module
        propagation module, either lowfsc.props_hlc or lowfsc.props_spc
        e.g.
        >>> from lowfsc import props_hlc
        >>> build_modes_estimator(prop=props_hlc,...)
    target : float
        target value for percentile p{percentile}
    percentile : float
        percentile to evaluate.  p100=max
        More robust than simple max for scenes that are not PSF-like
    iters : int
        number of iterations of error reduction to perform
    debug : bool, optional
        if true, print iteration #, p{percentile}, error, change (multiplier), and gain at start of loop

    Returns
    -------
    float
        final EM gain

    """
    for n in range(iters):
        i = prop.polychromatic(wvl, weights, dd)
        i = cam.expose(i)
        px = np.percentile(i, percentile)
        change = target / px
        err = px - target

        # float converts scalars to CPU data, in case on GPU
        px = float(px)
        err = float(err)
        change = float(change)
        if debug:
            print(f'iter {n}, p{percentile}={px:.1f}, err={err:.1f}, change={change:.3f}, gain={cam.em_gain:.1f}')
        cam.em_gain *= change

    return cam.em_gain
