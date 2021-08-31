"""Automatic routines for performing certain operations."""

from prysm.conf import config

from . import props

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def chop_bipolar(wvl, weights, design_data, ref_z, chop_zs, cam=None, nframes_avg=10_000):
    """Do bipolar chopping (up-down)/2 about a particular point.

    Parameters
    ----------
    wvl : `numpy.ndarray`
        wavelengths of light, um
    weights : `numpy.ndarray`
        weights associated with the wavelengths.  Should include stellar flux,
        QE, and throughput
    design_data : `data.DesignData` instance
        data with the roman pupil, etc
    cam : `emccd.EMCCD`
        camera
    ref_z : `numpy.ndarray`
        array of reference frame Zernike coefficients
    chop_zs : `iterable` of `numpy.ndarray`
        sequence of ndarrays containing chop offsets from the reference point
    cam : `emccd.EMCCD`
        camera object.  If not None, the model includes the camera; else chops
        are done in aerial intensities
    nframes_avg : `int`
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
            up = cam.expose(up, nframes_avg).mean(axis=0, dtype=config.precision)
            down = cam.expose(down, nframes_avg).mean(axis=0, dtype=config.precision)

        cd = (up - down) / 2
        ups.append(up)
        downs.append(down)
        cds.append(cd)

    return cds, ups, downs


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
