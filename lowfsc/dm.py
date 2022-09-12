"""CGI DMs."""

from astropy.io import fits

from prysm.fttools import pad2d

from prysm.experimental.dm import DM

# TODO runtime flexibility of influence function specification (too many coupled)
# parameters)

# InFLuence function FileName
DEFAULT_IFN_FN = 'influence_dm5v2.fits'

dm_act_pitch = 0.9906  # mm
ifn_sampling_factor = 10  # dimensionless
ifn_pitch = dm_act_pitch / ifn_sampling_factor  # mm

dm_model_res = 660
dm_angle_deg = (0, 0, 9.65)
nact = 48
dm_diam_mm = dm_act_pitch * nact
dm_diam_px = dm_diam_mm * ifn_sampling_factor


def setup_dms(dd, data_root, ifn_fn=DEFAULT_IFN_FN):
    """Bind prysm DM models to DesignData.

    Parameters
    ----------
    dd : DesignData
        designdata instance
    data_root : str, path, or path_like
        where the lowfssim data directory is
    ifn_fn : str
        filename of the influence function

    Notes
    -----
    This function or an equivalent must be called before dd.update_dm(n, actuator_map)
    see update_dm's documentation for more information on integrated modeling
    of DMs


    """
    ifn = fits.getdata(data_root/ifn_fn).squeeze()
    ifn = pad2d(ifn, out_shape=dm_model_res)
    mag = ifn_pitch / dd.dx_pup
    dm1 = DM(ifn, Nact=nact, sep=ifn_sampling_factor, rot=dm_angle_deg, upsample=mag)
    dm2 = DM(ifn, Nact=nact, sep=ifn_sampling_factor, rot=dm_angle_deg, upsample=mag)
    dd.dm1 = dm1
    dd.dm2 = dm2
    return
