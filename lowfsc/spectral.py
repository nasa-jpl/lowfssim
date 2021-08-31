"""Spectral shaping."""

from scipy.interpolate import interp1d

import numpy as np

# most of the contents of this file is heritage or not used.  That reflects
# a change in CGI practice.  Formerly, we used CBE values for coatings, etc.
# that posture has changed: we now lump things into a scalar value, which
# removes the spectral shaping from anything other than the star, detector,
# or FPM.


CGI_PUPIL_AREA = 0.641*2.363*2.363  # 64.1% of the enclosing rectangle (close enough)

# standard LOWFS bandpass, 9 wavelengths of spectral sampling
LOWFS_LOWER = (575-64)/1e3
LOWFS_UPPER = (575+64)/1e3


def mk_bandpass(n):
    """Make a LOWFS bandpass with n wavelengths."""
    if n == 1:
        return np.array([.575])

    return np.linspace(LOWFS_LOWER, LOWFS_UPPER, n)


LOWFS_BANDPASS = mk_bandpass(9)


CGI_BASIC_COATINGS = {
    'HRC': 7,       # = OTA+TCA, per Bijan
    'FSS99': 7+1,   # = CGI front-end + rad shield mirror (LOCAM)
    'Al': 2,        # = DM1+DM2, CGI front-end
    'BBAR': 3,      # LOWFS lenses
}
# additions...
# HLC => +1 FSS99 (SPM)
# SPC => +1 Al (SPM)


class ThroughputDatabase:
    """Database of spectral throughputs."""

    def __init__(self, emccd):  # historical: first arg coatings
        """Create a new database of throughputs.

        Parameters
        ----------
        coatings : `dict`
            keys of strings, which are the names of coatings
            values of scipy.interpolation.interp1d returns
        emccd : `scipy.interpolation.interp1d`
            interp1d of emccd data

        """
        # self.coatings = coatings
        self.emccd_interpf = emccd

    def __call__(self, corongraph_mode, wavelengths):
        """Spectral shape andthroughput for mode & wavelengths."""
        # hrc = self.coatings['HRC']
        # fss99 = self.coatings['FSS99']
        # al = self.coatings['Al']
        # bbar = self.coatings['bbar']

        # n_hrc = CGI_BASIC_COATINGS['HRC']
        # n_fss99 = CGI_BASIC_COATINGS['FSS99']
        # n_al = CGI_BASIC_COATINGS['Al']
        # n_bbar = CGI_BASIC_COATINGS['BBAR']
        # if corongraph_mode == 'HLC':
        #     #  not += 1 to avoid any chance of modifying the dict
        #     n_fss99 = n_fss99 + 1
        # else:
        #     n_al = n_al + 1

        # hrc = hrc(wavelengths) ** n_hrc
        # fss99 = fss99(wavelengths) ** n_fss99
        # al = al(wavelengths) ** n_al
        # bbar = bbar(wavelengths) ** n_bbar
        emccd = self.emccd_interpf(wavelengths)
        # return hrc * fss99 * al * bbar * emccd
        corongraph_mode = corongraph_mode.upper()
        # from LOWFS_photometry 20200801, Throughput sheet.  BOL REQs
        if corongraph_mode == 'HLC':
            return 0.3443 * emccd
        elif corongraph_mode == 'SPEC':
            return 0.3092 * emccd
        elif corongraph_mode == 'WFOV':
            return 0.3092 * emccd
        else:
            raise ValueError('invalid coronagraph mode: must be HLC, SPEC, or WFOV')

    @classmethod
    def bijan_data(cls, data_fldr):
        """Load data from Bijan."""
        emccd_data = np.genfromtxt(data_fldr/'emccd-spectral-qe.csv', delimiter=',', skip_header=1)
        emccd_wvl = emccd_data[:, 0]/1e3
        emccd_dat = emccd_data[:, 1]
        emccd_interpf = interp1d(emccd_wvl, emccd_dat, bounds_error=False, fill_value=0)

        # coating_data = np.genfromtxt(data_fldr/'coatings.csv', delimiter=',', skip_header=1)
        # coating_wvl = coating_data[:, 0]
        # coatings = {}
        # for i, lbl in enumerate(['HRC', 'FSS99', 'Al', 'BBAR']):
        #     data = coating_data[:, i+1]
        #     interpf = interp1d(coating_wvl, data)
        #     coatings[lbl] = interpf

        # return ThroughputDatabase(coatings, emccd_interpf)
        return ThroughputDatabase(emccd_interpf)


class StellarDatabase:
    """Database of stellar spectra."""

    def __init__(self, stellar_table, stellar_labels):
        """Create a new database of stellar throughputs.

        Parameters
        ----------
        stellar_table : `numpy.ndarray`
            columnnar table of stellar spectral fluxes
        stellar_labels : `numpy.ndarray`
            labels associated with the columns, as [g0v, av5], and so forth.
            The index of the label identifies its column

        """
        self.stellar_interpfs = {}
        self.table = stellar_table
        self.stellar_labels = stellar_labels

    def __call__(self, star_type, wavelengths, lower_lim=-1, upper_lim=-1):
        """Spectral weights associated with a given star type at a given wavelength."""
        stellar_interpf = self.stellar_interpfs.get(star_type, None)
        if lower_lim == -1:
            lower_lim = LOWFS_LOWER
        if upper_lim == -1:
            upper_lim = LOWFS_UPPER

        if stellar_interpf is None:
            # this line throws value error if star_type not in list
            i = self.stellar_labels.index(star_type)

            # table has units of W/(m^2 m) == J/(s m^2 m)
            # => divide by J/ph, multiply by density, multiply by area
            # == ph/s.
            # this could be cleaner with astropy units, but c'est la vie
            wvl = self.table[:, 0]*1e6  # 1e6; m => um
            photon_energy = self.table[:, 1]
            dat = self.table[:, i]
            dat = dat / photon_energy * 1e-9 * CGI_PUPIL_AREA  # 1e-9 = sampling density, 1 nm
            stellar_interpf = interp1d(wvl, dat)
            self.stellar_interpfs[star_type] = stellar_interpf

        ret = stellar_interpf(wavelengths)
        ret[wavelengths < lower_lim] = 0
        ret[wavelengths > upper_lim] = 0
        return ret

    def sparsity_fudge_factor(self, star_type, wavelengths, lower_lim=-1, upper_lim=-1):
        """Fudge factor to deal with spectral sparsity for a given set of wavelengths."""
        true = self(star_type, wavelengths, lower_lim=lower_lim, upper_lim=upper_lim).sum()
        wvl_dense = self.table[:, 0] * 1e6
        dense = self(star_type, wvl_dense, lower_lim=lower_lim, upper_lim=upper_lim).sum()
        return dense/true

    @classmethod
    def bijan_data(cls, data_fldr):
        """Load data from Bijan."""
        stellar_data = np.genfromtxt(data_fldr/'stellar-spectra.csv', delimiter=',', skip_header=1)
        labels = [
            'wvl',
            'photon energy',
            'b3v',
            'a0v',
            'a5v',
            'f5v',
            'g0v',
            'f5v',
            'k0v',
            'k5v',
            'm0v',
            'm5v'
        ]

        return StellarDatabase(stellar_data, labels)
