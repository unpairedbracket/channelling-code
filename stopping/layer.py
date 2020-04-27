import numpy as np

from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter

data_dir = 'srim_data/'
suffix = '.npz'

class Layer:
    name = ''

    interp_S = None
    interp_E = None
    interp_R = None

    thickness = -1 # in micron
    density = -1 # in g/cc

    colour = None

    def __init__(self, name, thickness, density, colour=None):
        self.name = name
        data = np.load(data_dir + name + suffix)

        self.thickness = thickness
        self.density = density

        # E in KeV
        self.logE = np.log(data['energy'])
        # R in micron
        self.logR = np.log(data['prange'])
        # S in keV/micron
        self.logS = np.log(data['stop_e'] + data['stop_N'])

        # The method I use here is implicitly
        # very sensitive to the gradient dlog(R)/dlog(E)
        # so smooth out weirdness in that.
        self.logR = smooth_gradient(self.logR, self.logE)

        self.interp_R = interp1d(self.logE, self.logR, fill_value='extrapolate')
        self.interp_E = interp1d(self.logR, self.logE, fill_value='extrapolate')

        self.interp_S = interp1d(self.logE, self.logS, fill_value='extrapolate')

        self.colour = colour
        
    @property
    def active(self):
        return self.colour is not None

    def pass_energies(self, energies, multipliers):
        ranges = np.exp(self.interp_R(np.log(energies)))
        new_ranges = np.maximum(0, ranges - multipliers * self.thickness)
        return np.exp(self.interp_E(np.log(new_ranges)))

    def forwardpass(self, energies, multipliers=None):
        if multipliers is None:
            multipliers = 1

        return self.pass_energies(energies, multipliers)

    def backpass(self, energies, multipliers=None):
        if multipliers is None:
            multipliers = -1

        return self.pass_energies(energies, multipliers)

    def dose_from_energies(self, energies_keV, nC_per_cm2=1):
        '''
        Convert energy per proton to dose
        Input energies in keV, proton charge fluence in nC/cm**2
        Output in Grays
        '''
        return energies_keV * nC_per_cm2 / ( self.density * self.thickness ) * 10

    def __repr__(self):
        return str(self.thickness) + ' microns of ' + self.name

def smooth_gradient(A, x, sigma=1):
    dA = np.diff(A)
    dx = np.diff(x)
    deriv_smoothed_dx = gaussian_filter(dA / dx, sigma) * dx
    return np.cumsum(np.insert(deriv_smoothed_dx, 0, A[0]))

