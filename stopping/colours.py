import numpy as np

from matplotlib.cm import get_cmap
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

data_dir = 'colour_calibration'

def generic(half_level, cmap=None):
    cm = get_cmap(cmap)
    return lambda D: cm( D / (half_level + D))

def calibrated(filename, func='tanh'):
    calib = np.load(f'{data_dir}/{filename}.npz')

    dose = calib['dose']
    lD = np.log(dose)

    OD = calib['OD']
    lOD = np.log(OD)

    if func == 'lerp':
        get_R = interp1d(lD, lOD[0], fill_value='extrapolate')
        get_G = interp1d(lD, lOD[1], fill_value='extrapolate')
        get_B = interp1d(lD, lOD[2], fill_value='extrapolate')
    else:
        if func == 'tanh':
            F = lambda ld, a, b, c, d: a + b * np.tanh(c + d * ld)
            p0 = [0, 1, 0, 1]
        params_R, _ = curve_fit(F, lD, lOD[0], p0)
        params_G, _ = curve_fit(F, lD, lOD[1], p0)
        params_B, _ = curve_fit(F, lD, lOD[2], p0)

        get_R = lambda ld: F(ld, *params_R)
        get_G = lambda ld: F(ld, *params_G)
        get_B = lambda ld: F(ld, *params_B)


    def colour_value(dose):
        ldose = np.atleast_2d(np.log(dose))
        OD_r = np.exp( get_R( ldose ) )
        OD_g = np.exp( get_G( ldose ) )
        OD_b = np.exp( get_B( ldose ) )

        return 10**-np.stack((OD_r, OD_g, OD_b), axis=-1)

    return colour_value
