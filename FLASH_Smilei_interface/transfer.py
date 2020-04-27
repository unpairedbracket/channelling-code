import happi
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve2d

theta = -20
theta_rad = theta * np.pi / 180

def BF(A, passes):
    for i in np.arange(passes):
        A = np.convolve(np.concatenate((A[[0]], A, A[[-1]])), [1/4, 1/2, 1/4], 'valid')
    return A

def BF2(A, passes):
    kernel = np.array([ [1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1] ]) / 16
    for i in np.arange(passes):
        A = np.concatenate((A[[0],:], A, A[[-1],:]), axis=0)
        A = np.concatenate((A[:,[0]], A, A[:,[-1]]), axis=1)
        A = convolve2d(A, kernel, 'valid')
    return A



ts = 43200
c = 2.9979e8
k0 = 2*np.pi / 1e-6
S = happi.Open('.', k0 * c)
B_smilei = 107.1E6 # Gauss
B_flash = np.sqrt(4*np.pi)

x_smilei = np.linspace(0, 1, 1025) * S.namelist.Lsim[0] / k0
y_smilei = np.linspace(-1, 1, 161) * S.namelist.Lsim[1] / k0

X, Y = np.meshgrid( (x_smilei[:-1] + x_smilei[1:])/2, (y_smilei[:-1] + y_smilei[1:])/2)
subs = {'axis3':80*2*np.pi}
avg={'axis2':[120*2*np.pi,150*2*np.pi]}
nele = S.Probe(4, '-(Rho_ele_mode_0+Rho_ele_mode_1+Rho_ele_mode_2)', subset=subs, units=['um', 'fs'])._getDataAtTime(ts)
nele_bg = S.Probe(4, '-(Rho_ele_mode_0+Rho_ele_mode_1+Rho_ele_mode_2)', subset=subs, average=avg, units=['um', 'fs'])._getDataAtTime(ts)
Bz = S.Probe(4, 'Bz', subset=subs, units=['um', 'fs'])._getDataAtTime(ts)

while nele[nele <= 0].size > 0:
    nele[nele <= 0] = BF2(nele, 1)[nele <= 0]

nele_bg = BF(nele_bg, 10)

lnele_mult = np.log(nele.T / nele_bg) * np.exp(- (np.fmax(np.abs(Y) - 40e-6, 0)/20e-6)**2)

Bz_filt = (BF2(Bz.T, 1) * np.exp(-(np.fmax(150e-6 - X, 0)/10e-6)**2 -(np.fmax(X - 700e-6, 0)/20e-6)**2 - (np.fmax(np.abs(Y) - 40e-6, 0)/20e-6)**2))

B_thresh = 0.1 * np.abs(Bz_filt).max(axis=0, keepdims=True)
Bz_filt *= 1 - np.exp(-np.abs(Bz_filt / B_thresh)**6) # Screen out noise values

f_Bz = RegularGridInterpolator((X[0,:], Y[:,0]), Bz_filt.T * B_smilei / B_flash, method='linear', bounds_error=False, fill_value = 0)
f_log_nele = RegularGridInterpolator((X[0,:], Y[:,0]), lnele_mult.T, method='linear', bounds_error=False, fill_value = 0)

with h5py.File('lasslab_hdf5_chk_0011') as f:

    nblock = f['bflags'].shape[0]
    dens = f['dens']
    cham = f['cham']
    targ = f['targ']
    magz = f['magz']
    magp = f['magp']
    ener = f['ener']
    eint = f['eint']
    eele = f['eele']
    bbs = f['bounding box']

    for i in np.arange(nblock):
        bb = bbs[i]
        xf = np.linspace(bb[0,0], bb[0,1], 16+1)
        yf = np.linspace(bb[1,0], bb[1,1], 16+1)
        #zf = np.linspace(bb[2,0], bb[2,1], 16+1)
        x0 = (xf[:-1] + xf[1:]).reshape((1,16))/2
        y0 = (yf[:-1] + yf[1:]).reshape((16,1))/2
        x = x0 * np.cos(theta_rad) + y0 * np.sin(theta_rad)
        y = y0 * np.cos(theta_rad) - x0 * np.sin(theta_rad)
        #z = (zf[:-1] + zf[1:]).reshape((16,1,1))/2
        dens_old = dens[i,0,:,:]
        magp_old = magp[i,0,:,:]
        emag = magp_old / dens_old
        dmagp = -magz[i,0,:,:]**2 / 2.0
        magz[i,0,:,:] += f_Bz((x/100, y/100))
        dmagp += magz[i,0,:,:]**2 / 2.0
        magp[i,0,:,:] += dmagp

        dens_t_new = targ[i,0,:,:] * np.exp( f_log_nele((x/100, y/100)) ) * dens_old
        dens_c_new = np.fmax( cham[i,0,:,:] * dens_old, 1e-6 - dens_t_new )
        dens_new  = (dens_c_new + dens_t_new )
        cham_new = dens_c_new / dens_new
        targ_new = dens_t_new / dens_new
        dens[i,0,:,:] = dens_new
        targ[i,0,:,:] = targ_new
        cham[i,0,:,:] = cham_new
        demag = (magp_old + dmagp) / dens_new - emag
        ener[i,0,:,:] += demag
        eint[i,0,:,:] += demag
        eele[i,0,:,:] += demag

    dt = f['real scalars'][1]
    dt[1] = 1e-16
    f['real scalars'][1] = dt

