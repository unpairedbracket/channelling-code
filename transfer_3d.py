import happi
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve, convolve2d

from tqdm import tqdm

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

def BF3(A, passes):
    kernel = np.array([ [ [1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1] ],
                        [ [2, 4, 2],
                          [4, 8, 4],
                          [2, 4, 2] ],
                        [ [1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1] ] ]) / 64
    for i in np.arange(passes):
        A = np.concatenate((A[[0],:,:], A, A[[-1],:,:]), axis=0)
        A = np.concatenate((A[:,[0],:], A, A[:,[-1],:]), axis=1)
        A = np.concatenate((A[:,:,[0]], A, A[:,:,[-1]]), axis=2)
        A = convolve(A, kernel, 'valid')
    return A



ts = 43200
c = 2.9979e8
k0 = 2*np.pi / 1e-6
S = happi.Open('.', k0 * c)
B_smilei = 107.1E6 # Gauss
B_flash = np.sqrt(4*np.pi)

x_smilei = np.linspace(0, 1, 1025) * S.namelist.Lsim[0] / k0
y_smilei = np.linspace(-1, 1, 161) * S.namelist.Lsim[1] / k0
z_smilei = np.linspace(-1, 1, 161) * S.namelist.Lsim[1] / k0
print('Getting Data')
X, Y, Z = np.meshgrid( (x_smilei[:-1] + x_smilei[1:])/2, (y_smilei[:-1] + y_smilei[1:])/2, (z_smilei[:-1] + z_smilei[1:])/2,indexing='ij')
subs = {'axis3':80*2*np.pi}
avg={'axis2':[120*2*np.pi,150*2*np.pi]}
nele = S.Probe(4, '-(Rho_ele_mode_0+Rho_ele_mode_1+Rho_ele_mode_2)', units=['um', 'fs'])._getDataAtTime(ts)
nele_bg = S.Probe(4, '-(Rho_ele_mode_0+Rho_ele_mode_1+Rho_ele_mode_2)', subset=subs, average=avg, units=['um', 'fs'])._getDataAtTime(ts)
Bx = S.Probe(4, 'Bx', units=['um', 'fs'])._getDataAtTime(ts)
By = S.Probe(4, 'By', units=['um', 'fs'])._getDataAtTime(ts)
Bz = S.Probe(4, 'Bz', units=['um', 'fs'])._getDataAtTime(ts)
print('Filering ne')
for i in tqdm(np.arange(10)):
    if nele[nele <= 0].size == 0:
        break
    nele[nele <= 0] = BF3(nele, 1)[nele <= 0]
nele[nele<= 1e-10] = 1e-10

nele_bg = BF(nele_bg, 10).reshape((nele_bg.size, 1, 1))
nele[nele<= 1e-10] = 1e-10

R = np.sqrt(Y**2 + Z**2)

lnele_mult = np.log(nele / nele_bg) * np.exp(- (np.fmax(R - 40e-6, 0)/20e-6)**2)

print('Filering B')
Bx_filt = BF3(Bx * np.exp(-(np.fmax(200e-6 - X, 0)/10e-6)**2 -(np.fmax(X - 700e-6, 0)/20e-6)**2 - (np.fmax(R - 40e-6, 0)/20e-6)**2), 10)
By_filt = BF3(By * np.exp(-(np.fmax(200e-6 - X, 0)/10e-6)**2 -(np.fmax(X - 700e-6, 0)/20e-6)**2 - (np.fmax(R - 40e-6, 0)/20e-6)**2), 10)
Bz_filt = BF3(Bz * np.exp(-(np.fmax(200e-6 - X, 0)/10e-6)**2 -(np.fmax(X - 700e-6, 0)/20e-6)**2 - (np.fmax(R - 40e-6, 0)/20e-6)**2), 10)

B_thresh = 0.1 * np.sqrt(Bx_filt**2+By_filt**2+Bz_filt**2).max(axis=1, keepdims=True).max(axis=2, keepdims=True) + 1e-5
Bx_filt *= 1 - np.exp(-np.abs(Bx_filt / B_thresh)**6) # Screen out noise values
By_filt *= 1 - np.exp(-np.abs(By_filt / B_thresh)**6) # Screen out noise values
Bz_filt *= 1 - np.exp(-np.abs(Bz_filt / B_thresh)**6) # Screen out noise values

print('Making interpolators')
f_Bx = RegularGridInterpolator((Z[0,0,:], Y[0,:,0], X[:,0,0]), np.transpose(Bx_filt,(2,1,0)) * B_smilei / B_flash, method='linear', bounds_error=False, fill_value = 0)
f_By = RegularGridInterpolator((Z[0,0,:], Y[0,:,0], X[:,0,0]), np.transpose(By_filt,(2,1,0)) * B_smilei / B_flash, method='linear', bounds_error=False, fill_value = 0)
f_Bz = RegularGridInterpolator((Z[0,0,:], Y[0,:,0], X[:,0,0]), np.transpose(Bz_filt,(2,1,0)) * B_smilei / B_flash, method='linear', bounds_error=False, fill_value = 0)
f_log_nele = RegularGridInterpolator((Z[0,0,:], Y[0,:,0], X[:,0,0]), np.transpose(lnele_mult,(2,1,0)), method='linear', bounds_error=False, fill_value = 0)

with h5py.File('lasslab_hdf5_chk_0011') as f:

    nblock = f['bflags'].shape[0]
    dens = f['dens']
    cham = f['cham']
    targ = f['targ']
    magx = f['magx']
    magy = f['magy']
    magz = f['magz']
    Bx = f['fcx1']
    By = f['fcy1']
    Bz = f['fcz1']
    Bxi = f['fcx1']
    Byi = f['fcy1']
    Bzi = f['fcz1']
    magp = f['magp']
    ener = f['ener']
    eint = f['eint']
    eele = f['eele']
    bbs = f['bounding box']

    for i in tqdm(np.arange(nblock)):
        bb = bbs[i]
        xf = np.linspace(bb[0,0], bb[0,1], 16+1)
        yf = np.linspace(bb[1,0], bb[1,1], 16+1)
        zf = np.linspace(bb[2,0], bb[2,1], 16+1)
        x0 = (xf[:-1] + xf[1:])/2
        y0 = (yf[:-1] + yf[1:])/2
        z0 = (zf[:-1] + zf[1:])/2
        Z0, Y0, X0 = np.meshgrid(z0, y0, x0, indexing='ij')
        Xr = X0 * np.cos(theta_rad) + Y0 * np.sin(theta_rad)
        Yr = Y0 * np.cos(theta_rad) - X0 * np.sin(theta_rad)
        Zr = Z0
        Zx, Yx, Xx = np.meshgrid(z0, y0, xf, indexing='ij')
        Zy, Yy, Xy = np.meshgrid(z0, yf, x0, indexing='ij')
        Zz, Yz, Xz = np.meshgrid(zf, y0, x0, indexing='ij')
        Xxr = Xx * np.cos(theta_rad) + Yx * np.sin(theta_rad)
        Yxr = Yx * np.cos(theta_rad) - Xx * np.sin(theta_rad)
        Zxr = Zx
        Xyr = Xy * np.cos(theta_rad) + Yy * np.sin(theta_rad)
        Yyr = Yy * np.cos(theta_rad) - Xy * np.sin(theta_rad)
        Zyr = Zy
        Xzr = Xz * np.cos(theta_rad) + Yz * np.sin(theta_rad)
        Yzr = Yz * np.cos(theta_rad) - Xz * np.sin(theta_rad)
        Zzr = Zz
        dens_old = dens[i,:,:,:]
        dmagp = -(magx[i,:,:,:]**2+magy[i,:,:,:]**2+magz[i,:,:,:]**2) / 2.0
        magx[i,:,:,:] += f_Bx((Zr/100, Yr/100, Xr/100))
        magy[i,:,:,:] += f_By((Zr/100, Yr/100, Xr/100))
        magz[i,:,:,:] += f_Bz((Zr/100, Yr/100, Xr/100))
        Bx[i,:,:,:] += f_Bx((Zxr/100, Yxr/100, Xxr/100))
        By[i,:,:,:] += f_By((Zyr/100, Yyr/100, Xyr/100))
        Bz[i,:,:,:] += f_Bz((Zzr/100, Yzr/100, Xzr/100))
        Bxi[i,:,:,:] += f_Bx((Zxr/100, Yxr/100, Xxr/100))
        Byi[i,:,:,:] += f_By((Zyr/100, Yyr/100, Xyr/100))
        Bzi[i,:,:,:] += f_Bz((Zzr/100, Yzr/100, Xzr/100))
        dmagp += (magx[i,:,:,:]**2+magy[i,:,:,:]**2+magz[i,:,:,:]**2) / 2.0
        magp[i,:,:,:] += dmagp

        dens_t_new = targ[i,:,:,:] * np.exp( f_log_nele((Zr/100, Yr/100, Xr/100)) ) * dens_old
        dens_c_new = np.fmax( cham[i,:,:,:] * dens_old, 1e-6 - dens_t_new )
        dens_new  = (dens_c_new + dens_t_new )
        cham_new = dens_c_new / dens_new
        targ_new = dens_t_new / dens_new
        dens[i,:,:,:] = dens_new
        targ[i,:,:,:] = targ_new
        cham[i,:,:,:] = cham_new

    dt = f['real scalars'][1]
    dt[1] = 1e-16
    f['real scalars'][1] = dt

