import yt

import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator as RGI

angle = 10
angle_rad = angle * np.pi/180.0

At = 6.5
Zt = 3.5
Ac = 4
Zc = 2

const = (At/Zt)/(Ac/Zc)

F = yt.load('lasslab_hdf5_chk_0011_old')
F.periodicity = (True, True, True)
scg = F.smoothed_covering_grid(F.max_level, left_edge = [0,0,0], dims=[600, 600, 1])

x = scg['x'][:,0,0].to('um')
y = scg['y'][0,:,0].to('um')

x = x-x[0]
y = y-y[0]

print('X: ', x[[0,-1]])
print('Y: ', y[[0,-1]])

nele = (scg['nele'][:,:,0] / yt.YTQuantity(1e21, 'cm**-3')).to('1')
targ = scg['targ'][:,:,0]

tele = scg['tele'][:,:,0].to_equivalent('keV', 'thermal')
tion = scg['tion'][:,:,0].to_equivalent('keV', 'thermal')

nele_targ = nele *  targ / (targ + const * (1-targ))

L = np.linspace(1, 1024, 1024)

print('x: ', L[[0,-1]] * np.cos(angle_rad))
print('y: ', L[[0,-1]] * np.sin(angle_rad))

Ine = RGI((x, y), nele_targ)
ITe = RGI((x, y), tele)
ITi = RGI((x, y), tion)

nele_L = Ine((L * np.cos(angle_rad), L * np.sin(angle_rad)))
tele_L = ITe((L * np.cos(angle_rad), L * np.sin(angle_rad)))
tion_L = ITi((L * np.cos(angle_rad), L * np.sin(angle_rad)))

np.savez('flash_smilei_{}deg'.format(angle), x=L, nele=nele_L, tele=tele_L, tion=tion_L)
