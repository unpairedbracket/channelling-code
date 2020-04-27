import numpy as np

import matplotlib.pyplot as plt

import yt
from yt import YTQuantity as Q

from scipy.ndimage.filters import gaussian_filter

q = yt.physical_constants.charge_proton
M = yt.physical_constants.mass_hydrogen
m = yt.physical_constants.mass_electron
c = yt.physical_constants.speed_of_light

cm = Q(1, 'cm')
gauss = Q(1, 'gauss')

class FieldsYT():
    '''
    Implements a radiography routine that rotates around the z axis.
    theta = 0 looks along the x axis; theta = 90 along the y-axis
    '''
    
    def __init__(self, f, resolution_um, smooth_um=0, scale=1, dtheta = 5, cachefile='cache.npz'):
        if type(f) is str:
            self.TS = yt.load(f)            
        else:
            self.TS = f
            
        self.i = 0
        try:
            self.F = self.TS[self.i]
        except:
            self.F = self.TS
            self.TS = None
            self.times = np.array([F.current_time.to('ps')])
        else:
            self.times = np.array([F.current_time.to('ps') for F in self.TS])

        self.dtheta = dtheta
        self.n_angles = int(360 / self.dtheta)
        if 360 % self.dtheta != 0:
            print('Warning: dtheta doesn\'t divide a full rotation')

        self.C = self.F.domain_center
        
        Ws = np.sqrt( self.F.domain_width[0]**2 + self.F.domain_width[2]**2 )
        Wt = self.F.domain_width[1]
        self.W = [Ws, Wt, Ws]
        
        resolution = Q(resolution_um, 'um')
        px_s = int(Ws / resolution) + 1
        px_t = int(Wt / resolution) + 1
        self.px = [px_s, px_t]
        
        self.s0 = np.linspace( -Ws, Ws, px_s ) / 2
        self.t0 = np.linspace( -Wt, Wt, px_t ) / 2
        
        self.theta = 0
        self.strength_scale = scale
        self.smoothing = smooth_um / resolution_um
        
        self.cachefile = cachefile
        try:
            self._cache = np.load(self.cachefile, allow_pickle=True)['cache'][()]
        except FileNotFoundError:
            self._cache = {}
        
    def get_axes(self):
        return self.s0.to('um').v, self.t0.to('um').v
    
    def left(self):
        self.theta -= self.dtheta
        print('theta = %d' % self.theta)
        self.by_time = []
        self.updating = 'angle'
        return True

    def right(self):
        self.theta += self.dtheta
        print('theta = %d' % self.theta)
        self.by_time = []
        self.updating = 'angle'
        return True

    def up(self):
        if self.TS is not None and self.i < len(self.TS) - 1:
            self.i += 1
            print('i = %d' % self.i)
            return True
        return False

    def down(self):
        if self.TS is not None and self.i > 0:
            self.i -= 1
            print('i = %d' % self.i)
            return True
        return False

    def get_dP(self):
        return self._get_dP(self.i)
    
    def _get_dP(self, i):
        key = (i, self.theta % 360)
        if key in self._cache:
            return self._cache[key]
        
        if self.TS is not None:
            self.F = self.TS[i]

        th = np.pi/180 * self.theta
        N = [np.sin(th), 0, np.cos(th)]
        print('projecting')
        Bx = yt.off_axis_projection(self.F, self.C, N, self.W, self.px, 'magnetic_field_x', north_vector=[0,1,0], num_threads=4)
        By = yt.off_axis_projection(self.F, self.C, N, self.W, self.px, 'magnetic_field_y', north_vector=[0,1,0], num_threads=4)
        Bz = yt.off_axis_projection(self.F, self.C, N, self.W, self.px, 'magnetic_field_z', north_vector=[0,1,0], num_threads=4)
        print('done')
        # S = [ cos(th), 0, -sin(th)]
        # T = [ 0, 1, 0 ]
        # U = N

        Bs = cm * ((Bx * np.cos(th) - Bz * np.sin(th) )/cm) # .to_equivalent('gauss', 'CGS')
        Bt = cm *  (By/cm) # .to_equivalent('gauss', 'CGS')

        if self.smoothing > 0:
            Bs = gaussian_filter(Bs / (gauss*cm), self.smoothing) * gauss*cm
            Bt = gaussian_filter(Bt / (gauss*cm), self.smoothing) * gauss*cm

        # The momenta here will be converted to PIC units (P / (m_e c))
        dPs = - ((q * Bt) / (m*c**2)).to('1').v
        dPt = + ((q * Bs) / (m*c**2)).to('1').v
        
        dPs *= self.strength_scale
        dPt *= self.strength_scale
        
        self._cache[key] = (dPs, dPt)

        return dPs, dPt
    
    def dP_at_time(self, t):
        ''' t in ps '''
        if t < self.times.min():
            raise RuntimeWarning('Time before start of simulation range')
            return self._get_dP(0)
        if t > self.times.max():
            print('Time after end of simulation range')
            return self._get_dP(len(self.times) - 1)
        b = np.digitize(t, self.times)
        pre_time = self.times[b - 1]
        dP_pre = self._get_dP(b - 1)
        post_time = self.times[b]
        dP_post = self._get_dP(b)
        f = (t - pre_time) / (post_time - pre_time)
        dPs = dP_pre[0] + f * (dP_post[0] - dP_pre[0])
        dPt = dP_pre[1] + f * (dP_post[1] - dP_pre[1])
        return dPs, dPt
    
    def save(self):
        np.savez(self.cachefile, cache=self._cache)