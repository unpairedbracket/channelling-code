import numpy as np

import matplotlib.pyplot as plt

import yt
from yt import YTQuantity as Q

q = yt.physical_constants.charge_proton
M = yt.physical_constants.mass_hydrogen
m = yt.physical_constants.mass_electron
c = yt.physical_constants.speed_of_light

cm = Q(1, 'cm')

class ProtonRadiography():

    def __init__(self, f):
        if type(f) is str:
            self.F = yt.load(f)
        else:
            self.F = f

        self.C = self.F.domain_center
        
        self.Ws = np.sqrt( self.F.domain_width[0]**2 + self.F.domain_width[2]**2 )
        self.Wt = self.F.domain_width[1]
        self.W = [self.Ws, self.Wt, self.Ws]

    def proton_dP(self, theta, resolution_um, energy_MeV):
        th = np.pi/180 * theta
        N = [np.sin(th), 0, np.cos(th)]
        
        resolution = Q(resolution_um, 'um')
        px_s = int(self.Ws / resolution) + 1
        px_t = int(self.Wt / resolution) + 1
        px = [px_s, px_t]
        
        Bx = yt.off_axis_projection(self.F, self.C, N, self.W, px, 'magnetic_field_x', north_vector=[0,1,0])
        By = yt.off_axis_projection(self.F, self.C, N, self.W, px, 'magnetic_field_y', north_vector=[0,1,0])
        Bz = yt.off_axis_projection(self.F, self.C, N, self.W, px, 'magnetic_field_z', north_vector=[0,1,0])

        # S = [ cos(th), 0, -sin(th)]
        # T = [ 0, 1, 0 ]
        # U = N

        Bs = cm * ((Bx * np.cos(th) - Bz * np.sin(th))/cm).to_equivalent('gauss', 'CGS')
        Bt = cm *  (By/cm).to_equivalent('gauss', 'CGS')

        s0 = np.linspace( -self.Ws, self.Ws, px_s ) / 2
        t0 = np.linspace( -self.Wt, self.Wt, px_t ) / 2

        [S0, T0] = np.meshgrid(s0, t0, indexing='ij')

        # The momenta here will be converted to PIC units (P / (m_e c))
        # F ~= N x B = [ 
        dPs = - (q * Bt) / (m*c**2)
        dPt = + (q * Bs) / (m*c**2)

        E_normalised = Q(energy_MeV, 'MeV') / (m*c**2)
        P0 = np.sqrt((E_normalised + M/m)**2 - (M/m)**2 )

        return S0, T0, dPs, dPt, P0

def plot_deformed_grid(S, T, N):
    plt.plot(S[::N, ::N], T[::N, ::N], 'k', S[::N, ::N].T, T[::N, ::N].T, 'k')
