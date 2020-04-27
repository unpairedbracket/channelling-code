import pickle

import numpy as np
from numpy import pi

import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator

import yt
from yt import YTQuantity as Q

c = yt.physical_constants.speed_of_light
m = yt.physical_constants.mass_electron
e = yt.physical_constants.charge_proton.to_equivalent('C', 'SI')
eps_0 = yt.physical_constants.eps_0
cm = Q(1, 'cm')

class Laser:
    def __init__(self, wavelength_nm=800):
        self.set_wavelength(wavelength_nm)

    def set_wavelength(self, wavelength_nm):
        lbd = Q(wavelength_nm, 'nm')
        self.k =  (2 * pi / lbd).to('1/cm')
        self.omega = (c * self.k).to('1/s')
        self.n_c = (m * eps_0 * self.omega**2 / e**2).to('cm**-3')

    def get_critical_density(self):
        return self.n_c


laser_singleton = Laser()

@yt.derived_field('nele_targ', units='cm**-3')
def _nele_targ(field, data):
    return data['nele'] * data['targ'] / ( data['targ'] + 13/14 * (1 - data['targ']))

@yt.derived_field('deta', units='1')
def _deta(field, data):
    return np.sqrt(1+0j - data['nele_targ']/laser_singleton.get_critical_density() ).real - 1

@yt.derived_field('kappa', units='1')
def _kappa(field, data):
    return np.sqrt(1+0j - data['nele_targ']/laser_singleton.get_critical_density() ).imag

class GatedInterferometry:
    simulation = None
    L = None

    def __init__(self, sim_filename, fringe_um, downscale_factor, lambda_nm=None, fringe_angle=0, refine_level=None):

        if lambda_nm is not None:
            laser_singleton.set_wavelength(lambda_nm)

        self.set_fringe(fringe_um, fringe_angle)

        self.downscale_factor = downscale_factor
        self.level = refine_level
        self.open_simulation(sim_filename)

    def set_fringe(self, fringe_um, fringe_angle):
        fringe_spacing =  Q(fringe_um, 'um')
        self.k_s =  (2 * pi / fringe_spacing).to('1/cm') * np.array([np.cos(fringe_angle), np.sin(fringe_angle)])

    def set_distance(self, L_cm):
        self.L = Q(L_cm, 'cm')

    def open_simulation(self, sim_filename):
        self.simulation = yt.load(sim_filename)

        self.simulation.periodicity = (True, True, True)

        self.time = self.simulation.current_time.to('ps')
        
        x0, z0 = self.simulation.domain_left_edge[[0, 2]]
        Dx, Dz = self.simulation.domain_width[[0, 2]]
        Nx, Nz = self.simulation.domain_dimensions[[0, 2]]
        self.level = self.level or self.simulation.max_level

        self.x = np.linspace(x0, x0+Dx, Nx * 2**self.level)
        self.z = np.linspace(z0, z0+Dz, Nz * 2**self.level)
        self.x_px = np.linspace(x0, x0+Dx, Nx * 2**self.level / self.downscale_factor)
        self.z_px = np.linspace(z0, z0+Dz, Nz * 2**self.level / self.downscale_factor)

    def setup(self):
        sim = self.simulation
        total_distance = sim.domain_width[1]
        dims = sim.domain_dimensions * sim.refine_by**self.level
        edge = sim.domain_left_edge

        scg = sim.smoothed_covering_grid(
            level=self.level,
            left_edge=edge,
            dims=dims
        )

        self.deta = (scg['deta'].mean(axis=(1)) * total_distance)
        self.kapa = (scg['kappa'].mean(axis=(1)) * total_distance)
        self.deta_x, self.deta_z = np.gradient(self.deta / cm, self.x / cm, self.z / cm)

    def load(self, suffix=None):
        filename = f'cache/GOI_{suffix}.p' if suffix else 'GOI.p'
        self.deta, self.kapa = pickle.load(open(filename, 'r+b'))
        self.deta_x, self.deta_z = np.gradient(self.deta / cm, self.x / cm, self.z / cm)

    def save(self, suffix=None):
        filename = f'cache/GOI_{suffix}.p' if suffix else 'GOI.p'
        pickle.dump((self.deta, self.kapa), open(filename, 'w+b'))

    def run(self, N=None, plot=False):
        refract = (self.L is not None and self.L > 0)
        if refract and N is None:
            print("Need to specify a N when refraction is a factor")
            N = 1

        x, z = self.x, self.z

        dphi_interp = RegularGridInterpolator((x, z), (laser_singleton.k * self.deta).to('1'), bounds_error=False, fill_value=0)
        damp_interp = RegularGridInterpolator((x, z), (laser_singleton.k * self.kapa).to('1'), bounds_error=False, fill_value=0)
        deta_x_interp = RegularGridInterpolator((x, z), self.deta_x, bounds_error=False, fill_value=0)
        deta_z_interp = RegularGridInterpolator((x, z), self.deta_z, bounds_error=False, fill_value=0)

        I_interf = np.zeros((self.x_px.size - 1, self.z_px.size - 1))
        I_shadog = np.zeros((self.x_px.size - 1, self.z_px.size - 1))

        if refract:
            X0, Z0 = np.meshgrid(self.x_px[:-1], self.z_px[:-1], indexing='ij')
            dx = self.x_px.ptp() / (self.x_px.size - 1)
            dz = self.z_px.ptp() / (self.z_px.size - 1)
            for i in np.arange(N):
                x_rnd = X0 + np.random.uniform((0*dx/cm).to('1'), (1*dx/cm).to('1'), X0.shape) * cm
                z_rnd = Z0 + np.random.uniform((0*dz/cm).to('1'), (1*dz/cm).to('1'), Z0.shape) * cm

                X_rnd = x_rnd + self.L * deta_x_interp((x_rnd, z_rnd))
                Z_rnd = z_rnd + self.L * deta_z_interp((x_rnd, z_rnd))
                d = self.L * (np.sqrt( 1 + deta_x_interp((x_rnd, z_rnd))**2 + deta_z_interp((x_rnd, z_rnd))**2 ) - 1)

                amplitudes = np.exp(1j * (dphi_interp((x_rnd, z_rnd)) + self.k_s[0] * X_rnd + self.k_s[1] * Z_rnd + laser_singleton.k * d)) * np.exp(-damp_interp((x_rnd, z_rnd)))

                a_r, _, _ = np.histogram2d(X_rnd.flatten(), Z_rnd.flatten(), weights=amplitudes.real.flatten(), bins=[self.x_px, self.z_px])
                a_i, _, _ = np.histogram2d(X_rnd.flatten(), Z_rnd.flatten(), weights=amplitudes.imag.flatten(), bins=[self.x_px, self.z_px])

                amp = a_r + 1j * a_i
                
                I_i = np.abs( 1 + amp ) ** 2 / 4
                I_s = np.abs( amp ) ** 2

                I_interf += I_i / N
                I_shadog += I_s / N

        else:
            x_v, z_v = np.meshgrid(self.x_px, self.z_px, indexing='ij')
            corner_dphi = dphi_interp(((x_v/cm).to('1'), (z_v/cm).to('1')))
            Dx = ( corner_dphi[1:, 1:] + corner_dphi[1:, :-1] - corner_dphi[:-1, 1:] - corner_dphi[:-1, :-1] ) / 4
            Dz = ( corner_dphi[1:, 1:] - corner_dphi[1:, :-1] + corner_dphi[:-1, 1:] - corner_dphi[:-1, :-1] ) / 4

            x_centre = (self.x_px[1:] + self.x_px[:-1])/2
            z_centre = (self.z_px[1:] + self.z_px[:-1])/2
            X, Z = np.meshgrid(x_centre, z_centre, indexing='ij')
            damp_centre = damp_interp(((X/cm).to('1'), (Z/cm).to('1')))
            dphi_centre = dphi_interp(((X/cm).to('1'), (Z/cm).to('1')))
            X_dx = ( x_v[1:, 1:] + x_v[1:, :-1] - x_v[:-1, 1:] - x_v[:-1, :-1] ) / 4
            Z_dx = ( z_v[1:, 1:] - z_v[1:, :-1] + z_v[:-1, 1:] - z_v[:-1, :-1] ) / 4

            I_interf = ( 1 +  np.exp(-2*damp_centre) + 2*np.sinc((Dx + self.k_s[0] * X_dx)/np.pi) * np.sinc((Dz + self.k_s[1] * Z_dx)/np.pi) * np.cos(dphi_centre + self.k_s[0] * X + self.k_s[1] * Z) * np.exp(-damp_centre) ) / 4
            I_shadog = np.exp(-2*damp_centre)

        if plot:
            fig = plt.figure()
            axs = fig.subplots(nrows=1, ncols=2)
            fig.suptitle('Gated Interferometry at %f ps' % self.time)
            # Shadowgraphy
            axs[0].pcolormesh(self.z_px, self.x_px, I_shadog, vmin=0, vmax=1)
            axs[0].set_title('Shadowgraphy')
            axs[0].set_aspect(1)
            # Interferometry
            axs[1].pcolormesh(self.z_px, self.x_px, I_interf, vmin=0, vmax=1)
            axs[1].set_title('Interferometry')
            axs[1].set_aspect(1)

        return I_shadog, I_interf, self.x_px, self.z_px

class StreakedInterferometry:
    simulations = None
    L = None

    def __init__(self, sim_filename_pattern, fringe_um, downscale_factor, lambda_nm=None):

        if lambda_nm is not None:
            laser_singleton.set_wavelength(lambda_nm)

        self.set_fringe(fringe_um)

        self.open_simulation(sim_filename_pattern)
        self.set_scale_factor(downscale_factor)

    def set_fringe(self, fringe_um):
        fringe_spacing =  Q(fringe_um, 'um')
        self.k_s =  (2 * pi / fringe_spacing).to('1/cm')

    def set_distance(self, L_cm):
        self.L = Q(L_cm, 'cm')

    def open_simulation(self, sim_filename_pattern):
        self.simulations = yt.load(sim_filename_pattern)

    def set_scale_factor(self, new_factor):
        self.downscale_factor = new_factor
        base_sim = self.simulations[-1]
        x0 = base_sim.domain_left_edge
        Dx = base_sim.domain_width
        Nx = base_sim.domain_dimensions
        self.level = base_sim.max_level

        self.x, self.y, self.z = ( np.linspace(xi, xi+dx, nx * 2**self.level) for xi, dx, nx in zip(x0, Dx, Nx) )
        self.x_px = np.linspace(x0[0], x0[0]+Dx[0], Nx[0] * 2**self.level / self.downscale_factor)

    def get_projections(self, sim):
        sim.periodicity = (True, True, True)
        time = sim.current_time.to('ps')
        total_distance = sim.domain_width[1]
        dims = sim.domain_dimensions * 2**self.level
        edge = sim.domain_left_edge
        if sim.dimensionality == 3:
            dx = sim.domain_width / dims
            dims[2] = 3
            edge[2] = - 3 * dx[2] / 2
        scg = sim.smoothed_covering_grid(
            level=self.level,
            left_edge=edge,
            dims=dims
        )

        dphi = (laser_singleton.k * scg['deta'].mean(axis=(1,2)) * total_distance).to('1')
        damp = (laser_singleton.k * scg['kappa'].mean(axis=(1,2)) * total_distance).to('1')
        deta_x = np.gradient(dphi, self.x / cm) / (laser_singleton.k * cm)
        deta_zz = np.zeros_like(deta_x) / cm

        if sim.dimensionality == 3:
            deta_zz = ((scg['deta'][:,:,0] - 2*scg['deta'][:,:,1] + scg['deta'][:,:,2])/dx[2]**2).mean(axis=1)*total_distance

        return time, dphi, damp, deta_x, deta_zz


    def setup(self):
        self.lines = [ self.get_projections(sim) for sim in self.simulations ]

    def load(self, suffix=None):
        filename = 'SOI_'+suffix+'.p' if suffix else 'SOI.p'
        self.lines = pickle.load(open(filename, 'r+b'))

    def save(self, suffix=None):
        filename = 'SOI_'+suffix+'.p' if suffix else 'SOI.p'
        pickle.dump(self.lines, open(filename, 'w+b'))

    def run(self, N):
        refract = ( self.L is not None and self.L > 0 )
        x = self.x
        t = np.zeros((len(self.simulations),))

        I_interf = np.zeros((self.x_px.size - 1))
        I_shadog = np.zeros((self.x_px.size - 1))
        dx = self.x_px.ptp() / (self.x_px.size - 1)

        all_interf = np.zeros((t.size, self.x_px.size - 1))
        all_shadog = np.zeros((t.size, self.x_px.size - 1))

        for idx, (time, dphi, damp, deta_x, deta_zz) in enumerate(self.lines):

            dphi_interp = RegularGridInterpolator(((x/cm).to('1'),), dphi, bounds_error=False, fill_value=0)
            damp_interp = RegularGridInterpolator(((x/cm).to('1'),), damp, bounds_error=False, fill_value=0)

            I_interf[:] = 0
            I_shadog[:] = 0

            if refract:
                deta_x_interp = RegularGridInterpolator(((x/cm).to('1'),), deta_x, bounds_error=False, fill_value=0)

                if deta_zz is not None:
                    deta_zz_interp = RegularGridInterpolator(((x/cm).to('1'),), self.L * deta_zz, bounds_error=False, fill_value=0)

                for i in np.arange(N):
                    x_rnd = (self.x_px[:-1]/cm) + np.random.uniform((0*dx/cm).to('1'), (1*dx/cm).to('1'), I_interf.size)

                    d = self.L * (np.sqrt( 1 + deta_x_interp(x_rnd)**2 ) - 1)
                    amplitudes = np.exp(1j * (dphi_interp(x_rnd) + self.k_s * x_rnd*cm + laser_singleton.k * d)) * np.exp(-damp_interp(x_rnd))

                    if deta_zz is not None:
                        amplitudes /= np.sqrt(np.abs(1 + deta_zz_interp(x_rnd)))

                    X_rnd = x_rnd*cm + self.L * deta_x_interp(x_rnd)
                    a_r, _ = np.histogram(X_rnd, weights=amplitudes.real, bins=self.x_px)
                    a_i, _ = np.histogram(X_rnd, weights=amplitudes.imag, bins=self.x_px)

                    amp = a_r + 1j * a_i
                    
                    I_i = np.abs( 1 + amp ) ** 2 / 4
                    I_s = np.abs( amp ) ** 2

                    I_interf += I_i / N
                    I_shadog += I_s / N
            else:
                dphi_px = dphi_interp((self.x_px/cm).to('1'))
                damp_px = damp_interp((self.x_px/cm).to('1'))
                x_centre = (self.x_px[1:] + self.x_px[:-1])/2
                x_dx = (self.x_px[1:] - self.x_px[:-1])/2
                dphi_centre = (dphi_px[1:] + dphi_px[:-1])/2
                diff_dphis = (dphi_px[1:] - dphi_px[:-1])/2
                damp_centre = (damp_px[1:] + damp_px[:-1])/2
                I_interf = (1 + np.sinc((diff_dphis + self.k_s * x_dx)/np.pi) * np.cos(dphi_centre + self.k_s * x_centre) * np.exp(-2*damp_centre) ) / 2
                I_shadog = np.ones_like(I_interf)

            t[idx] = time
            all_interf[idx] = I_interf
            all_shadog[idx] = I_shadog

        fig = plt.figure()
        axs = fig.subplots(nrows=1, ncols=2)
        # Shadowgraphy
        axs[0].pcolormesh(t, self.x_px, all_shadog.T, vmin=0, vmax=1)
        # Interferometry
        axs[1].pcolormesh(t, self.x_px, all_interf.T, vmin=0, vmax=1)

        return all_interf, all_shadog

