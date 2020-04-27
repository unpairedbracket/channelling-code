# ----------------------------------------------------------------------------------------
# 					SIMULATION PARAMETERS FOR THE PIC-CODE SMILEI
# ----------------------------------------------------------------------------------------

import math
import random
from numpy import exp, pi, sqrt, log, s_, load, interp
from scipy.interpolate import RegularGridInterpolator

l0 = 2.*pi				# reference wavelength
t0 = l0					# optical cycle

Lsim = [1024*l0, 80.*l0]	        # length of the simulation
Tsim = 1500.*t0				# duration of the simulation
resx = 16.				# nb of cells in on laser wavelength
resy = 16.
rest = 40.				# nb of timestep in one optical cycle

energy = 100.0 # J
pulse_length = 500 # fs
harmonic = 1.0
spot = 5.0 # wavelengths

power = energy / pulse_length # PW
P0 = 86.0e-6 # 86GW = 86e-6 PW

a0 = sqrt( power / P0 ) / ( spot * harmonic )
#a0 = 44 / spot
midplane = 0
fs = 0.3 # cycles
pulse_cycles = pulse_length * fs

# x and y in micron
FLASH = load('flash_smilei_20deg.npz')
micron = l0
x = FLASH['x'] * micron
OTS = 0*micron
focal_pos = [Lsim[0] - (OTS+800*micron), midplane]

# Density and temperature in units of critical density and electron rest energy
nele = FLASH['nele']
tele = FLASH['tele'] / 511.00
tion = FLASH['tion'] / 511.00

#nele_interp = RegularGridInterpolator((x,), nele, bounds_error=False, fill_value=0)
#tele_interp = RegularGridInterpolator((x,), tele, bounds_error=False, fill_value=0)
#tion_interp = RegularGridInterpolator((x,), tion, bounds_error=False, fill_value=0)

#get_nele = lambda x, y: nele_interp((OTS + x,)) 
#get_tele = lambda x, y: tele_interp((OTS + x,)) 
#get_tion = lambda x, y: tion_interp((OTS + x,)) 

get_nele = lambda x_, y_: interp(OTS + x_, x, nele)
get_tele = lambda x_, y_: interp(OTS + x_, x, tele)
get_tion = lambda x_, y_: interp(OTS + x_, x, tion)

random.seed(random.random()*smilei_mpi_rank)

Main(
    geometry = "AMcylindrical",
    number_of_AM = 3,

    interpolation_order = 2,

    cell_length = [l0/resx, l0/resy],
    grid_length  = Lsim,

    number_of_patches = [ 16*32, 16*4 ],

    timestep = t0/rest,
    simulation_time = Tsim,

    EM_boundary_conditions = [
	['silver-muller'],
	['buneman'],
    ],

    print_every = 100,
    solve_poisson = False,

    random_seed = random.randint(0,smilei_rand_max)
)

LoadBalancing(
    initial_balance = True,
    every = 500,
    cell_load = 1.,
    frozen_particle_load = 0.1
)

LaserGaussianAM(
    box_side        = "xmax",
    omega           = harmonic,
    a0              = a0,
    focus           = focal_pos,
    waist           = spot * l0,
    time_envelope   = tsin2plateau(fwhm=10.*t0, plateau=(pulse_cycles - 10.)*t0)
)

Species(
    name = 'ele',
    position_initialization = 'random',
    momentum_initialization = 'maxwell-juettner',
    temperature = [get_tele],
    ionization_model = 'none',
    particles_per_cell = 24,
    c_part_max = 1.0,
    mass = 1.0,
    charge = -1.0,
    charge_density = get_nele,
    time_frozen = 0.,
    boundary_conditions = [
    	["reflective", "reflective"],
    	["periodic", "remove"],
    ],
)

Species(
    name = 'ion',
    position_initialization = 'random',
    momentum_initialization = 'maxwell-juettner',
    temperature = [get_tion],
    ionization_model = 'none',
    particles_per_cell = 16,
    c_part_max = 1.0,
    mass = 1836.0 * 6.5,
    charge = 3.5,
    charge_density = get_nele,
    time_frozen = 0,
    boundary_conditions = [
        ["reflective", "reflective"],
        ["periodic", "remove"],
    ],
)

def get_field_modes(fields, modes):
    return [ '{field}_mode_{mode}'.format(field=f, mode=n) for f in fields for n in modes ]

DiagProbe(
    every = 10 * rest,
    origin = [0, 0, 0],
    corners = [
        [Lsim[0], 0, 0],
	[0, 0, Lsim[1]]
    ],
    number = [1024 * 4, 80 * 4]
)

DiagProbe(
    every = 10 * rest,
    origin = [0, 0, 0],
    corners = [
        [Lsim[0], 0, 0],
	[0, 0, Lsim[1]]
    ],
    number = [1024 * 4, 80 * 4],
    fields = get_field_modes(['Rho_ele', 'Rho_ion'], [0,1,2])
)

DiagProbe(
    every = 10 * rest,
    origin = [Lsim[0]/2, -Lsim[1], -Lsim[1]],
    corners = [
        [Lsim[0]/2, +Lsim[1], -Lsim[1]],
	[Lsim[0]/2, -Lsim[1], +Lsim[1]]
    ],
    number = [160 * 4, 160 * 4]
)

DiagProbe(
    every = 10 * rest,
    origin = [Lsim[0]/2, -Lsim[1], -Lsim[1]],
    corners = [
        [Lsim[0]/2, +Lsim[1], -Lsim[1]],
	[Lsim[0]/2, -Lsim[1], +Lsim[1]]
    ],
    number = [160 * 4, 160 * 4],
    fields = get_field_modes(['Rho_ele', 'Rho_ion'], [0,1,2])
)

DiagProbe(
    every = 20 * rest,
    origin = [0 , -Lsim[1], -Lsim[1]],
    corners = [
	[Lsim[0], -Lsim[1], -Lsim[1]],
        [     0 , +Lsim[1], -Lsim[1]],
	[     0 , -Lsim[1], +Lsim[1]]
    ],
    number = [1024, 160, 160],
    fields = get_field_modes(['Rho_ele', 'Rho_ion'], [0,1,2]) + [ 'Bx', 'By', 'Bz' ]
)

