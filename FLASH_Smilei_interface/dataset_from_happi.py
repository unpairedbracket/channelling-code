import numpy as np

import happi
import yt

Q = yt.YTQuantity
c = yt.physical_constants.speed_of_light

def change_magnetic_field_name(name):
    if name == 'Bx':
        return 'magnetic_field_x'
    if name == 'By':
        return 'magnetic_field_y'
    if name == 'Bz':
        return 'magnetic_field_z'
    return name

def to_dataset(S, timestep, fields, **kwargs):
    omega_r = Q( S._reference_angular_frequency_SI, 's**-1')
    L_r = c / omega_r
    data = { change_magnetic_field_name(name): S.Field(number, name, **kwargs, units=[unit]).getData(timestep=timestep)[0] * Q(1, unit) for name, number, unit in fields }
    ddim = data[change_magnetic_field_name(fields[0][0])].shape
    length_unit = 'um'
    time_unit = 'fs'
    bbox = np.array( [[0, L_i] for L_i in S.namelist.Main.grid_length] ) * L_r
    if 'subset' in kwargs:
        subset = kwargs['subset']
        axes = ['x', 'y', 'z']
        for i, ax in enumerate(axes):
            if ax in subset:
                bbox[i, :] = subset[ax] * L_r

    return yt.load_uniform_grid(data, ddim, length_unit=length_unit, bbox=(bbox / Q(1,length_unit)).to('1'), time_unit=time_unit)

