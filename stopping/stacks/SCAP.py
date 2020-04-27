import numpy as np

from film import get_film_layers
from filters import get_filter

# Films
EBT2 = get_film_layers('EBT2')
EBT3 = get_film_layers('EBT3')
HDV2 = get_film_layers('HDV2')
HD810= get_film_layers('HD810')

# Filters
Al = get_filter('Al', 6)
Fe = get_filter('Fe', 250)
Fe_thick = get_filter('Fe', 500)

def get_SCAP_stack(shot, flatten=True):
    if shot < 5193:
        print('Shot number too low!')
        return []
    if shot > 5238:
        print ('Shot number too high!')

    layers = [
        Al,
        HDV2, HDV2, HDV2, HDV2,
        Fe,
        HDV2, EBT3,
        Fe,
        HDV2, EBT3,
        Fe,
        HDV2, EBT3,
        Fe,
        HDV2, EBT3,
        Fe_thick,
        EBT3, EBT3,
        Fe_thick,
        EBT3,
        Fe_thick,
        EBT3,
        Fe_thick,
        EBT3,
        Fe_thick,
        EBT3
    ]

    config_edges = [ 5193, 5195, 5198, 5230, 5234, 5236 ]
    missing_layers = [ 14, 8, 6, 4, 2, 0 ]
    config = np.digitize(shot, config_edges)
    print('Using config', config)


    if missing_layers[config-1] > 0:
        layers = layers[:-missing_layers[config-1]]

    if config == 1:
        for i in [10, 7]:
            layers.pop(i)

    if flatten:
        layers = sum(layers, [])

    return layers
