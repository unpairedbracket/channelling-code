from film import get_film_layers

# Films
EBT2 = get_film_layers('EBT2')
EBT3 = get_film_layers('EBT3')
HDV2 = get_film_layers('HDV2')
HD810= get_film_layers('HD810')

def get_ministack(flatten=True):
    layers = [ HD810, HDV2, EBT2, EBT3 ]
    if flatten:
        layers = sum(layers, [])

    return layers
