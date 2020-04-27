from layer import Layer
import colours

known_film_types = {
        'HD810': [
            {'name': 'gelatine', 'thickness': 0.75, 'density': 1.2},
            {'name': 'HD810', 'thickness': 6.5, 'density': 1.08, 'colour': colours.calibrated('HD810')},
            {'name': 'polyester', 'thickness': 96.52, 'density': 1.35}
        ],
        'HDV2': [
            {'name': 'HD810', 'thickness': 8, 'density': 1.08, 'colour': colours.calibrated('HDV2')},
            {'name': 'polyester', 'thickness': 97, 'density': 1.35}
        ],
        'EBT2': [
            {'name': 'polyester', 'thickness': 50, 'density': 1.35},
            {'name': 'polyester', 'thickness': 25, 'density': 1.2}, # Adhesive
            {'name': 'polyester', 'thickness': 5, 'density': 1.2}, # Top coat
            {'name': 'EBT', 'thickness': 30, 'density': 1.2, 'colour': colours.calibrated('EBT2')},
            {'name': 'polyester', 'thickness': 175, 'density': 1.35}
        ],
        'EBT3': [
            {'name': 'polyester', 'thickness': 125, 'density': 1.35},
            {'name': 'EBT', 'thickness': 30, 'density': 1.2, 'colour': colours.calibrated('EBT3')},
            {'name': 'polyester', 'thickness': 125, 'density': 1.35}
        ]
}

def get_film_layers(film_type):
    if film_type in known_film_types:
        return [Layer(**layer) for layer in known_film_types[film_type]]

    else:
        print('UNKNOWN FILM TYPE HELP')
        return []

