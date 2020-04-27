from layer import Layer

materials = {
        'Fe': {'name': 'iron', 'density': 7.874},
        'Al': {'name': 'aluminium', 'density': 2.70}
}

def get_filter(material, thickness):
    ''' Returns a list for concatenation into stacks '''
    if material in materials:
        return [ Layer(**materials[material], thickness=thickness) ]

    else:
        print('UNKNOWN FILM TYPE HELP')
        return []

