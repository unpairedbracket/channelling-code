import numpy as np

def get_deposited_energies(stack, energies):
    for layer in stack:
        new_energies = layer.forwardpass(energies)
        energies, energies_dropped = new_energies, energies - new_energies
        yield energies_dropped

def get_principal_energies(stack):
    energies = np.zeros((0))
    for layer in reversed(stack):
        energies = np.insert(energies, 0, 0)
        energies = layer.backpass(energies)

    return energies
