def ground_state_energy(element):
    if element == "Ag":
        ground_state_energy = -0.19820116
    elif element == "Al":
        ground_state_energy = -0.31455471
    elif element == "As":
        ground_state_energy = -1.70125462
    elif element == "Au":
        ground_state_energy = -0.18494148
    elif element == "Ba":
        ground_state_energy = -0.03101409
    elif element == "Be":
        ground_state_energy = -0.03829264
    elif element == "Bi":
        ground_state_energy = -1.32456394
    elif element == "Ca":
        ground_state_energy = -0.00669804
    elif element == "Cd":
        ground_state_energy = -0.01269252
    elif element == "Cr":
        ground_state_energy = -5.43535568
    elif element == "Cs":
        ground_state_energy = -0.13345046
    elif element == "Cu":
        ground_state_energy = -0.24231333
    elif element == "Ga":
        ground_state_energy = -0.27760085
    elif element == "Gd":
        ground_state_energy = -0.35543567
    elif element == "Ge":
        ground_state_energy = -0.77067288
    elif element == "In":
        ground_state_energy = -0.22759447
    elif element == "K":
        ground_state_energy = -0.15885132
    elif element == "La":
        ground_state_energy = -0.63873053
    elif element == "Li":
        ground_state_energy = -0.29780591  # Li_sv, 800eV
    elif element == "Mg":
        ground_state_energy = -0.00040000
    elif element == "Na":
        ground_state_energy = -0.21928380
    elif element == "P":
        ground_state_energy = -1.88664411
    elif element == "Pb":
        ground_state_energy = -0.58498619
    elif element == "Sb":
        ground_state_energy = -1.43005001
    elif element == "Sc":
        ground_state_energy = -1.98188993
    elif element == "Si":
        ground_state_energy = -0.87162900
    elif element == "Sn":
        ground_state_energy = -0.64316709
    elif element == "Sr":
        ground_state_energy = -0.02805273
    elif element == "Te":
        ground_state_energy = -0.41476686
    elif element == "Eu":
        ground_state_energy = 6.9898249e-07
    elif element == "Y":
        ground_state_energy = -2.26275664
    else:
        ground_state_energy = False

    return ground_state_energy


def atom_variable(element):
    if element == "Ag":
        atomic_length = 2.93
    elif element == "Al":
        atomic_length = 2.86
    elif element == "Au":
        atomic_length = 2.94
    elif element == "Ba":
        atomic_length = 4.35
    elif element == "Be":
        atomic_length = 2.20
    elif element == "Bi":
        atomic_length = 3.067
    elif element == "Ca":
        atomic_length = 3.90
    elif element == "Cd":
        atomic_length = 3.05
    elif element == "Cr":
        atomic_length = 2.45
    elif element == "Cs":
        atomic_length = 5.48
    elif element == "Cu":
        atomic_length = 2.57
    elif element == "Ga":
        atomic_length = 2.75
    elif element == "Ge":
        atomic_length = 2.503
    elif element == "Hf":
        atomic_length = 3.10
    elif element == "Hg":
        atomic_length = 3.37
    elif element == "In":
        atomic_length = 3.39
    elif element == "Ir":
        atomic_length = 2.74
    elif element == "K":
        atomic_length = 4.71
    elif element == "La":
        atomic_length = 3.703
    elif element == "Li":
        atomic_length = 3.05
    elif element == "Mg":
        atomic_length = 2.557
    elif element == "Mo":
        atomic_length = 2.727
    elif element == "Na":
        atomic_length = 3.73
    elif element == "Nb":
        atomic_length = 2.802
    elif element == "Os":
        atomic_length = 2.686
    elif element == "Pb":
        atomic_length = 3.5
    elif element == "Pd":
        atomic_length = 2.786
    elif element == "Pt":
        atomic_length = 2.81
    elif element == "Rb":
        atomic_length = 5.06
    elif element == "Re":
        atomic_length = 2.745
    elif element == "Rh":
        atomic_length = 2.705
    elif element == "Ru":
        atomic_length = 2.642
    elif element == "Sc":
        atomic_length = 3.196
    elif element == "Si":
        atomic_length = 2.368
    elif element == "Sn":
        atomic_length = 2.81
    elif element == "Sr":
        atomic_length = 4.268
    elif element == "Ta":
        atomic_length = 2.869
    elif element == "Ti":
        atomic_length = 2.87
    elif element == "Tl":
        atomic_length = 3.453
    elif element == "V":
        atomic_length = 2.642
    elif element == "W":
        atomic_length = 2.746
    elif element == "Y":
        atomic_length = 3.53
    elif element == "Zn":
        atomic_length = 2.655
    elif element == "Zr":
        atomic_length = 3.176
    else:
        atomic_length = False
    return atomic_length
