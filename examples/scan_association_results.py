from aiida.orm import load_node
from aiida import load_profile
from ase.units import Hartree, J
from scipy.constants import R, Avogadro
import numpy as np

load_profile()

relaxed_structures = {
    "PF6": {
        "pair": 2926,
        "anion": 2910,
        "cation": 2681
    },
    "BoPh": {
        "pair": 2956,
        "anion": 2923,
        "cation": 2681,
    },
    "BPh": {
        "pair": 2936,
        "anion": 2917,
        "cation": 2681
    },
    "Bhfip": {
        "pair": 2950,
        "anion": 2920,
        "cation": 2681
    },
}

calculations = {
    "PF6": {
        "pair": 2867,
        "anion": 2901,
        "cation": 2667
    },
    "BoPh": {
        "pair": 2861,
        "anion": 2892,
        "cation": 2667,
    },
    "BPh": {
        "pair": 2864,
        "anion": 2898,
        "cation": 2667
    },
    "Bhfip": {
        "pair": 2858,
        "anion": 2895,
        "cation": 2667
    },
}

for molecule in calculations.keys():
    print(load_node(calculations[molecule]["pair"]).inputs.parameters.get_dict())
    pair_params = load_node(calculations[molecule]["pair"]).outputs.output_parameters.get_dict()
    anion_params = load_node(calculations[molecule]["anion"]).outputs.output_parameters.get_dict()
    cation_params = load_node(calculations[molecule]["cation"]).outputs.output_parameters.get_dict()
    E_pair = pair_params["scfenergies"][-1] # + pair_params["dispersionenergies"][-1]
    E_anion = anion_params["scfenergies"][-1] # + anion_params["dispersionenergies"][-1]
    E_cation = cation_params["scfenergies"][-1] # + cation_params["dispersionenergies"][-1]

    dE = E_pair - (E_anion + E_cation)
    print(molecule, f"{dE:.3e}")
    T = 298.15 # orca default
    # R is in J/(mol*K)
    dE = (dE / J) * Avogadro # J/mol
    K = np.exp(-dE/(R*T))
    # print(molecule, f"{K:.3e}")

    struct = load_node(relaxed_structures[molecule]["pair"])
    atoms = struct.get_ase()
    symbols = atoms.get_chemical_symbols()
    if "B" in symbols:
        B_idx = symbols.index("B")
        Na_idx = symbols.index("Na")
        d = np.linalg.norm(atoms.positions[B_idx] - atoms.positions[Na_idx])
    elif "P" in symbols:
        P_idx = symbols.index("P")
        Na_idx = symbols.index("Na")
        d = np.linalg.norm(atoms.positions[P_idx] - atoms.positions[Na_idx])
    # print(molecule, f"{d:.3e}")
