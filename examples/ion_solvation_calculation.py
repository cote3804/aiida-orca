# Calculating a kind of association energy where I start from the loan salt in implicit solvent,
# and complex the cation with 2 DG molecules and opitmize the structure.
# dG_a = G_solvated_salt - (G_bare_salt + 2*G_DG)
# A more negative dG_a means a more favorable association

from aiida.orm import load_node
from aiida import load_profile
from ase.visualize import view
load_profile("cooper")

pks = {
    "BoPh": {
        "bare_salt": 3965,
        "solvated_salt": 3889,
        "anion": 2892,
    },
    "BPh": {
        "bare_salt": 3961,
        "solvated_salt": 3847,
        "anion": 2898,
    },
    "DG": {
        "2_hug": 3242
    },
    "Na":{
        "2DG_solvated": 2671
    }
}

def get_energy(outputs:dict) -> float:
    E_SCF = outputs["scfenergies"][-1]
    # E_dispersion = outputs["dispersionenergies"][-1]
    E_dispersion = 0
    return E_SCF + E_dispersion


def calculate_cation_solvation():
    salts = ["BoPh", "BPh"]
    # salts = ["BoPh"]
    for salt in salts:
        solvated_calc = load_node(pks[salt]["solvated_salt"])
        bare_calc = load_node(pks[salt]["bare_salt"])
        solvent_calc = load_node(pks["DG"]["2_hug"])
        # view(solvated_calc.outputs.relaxed_structure.get_ase())
        # view(bare_calc.outputs.relaxed_structure.get_ase())
        # view(solvent_calc.outputs.relaxed_structure.get_ase())
        solvated_outputs = solvated_calc.outputs.output_parameters.get_dict()
        bare_outputs = bare_calc.outputs.output_parameters.get_dict()
        solvent_outputs = solvent_calc.outputs.output_parameters.get_dict()
        E_solvated = get_energy(solvated_outputs)
        E_bare = get_energy(bare_outputs)
        E_solvent = get_energy(solvent_outputs)
        Ea = E_solvated - (E_bare + E_solvent)
        print(salt, Ea)

def calculate_infinite_separation():
    salts = ["BoPh", "BPh"]
    # salts = ["BoPh"]
    for salt in salts:
        pair_calc = load_node(pks[salt]["solvated_salt"])
        anion_calc = load_node(pks[salt]["anion"])
        cation_calc = load_node(pks["Na"]["2DG_solvated"])
        # view(pair_calc.outputs.relaxed_structure.get_ase())
        # view(anion_calc.outputs.relaxed_structure.get_ase())
        # view(cation_calc.outputs.relaxed_structure.get_ase())
        pair_outputs = pair_calc.outputs.output_parameters.get_dict()
        anion_outputs = anion_calc.outputs.output_parameters.get_dict()
        cation_outputs = cation_calc.outputs.output_parameters.get_dict()
        E_pair = get_energy(pair_outputs)
        E_anion = get_energy(anion_outputs)
        E_cation = get_energy(cation_outputs)
        Ea = E_pair - (E_anion + E_cation)
        print(salt, Ea)


if __name__ == "__main__":
    calculate_cation_solvation()
    calculate_infinite_separation()
