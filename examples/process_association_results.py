# process outcome of association workchains
from aiida.orm import load_node
from aiida import load_profile
from ase.units import Hartree
from ase.visualize import view

load_profile()

calculations = { # wB97X-D4 def2-TZVP
    "PF6": 3053,
    "BPh": 3057
}

calculations = { #r2SCAN-3c
    "PF6": 3622,
    "BPh": 3626,
    "BoPh": 3623,
    "Bhfip": 3629
}

def get_calc_type(calc_node):
    # assume pair has charge 0, cation has charge +1 and anion has charge -1
    inp = calc_node.inputs.parameters
    charge = inp.get_dict()["charge"]
    if charge == 0:
        return "pair"
    elif charge == 1:
        return "cation"
    elif charge == -1:
        return "anion"
    else:
        raise Exception

for mol, pk in calculations.items():
    n = load_node(pk)
    print(n.inputs.parameters.get_dict())
    calcs = n.base.links.get_outgoing().all_nodes()[:-2]
    energies = {}
    for calc in calcs:
        calc_type = get_calc_type(calc)
        print(mol, calc_type, calc.outputs.relaxed_structure.pk)
        # view(calc.outputs.relaxed_structure.get_ase())
        G = calc.outputs.output_parameters["scfenergies"][-1]
        # print(list(calc.outputs.output_parameters.keys()))
        energies[calc_type] = G
    dGa = (energies["pair"] - (energies["anion"] + energies["cation"]))
    print(mol, dGa)