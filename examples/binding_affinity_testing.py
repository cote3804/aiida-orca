# script to test binding affinity calculation from AssociationWorkCHain

from aiida.orm import load_node, CalcJobNode
from aiida import load_profile
from aiida_orca.calculations import OrcaCalculation
from ase import Atoms, Atom
from ase.units import Hartree, kJ, J
from scipy.constants import R, Avogadro
import numpy as np

load_profile()

res = {}

worknode = load_node(1246)
out = worknode.get_outgoing()
test_out = list(out)[0]
for output in list(out):
    if isinstance(output.node, CalcJobNode):
        # grabbing calculations
        inputs = output.node.base.links.get_incoming()
        outputs = output.node.base.links.get_outgoing()
        params = inputs.get_node_by_label("parameters")
        struct = inputs.get_node_by_label("structure")
        print(struct.get_ase().symbols)
        print(params.get_dict())
        charge = params["charge"]
        results = outputs.get_node_by_label("output_parameters")
        print(list(results.keys()))
        res[str(charge)] = results["freeenergy"]

print(res)

T = 298.15
dG = res["0"] - (res["1"] + res["-1"]) # Hartree
dG = (dG * Hartree / J) * Avogadro # J/mol
print(dG)
K = np.exp(-dG/(R*T))
print(K)

