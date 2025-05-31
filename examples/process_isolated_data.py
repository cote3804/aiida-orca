# process results for isolated molecules
# get association constants and redox potentials

from aiida.orm import load_node, QueryBuilder, CalcJobNode, CalcFunctionNode, StructureData
from aiida import load_profile
from ase.units import Ha
import pandas as pd

load_profile()

assoc_pks = [659, 661, 663, 667]
ion_pks = [681, 683, 686, 690]
calc = load_node(ion_pks[0])
out = calc.base.links.get_outgoing()
# print(list(out))
orca_calcs = []
free_energies = []
charges = []
molecules = []
for pk in ion_pks:
    calc = load_node(pk)
    incoming = calc.base.links.get_incoming().all_nodes()
    for inc in incoming:
        if isinstance(inc, StructureData):
            molecule = str(inc)
    for cal in calc.called:
        if isinstance(cal, CalcJobNode):
            orca_calcs.append(cal)
        for orca_calc in orca_calcs:
            params = orca_calc.outputs.output_parameters
            charge = params["charge"]
            if "freeenergy" in params.keys():
                free_energy = params["freeenergy"]
            else:
                free_energy = 0
            charges.append(charge)
            free_energies.append(free_energy)
            molecules.append(molecule)

df = pd.DataFrame({"free_energy": free_energies, "charge": charges})
print(df)

# qb = QueryBuilder()
# qb.append(assoc_pks)
# first = qb.first()
# print(first)
# res_d = qb.dict()
# print(res_d)