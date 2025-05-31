from aiida.orm import load_node
from aiida import load_profile
import numpy as np

load_profile("cooper")

#### BPh ####
hugged_pairs = [4303, 4832, 4310, 4333]
hugged_anions = [4320, 4324, 4327, 4333]
straight_anions = [4383, 4386, 4392, 4552]
straight_DG = load_node(4487)
hugged_DG = load_node(4490) # DG reference

#### BoPh ####
BoPh_hugged_pairs = [4764, 4594, 4600, 4738]
BoPh_hugged_anions = [4606, 4612, 4616, 4745]

print(load_node(2898).inputs.parameters.get_dict())
print(load_node(2898).outputs.remote_folder.get_remote_path())
print(load_node(2898).outputs.output_parameters.get_dict().keys())
BPh_reference_energy = load_node(2898).outputs.output_parameters.get_dict()["scfenergies"][-1] # no explicit solvent
BoPh_reference_energy = load_node(2892).outputs.output_parameters.get_dict()["scfenergies"][-1]
NaBPh_reference_energy = load_node(2876).outputs.output_parameters.get_dict()["scfenergies"][-1]
NaBoPh_reference_energy = load_node(2870).outputs.output_parameters.get_dict()["scfenergies"][-1]
Na_reference = load_node(2667).outputs.output_parameters.get_dict()["scfenergies"][-1] # has explicit DG
DG_reference = hugged_DG.outputs.output_parameters.get_dict()["scfenergies"][-1]

failed_calcs = [4306, 4333, 4395, 4603, 4621]

Ea_BoPh =  NaBoPh_reference_energy - (BoPh_reference_energy + Na_reference - DG_reference)
Ea_BPh = NaBPh_reference_energy - (BPh_reference_energy + Na_reference - DG_reference)

print("0DG BoPh binding energy: ", Ea_BoPh)
print("0DG BPh binding energy: ", Ea_BPh)

NaBoPh_reference_energy = load_node(4764).outputs.output_parameters.get_dict()["scfenergies"][-1] # 1 explicit diglyme
NaBPh_reference_energy = load_node(4303).outputs.output_parameters.get_dict()["scfenergies"][-1]

Ea_BoPh =  NaBoPh_reference_energy - (BoPh_reference_energy + Na_reference)
Ea_BPh = NaBPh_reference_energy - (BPh_reference_energy + Na_reference)
print("1DG BoPh binding energy: ", Ea_BoPh)
print("1DG BPh binding energy: ", Ea_BPh)

def calculate_association_energies(anion_pk_list, pair_pk_list):
    E_list = np.zeros(len(anion_pk_list)+1)
    for i, pk in enumerate(anion_pk_list):
        if pk in failed_calcs or pair_pk_list[i] in failed_calcs:
            continue
        num_solvent = i + 2 # +1 for index at 0 and +1 for explicit DG in Na reference
        anion_node = load_node(pk)
        outputs = anion_node.outputs.output_parameters.get_dict()
        E_anion = outputs["scfenergies"][-1]
        pair_node = load_node(pair_pk_list[i])
        outputs = pair_node.outputs.output_parameters.get_dict()
        E_pair = outputs["scfenergies"][-1]
        E_cation = Na_reference
        E_DG = DG_reference
        E_assoc = E_pair - (E_anion + E_cation - E_DG)
        E_list[i+1] = E_assoc
    return E_list

BPh_list = calculate_association_energies(hugged_anions, hugged_pairs)
print(BPh_list)
BoPh_list = calculate_association_energies(BoPh_hugged_anions, BoPh_hugged_pairs)
print(BoPh_list)