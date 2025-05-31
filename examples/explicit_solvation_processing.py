from aiida.orm import load_node
from aiida import load_profile
import numpy as np

load_profile("cooper")

#### BPh ####
hugged_pairs = [4303, 4832, 4310, 4315]
hugged_anions = [4320, 4324, 4327, 4333]
straight_anions = [4383, 4386, 4392, 4552]
straight_DG = load_node(4487)
hugged_DG = load_node(4490)

#### BoPh ####
BoPh_hugged_pairs = [4764, 4594, 4600, 4603]
BoPh_hugged_anions = [4606, 4612, 4616, 4621]


BPh_reference_energy = load_node(2898).outputs.output_parameters.get_dict()["scfenergies"][-1]
BoPh_reference_energy = load_node(2892).outputs.output_parameters.get_dict()["scfenergies"][-1]
NaBPh_reference_energy = load_node(2876).outputs.output_parameters.get_dict()["scfenergies"][-1]
NaBoPh_reference_energy = load_node(2870).outputs.output_parameters.get_dict()["scfenergies"][-1]

failed_calcs = [4306, 4333, 4395, 4603, 4621]

DG_arr = np.array([hugged_DG.outputs.output_parameters.get_dict()["scfenergies"][-1],
    straight_DG.outputs.output_parameters.get_dict()["scfenergies"][-1]])

min_idx = np.argmin(DG_arr)
# print(min_idx, "min index")
E_DG = DG_arr[min_idx]

print("BPh pairs")
for i, pk in enumerate(hugged_pairs):
    # i is number of DG molecules
    if pk in failed_calcs:
        continue
    print(i, pk)
    calc_node = load_node(pk)
    # print(list(calc_node.outputs))
    outputs = calc_node.outputs.output_parameters.get_dict()
    E_complex = outputs["scfenergies"][-1]
    E_ref = (i+1) * E_DG
    E = E_complex - E_ref
    E_relative = E - NaBPh_reference_energy
    print(f"{E_relative:.3}")

print("BPh anions")
for i, pk in enumerate(hugged_anions):
    # i is number of DG molecules
    if pk in failed_calcs:
        continue
    print(i, pk)
    calc_node = load_node(pk)
    # print(list(calc_node.outputs))
    outputs = calc_node.outputs.output_parameters.get_dict()
    E_complex = outputs["scfenergies"][-1]
    E_ref = (i+1) * E_DG
    E = E_complex - E_ref
    if i == 0:
        E0 = E
    E_relative = E - BPh_reference_energy
    print(f"{E_relative:.3}")

# for i, pk in enumerate(straight_anions):
#     # i is number of DG molecules
#     if pk in failed_calcs:
#         continue
#     print(i, pk)
#     calc_node = load_node(pk)
#     # print(list(calc_node.outputs))
#     outputs = calc_node.outputs.output_parameters.get_dict()
#     E_complex = outputs["scfenergies"][-1]
#     E_ref = (i+1) * E_DG
#     E = E_complex - E_ref
#     if i == 0:
#         E0 = E
#     E_relative = E - BPh_reference_energy
#     print(f"{E_relative:.10}")

print("BoPh pairs")
for i, pk in enumerate(BoPh_hugged_pairs):
    # i is number of DG molecules
    if pk in failed_calcs:
        continue
    print(i, pk)
    calc_node = load_node(pk)
    # print(list(calc_node.outputs))
    outputs = calc_node.outputs.output_parameters.get_dict()
    E_complex = outputs["scfenergies"][-1]
    E_ref = (i+1) * E_DG
    E = E_complex - E_ref
    E_relative = E - NaBoPh_reference_energy
    print(f"{E_relative:.3}")

print("BoPh anions")
for i, pk in enumerate(BoPh_hugged_anions):
    # i is number of DG molecules
    if pk in failed_calcs:
        continue
    print(i, pk)
    calc_node = load_node(pk)
    # print(list(calc_node.outputs))
    outputs = calc_node.outputs.output_parameters.get_dict()
    E_complex = outputs["scfenergies"][-1]
    E_ref = (i+1) * E_DG
    E = E_complex - E_ref
    if i == 0:
        E0 = E
    E_relative = E - BoPh_reference_energy
    print(f"{E_relative:.3}")