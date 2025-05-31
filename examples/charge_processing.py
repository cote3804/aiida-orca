from aiida.orm import load_node
from aiida import load_profile

load_profile()

calc_pks = {
    "NaBhfip": 2858,
    "NaBoPh": 2861,
    "NaBPh": 2864,
    "NaPF6": 2867,
}

calc_pks = {
    "NaBhfip": 2873,
    "NaBoPh": 2870,
    "NaBPh": 2876,
    "NaPF6": 2879,
}

def get_indices(num_list, target_num):
    indices = []
    for i, num in enumerate(num_list):
        if num == target_num:
            indices.append(i)
        else:
            continue
    return indices

for mol, pk in calc_pks.items():
    print(mol)
    n = load_node(pk)
    results = n.outputs.output_parameters.get_dict()
    mull = results["atomcharges"]["mulliken"]
    low = results["atomcharges"]["lowdin"]
    na_idx = results["atomnos"].index(11)
    atomnos = results["atomnos"]
    if 8 in results["atomnos"]:
        o_idx = get_indices(results["atomnos"], 8)
        o_mull = [mull[x] for x in o_idx]
        o_low = [low[x] for x in o_idx]
        print(sum(o_mull))
        print(sum(o_low))
    if mol in ["NaBoPh", "NaBPh"]:
        c_idx = get_indices(atomnos, 6)
        h_idx = get_indices(atomnos, 1)
        c_idx.extend(h_idx)
        ch_mull = [mull[x] for x in c_idx]
        ch_low = [low[x] for x in c_idx]
        ch_mull_sum = sum(ch_mull)
        ch_low_sum = sum(ch_low)
        print(f"{ch_mull_sum:.3}", f"{ch_low_sum:.3}")
