from aiida.orm import load_node
from aiida import load_profile

load_profile("cooper")

fluorinated_pks = {
    "NaBPh": 4932,
    "NaBoPh": 4935,
    "BPh": 4917,
    "BoPh": 4920,
}

regular_pks = {
    "NaPF6": 3823,
    "NaBoPh": 3826,
    "NaBhfip": 3830,
    # "NaBPh": 3834,
    "PF6": 3938,
    "BoPh": 3942,
    # "Bhfip": 3947,
    # "BPh": 3952
}

for mol, pk in fluorinated_pks.items():
    calc = load_node(pk)
    lumo = calc.outputs.lumo.value
    homo = calc.outputs.homo.value
    # print(mol, f"LUMO: {lumo}", f"HOMO: {homo}")

for mol, pk in regular_pks.items():
    calc = load_node(pk)
    lumo = calc.outputs.lumo.value
    homo = calc.outputs.homo.value
    print(mol, f"HOMO: {homo}",  f"LUMO: {lumo}")