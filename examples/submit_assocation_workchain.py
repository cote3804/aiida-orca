from aiida.orm import load_node, StructureData, Dict
from aiida.engine import submit, run
from aiida import load_profile
from ase.io import read
import os
from AssociationWorkChain import AssociationWorkChain

mol_path = "/home/coopy/onedrive/Research/Batteries/molecules"
molecules = ["NaPF6", "NaBoPh", "NaBhfip", "NaBPh"]
# molecules = ["NaPF6"] # testing



def main():
    load_profile()
    num_procs = 64

    resources = {
        "num_machines": 1, "num_mpiprocs_per_machine": num_procs
    }

    parameters = {
        "charge": 0, "multiplicity": 1, 'input_keywords': ["wB97X-D4", "def2-TZVP", "OPT", "NumFreq"],
        'input_blocks': {
            "cpcm": {
                "epsilon": 7.23, "refrac": 1.4097
                },
            "pal":{
                    "nprocs": num_procs
                    },
                #  "method RunTyp Opt": {"end": ""},
            "geom": {
                "MaxIter": 500
                },
                # "freq": {"CentralDiff": "true"}
        }
    }
    code = load_node(1139)
    parameters = Dict(parameters)
    resources = Dict(resources)
    for mol in molecules:
        try:
            atoms = read(os.path.join(mol_path, mol, f"{mol}.xyz"))
        except:
            atoms = read(os.path.join(mol_path, mol, f"{mol}.pdb"))
        structure = StructureData(ase=atoms) 
        # structure = load_node("b94662c8") # optimized PF6 structure
        submit(AssociationWorkChain, {"code": code, "structure": structure, "resources": resources, "parameters": parameters})

if __name__ == "__main__":
    main()