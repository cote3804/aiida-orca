from aiida.orm import load_node, StructureData, Dict
from aiida.engine import submit, run
from aiida import load_profile
from ase.io import read
from aiida_helpers import get_mols_path, remove_cation
import os
from AssociationWorkChain import AssociationWorkChain
from HomoLumoWorkChain import HomoLumoWorkChain

mol_path = "/home/coopy/onedrive/Research/Batteries/molecules"
molecules = ["NaPF6", "NaBoPh", "NaBhfip", "NaBPh"]
molecules = ["NaPF6"] # testing
molecules = ["H2O"] # testing
molecules = {"NaPF6":939, "NaBoPh":2117, "NaBhfip":2129, "NaBPh":1756} # optimized molecules

# molecules = {"NaPF6":939} # optimized PF6 for testing




def submit_from_calc_pk(pk_dict):
    load_profile()
    num_procs = 64

    resources = {
        "num_machines": 1, "num_mpiprocs_per_machine": num_procs
    }

    parameters = {
        "charge": 0, "multiplicity": 1, 'input_keywords': ["wB97X-D4", "def2-TZVP", "OPT", "LARGEPRINT"],
        # "charge": 0, "multiplicity": 1, 'input_keywords': ["PBEh-3c", "LARGEPRINT"], # fast calc testing
        'input_blocks': 
        {"cpcm": {
            "epsilon": 7.23, "refrac": 1.4097
            },
        "pal":{
                    "nprocs": num_procs
                    },
            #  "method RunTyp Opt": {"end": ""},
        # "geom": {
        #     "MaxIter": 500
        #     },
            # "freq": {"CentralDiff": "true"}
            }
    }
    code = load_node(2634) # orca on alpine_test followed by iconv with infiniband node constraint
    resources = Dict(resources)
    parameters = Dict(parameters)
    for mol, pk in pk_dict.items():
        relaxed_structure = load_node(pk).outputs.relaxed_structure

        # structure = load_node(pk)
        # atoms = structure.get_ase()
        # atoms = remove_cation(atoms)
        # structure = StructureData(ase=atoms)
        relaxed_structure.label = mol
        # structure = struct
        submit(HomoLumoWorkChain, {"code": code, "structure": relaxed_structure, "resources": resources, "parameters": parameters})

def submit_from_path(paths):
    pass


if __name__ == "__main__":
    mols_root = get_mols_path()
    mol_paths = ["NaBPh/NaBPh.xyz", "NaBoPh/NaBoPh.xyz", "BPh/BPh.xyz", "BoPh/BoPh.xyz"]
    fluorinated_paths = [os.path.join(mols_root, "Fluorinated", i) for i in mol_paths]
    fluorinated_pks = {
        "NaBPh": 4843,
        "NaBoPh": 4846,
        # "BPH": 4852,
        # "BoPh": 4855,
        }
    submit_from_calc_pk(fluorinated_pks)
    # submit_from_path()
