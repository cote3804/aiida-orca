from aiida.orm import load_node, load_code, Dict, StructureData
from aiida import load_profile
from aiida.engine import submit
from aiida_helpers import get_mols_path
from os.path import join as opj
from ase.io import read
from RelaxWorkChain import RelaxWorkChain

def get_structures(paths, root=None):
    if root == None:
        root = get_mols_path()
    atoms_list = []
    for path in paths:
        atoms = read(opj(root, path))
        atoms_list.append(atoms)
    return atoms_list

def get_parameters(charge=0, multiplicity=1):
    parameters = {
        "charge": charge, "multiplicity": multiplicity, 'input_keywords': ["r2SCAN-3c", "OPT"],
        'input_blocks': 
        {"cpcm": {
            "epsilon": 7.23, "refrac": 1.4097
            },
        "pal": {
            "nprocs": 64
        },
        }
    }
    return parameters

def submit_relaxations(atoms_list, salt_pair=True):
    for atoms in atoms_list:
        if salt_pair:
            if "Na" in atoms.get_chemical_symbols():
                charge = 0
            else:
                charge = -1
        else:
            charge = 0
        parameters = Dict(get_parameters(charge=charge))
        code = load_code(2634) # orca on alpine_test followed by iconv with infiniband node constraint
        structure = StructureData(ase=atoms)
        submit(RelaxWorkChain, structure=structure, parameters=parameters, code=code)


if __name__ == "__main__":
    load_profile()
    hugged_paths = [
        "NaBPhDG1/hugged.xyz", "NaBPhDG2/hugged.xyz", "NaBPhDG3/hugged.xyz","NaBPhDG4/hugged.xyz",
        "BPhDG1/hugged.xyz", "BPhDG2/hugged.xyz", "BPhDG3/hugged.xyz", "BPhDG4/hugged.xyz"
    ] 
    straight_paths = [
        "BPhDG1/straight.xyz", "BPhDG2/straight.xyz", "BPhDG3/straight.xyz", "BPhDG4/straight.xyz"
    ]
    DG_paths = [
        "diglyme/diglyme.xyz", "diglyme/diglyme_hug.xyz"
    ]
    BoPh_paths = [
        "NaBoPhDG1/hugged.xyz", "NaBoPhDG2/hugged.xyz", "NaBoPhDG3/hugged.xyz","NaBoPhDG4/hugged.xyz",
        "BoPhDG1/hugged.xyz", "BoPhDG2/hugged.xyz", "BoPhDG3/hugged.xyz", "BoPhDG4/hugged.xyz"
    ]
    correct_chelating_paths = [ # check 7/11 in Evan Notes
        "NaBoPhDG1/hugged.xyz", "NaBPhDG2/hugged.xyz"
    ]
    correct_chelating_paths = [ # NaBoPhDG1 finished correctly from these calcs ^ but BPhDG2 did not
        "NaBPhDG2/hugged.xyz"
    ]
    fluorinated_molecules = [
        "NaBPh/NaBPh.xyz", "NaBoPh/NaBoPh.xyz", "BPh/BPh.xyz", "BoPh/BoPh.xyz"
        ]
    root = opj(get_mols_path(), "Fluorinated") 
    atoms_list = get_structures(fluorinated_molecules, root=root)
    submit_relaxations(atoms_list, salt_pair=True)