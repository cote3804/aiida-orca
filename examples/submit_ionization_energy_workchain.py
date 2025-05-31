from aiida.orm import load_node, StructureData, Dict, load_code
from aiida.engine import submit, run
from aiida import load_profile
from ase.io import read
from ase import Atoms
import os
from IonizationEnergyWorkChain import ParallelIonizationEnergy
from IonizationEnergyWorkChain2 import IonizationEnergy
from aiida_helpers import get_mols_path

mol_path = "/home/coopy/onedrive/Research/Batteries/molecules"
mol_path = get_mols_path()
molecules = ["NaPF6", "NaBoPh", "NaBhfip", "NaBPh"]
# molecules = ["NaPF6"] # testing


def submit_anions(anions:list[Atoms]):
    code = load_code(2634) # orca on alpine_test followed by iconv with infiniband node constraint
    for atoms in anions:
        structure = StructureData(ase=atoms)
        submit(IonizationEnergy, {"code": code, "structure": structure})



def main():
    load_profile()
    num_procs = 8
    parameters = {
        "charge": 0, "multiplicity": 1, 'input_keywords': ["wB97X-D4", "def2-TZVP", "OPT", f"PAL8"], # always place pal command at end
        'extra_input_keywords': ["FREQ"],
        'input_blocks': {"cpcm": {"epsilon": 7.23, "refrac": 1.4097, "cds_cpcm": 2},
                        #  "method RunTyp Opt": {"end": ""},
                        "geom": {"MaxIter": 200}}
        } 
    parameters = Dict(parameters)
    resources = {
    "num_machines": 1, "num_mpiprocs_per_machine": num_procs, 
    }
    resources = Dict(resources)
    code = load_node("92")
    for mol in molecules:
        try:
            atoms = read(os.path.join(mol_path, mol, f"{mol}.xyz"))
        except:
            atoms = read(os.path.join(mol_path, mol, f"{mol}.pdb"))
        structure = StructureData(ase=atoms)
        submit(ParallelIonizationEnergy, {"code": code, "structure": structure, "parameters": parameters, "resources":resources})

if __name__ == "__main__":
    main()