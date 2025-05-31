# Submitting ionic relaxations starting from all of the structures generated
# by using RDKit to convert SMILES to XYZ files.

from pathlib import Path
from ase.io import read, write
from RelaxWorkChain import RelaxWorkChain
from ase.visualize import view
from aiida.orm import Dict, load_code, StructureData, QueryBuilder, WorkChainNode
from aiida.engine import submit
from aiida import load_profile
import datetime
import os

load_profile()
structures_path = Path("/home/coopy/onedrive/Research/my_scripts/representation_learning/CycleLearn/my_data/molecule_structures_no_cation")

def submit_calcs():
    atoms_list = []
    names_list = []
    allowed_files = [
        # "LiPF6", 
        "LiAsF6"
        ] # run for only a few molecules
    for file in structures_path.glob("*.xyz"):
        if file.name.split(".")[0] not in allowed_files:
            continue
        atoms = read(file)
        view(atoms)
        atoms_list.append(atoms)
        names_list.append(file.name.split(".")[0])

    ## Add Bhfip to the list
    # bhfip = read(Path("/home/coopy/onedrive/Research/Batteries/molecules/NaBhfip/NaBhfip.xyz"))
    # species = [atom.symbol for atom in bhfip]
    # Na_index = species.index("Na")
    # del bhfip[Na_index]
    # atoms_list.append(bhfip)
    # names_list.append("bhfip")
    for atoms, name in zip(atoms_list, names_list):
        if "Li" in name or name == "bhfip":
            charge = -1
            multiplicity = 1
        else:
            charge = 0
            multiplicity = 1
        
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

        parameters = Dict(parameters)
        code = load_code(2634) # orca on alpine_test followed by iconv with infiniband node constraint
        structure = StructureData(ase=atoms)
        structure.label = name
        structure.store()
        submit(RelaxWorkChain, structure=structure, parameters=parameters, code=code)

def query_results():
    qb = QueryBuilder()
    date = datetime.datetime(2025, 4, 16, 23, 0, 0)
    qb.append(
        WorkChainNode, 
        filters={
        "attributes.exit_status": 0,
        "ctime": {">=": date},
        "attributes.process_label": "RelaxWorkChain",
        },
        tag="workchain"
)
    qb.append(StructureData, 
            with_outgoing="workchain",
            # project=["label"]
            tag="pre-relaxed"
            )
    pre_relaxed_results = qb.all()
    qb = QueryBuilder()
    qb.append(
        WorkChainNode, 
        filters={
        "attributes.exit_status": 0,
        "ctime": {">=": date},
        "attributes.process_label": "RelaxWorkChain",
        },
        tag="workchain"
    )
    qb.append(
        StructureData,
        with_incoming="workchain",
        tag="relaxed"
    )
    relaxed_results = qb.all()

    dft_structures_path = Path("/home/coopy/onedrive/Research/my_scripts/representation_learning/CycleLearn/my_data/dft_structures")
    if not os.path.exists(dft_structures_path):
        dft_structures_path.mkdir(parents=True)
    for pre_res, rel_res in zip(pre_relaxed_results, relaxed_results):
        print(pre_res[0].get_ase())
        print(rel_res[0].get_ase())
        label = pre_res[0].label
        pre_atoms = pre_res[0].get_ase()
        rel_atoms = rel_res[0].get_ase()
        fname = dft_structures_path / Path(f"{label}.xyz")
        write(fname, rel_atoms)

# def query_failures():


if __name__ == "__main__":
    # submit_calcs()
    query_results()
