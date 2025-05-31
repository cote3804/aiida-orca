from aiida.orm import load_node, StructureData, Dict
from aiida import load_profile
from aiida.engine import submit
from AssociationWorkChain2 import AssociationWorkChain2

relaxed_structures = {
    # "PF6": {
    #     "pair": 2926,
    #     "anion": 2910,
    #     "cation": 2681
    # },
    "BoPh": {
        # "pair": 2956, # old pair calc that gets wrong binding energy ordering
        "pair": 4791,
        "anion": 2923,
        "cation": 2681,
    },
    "BPh": {
        # "pair": 2936, # old pair calc that gets wrong binding energy ordering
        "pair": 4429,
        "anion": 2917,
        "cation": 2681
    },
    # "Bhfip": {
    #     "pair": 2950,
    #     "anion": 2920,
    #     "cation": 2681
    # },
}

def main():
    nprocs = 64
    code = load_node(2634) # orca on alpine_test followed by iconv with infiniband node constraint
    parameters = {
            "charge": 0, "multiplicity": 1, 
            # 'input_keywords': ["wB97X-D4", "def2-QZVP", "def2/J", "OPT", "NumFreq", "TightSCF", "RIJCOSX"],
            'input_keywords': ["r2SCAN-3c", "OPT", "NumFreq", "TightSCF"],
            'input_blocks': {
                "cpcm": {
                    "epsilon": 7.23, "refrac": 1.4097
                    },
                "pal":{
                        "nprocs": nprocs
                        },
                "geom": {
                    "MaxIter": 500
                    },
            }
        }
    parameters = Dict(parameters)
    for molecule in relaxed_structures.keys():
        pair_struct = load_node(relaxed_structures[molecule]["pair"])
        anion_struct = load_node(relaxed_structures[molecule]["anion"])
        cation_struct = load_node(relaxed_structures[molecule]["cation"])
        inputs = {
            "pair_structure": pair_struct,
            "anion_structure": anion_struct,
            "cation_structure": cation_struct,
            "parameters": parameters,
            "code": code,
        }
        submit(AssociationWorkChain2, **inputs)
        

if __name__ == "__main__":
    load_profile()
    main()