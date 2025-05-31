from ComplexOptimizeWorkChain import ComplexOptimizeWorkChain
from aiida.orm import load_node, Dict, Str
from aiida.engine import submit
from aiida import load_profile

Na_complex_pks = [472, 462, 475, 440]
diglyme_complex_pks = [1318, 1321, 1324, 1327]
Na_diglyme_complex_pks = [1370]
Na_PF6_diglyme_pks = [3457]
bph_diglyme_pks = [5123]
boph_diglyme_pks = [5119]
accounts = ["ucb471_asc1", "ucb487_asc1"]

num_procs = 64
parameters = {
            # Always put PAL command at end of input_keywords list
            "charge": 0, "multiplicity": 1, 
            # 'input_keywords': ["wB97X-D4", "def2-TZVP", "OPT", "NumFreq", f"PAL{num_procs}"],
            'input_keywords': ["r2SCAN-3c", "OPT"],
            'input_blocks': 
            {
                "cpcm": {
                    "epsilon": 7.23, "refrac": 1.4097
                },
                "pal":{
                    "nprocs": num_procs
                    },
                "geom": {
                    "MaxIter": 500
                },
            }
        }

def submit_calc(code, struct_dict, parameters, account):
    parameters = Dict(parameters)
    submit(ComplexOptimizeWorkChain, {"code": code, "structure_dict": struct_dict, "parameters": parameters, "account": account})


def main():
    load_profile()
    # for complex_pk in Na_complex_pks:
    #     struct_dict = load_node(complex_pk)
    #     code = load_node(92)
    #     submit_calc(code, struct_dict, parameters)
    # for complex_pk in diglyme_complex_pks:
    #     # solvent (diglyme) plus charged anion
    #     struct_dict = load_node(complex_pk).outputs.result
    #     sub_params = parameters.copy()
    #     sub_params["charge"] = -1
    #     sub_params["multiplicity"] = 1
    #     # struct_dict = load_node(complex_pk)
    #     account = Str(accounts[1])
    #     code = load_node(92)
    #     submit_calc(code, struct_dict, sub_params)
    for complex_pk in boph_diglyme_pks:
        struct_dict = load_node(complex_pk).outputs.result
        # solvent (diglyme) plus charged cation (Na)
        sub_params = parameters.copy()
        sub_params["charge"] = 0
        sub_params["multiplicity"] = 1
        code = load_node(2634) # orca on alpine_test followed by iconv with infiniband node constraint
        account = Str(accounts[-1])
        submit_calc(code, struct_dict, sub_params, account)


if __name__ == "__main__":
    main()