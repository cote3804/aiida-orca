from aiida.orm import QueryBuilder, Node
from aiida import load_profile
from HomoLumoWorkChain import HomoLumoWorkChain
from OrcaWorkChain import OrcaWorkChain
from ase import Atoms
from ase.visualize import view

load_profile()

qb = QueryBuilder()
# filters = {
#     'attributes.process_state': 'finished',
# }
filters={                       # Specifying the filters:
        'attributes.process_state':{'==':'finished'}, 
        # 'attributes.pk':{'>':0}    
    }
# filters = {HomoLumoWorkChain.inputs.}
qb.append(
    OrcaWorkChain,
    filters=filters
)

results = qb.all()
# print(results)
for res in results:
    inp = res[0].inputs
    params = inp.parameters.get_dict()
    if 'wB97X-D4' in params['input_keywords'] and 'LARGEPRINT' in params['input_keywords']:
        atoms = res[0].inputs.structure.get_ase()
        species = atoms.get_chemical_symbols()
        O_count = species.count("O")
        B_count = species.count("B")
        if O_count == 3 and B_count == 1:
            print(res[0].pk)
            pass
    out = res[0].outputs
    # print(out)
    # print(list(out), inp.structure.get_ase())
    # if "remote_folder" in list(out):
        # print(inp.structure.get_ase())
        # print(out.remote_folder.get_remote_path())