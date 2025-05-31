from aiida.orm import QueryBuilder, load_node, StructureData, Dict, load_code
from aiida import load_profile
from aiida.engine import submit
from RelaxWorkChain import RelaxWorkChain
from ase.visualize import view
from io import StringIO
from ase.io import read

load_profile("cooper")

def resubmit_timeout_structure(node):
    # print(node.exit_status)
    wc_node = node.called[0]
    if wc_node.exit_status == None:
        return None
    else:
        calc_node = wc_node.called[0]
    xyz_string = calc_node.outputs.retrieved.get_object_content("aiida.xyz")
    with StringIO(xyz_string) as f:
        optimized_atoms = read(f, format="xyz")
    structure = StructureData(ase=optimized_atoms)
    parameters = node.get_incoming(node_class=Dict).all_nodes()[0]
    code = load_code(2634) # orca on alpine_test followed by iconv with infiniband node constraint
    inputs = {
        "structure": structure,
        "code": code,
        "parameters": parameters
    }
    submit(RelaxWorkChain, **inputs)

def resubmit_relaxation(node):
    structure = node.get_incoming(node_class=StructureData).all_nodes()[0]
    parameters = node.get_incoming(node_class=Dict).all_nodes()[0]
    code = load_code(2634) # orca on alpine_test followed by iconv with infiniband node constraint
    inputs = {
        "structure": structure,
        "code": code,
        "parameters": parameters
    }
    submit(RelaxWorkChain, **inputs)

qb = QueryBuilder()
finished_filter = {"or":[
    {"attributes.process_state": "finished"},
    {"attributes.exit_status": {"in": [11]}},
    ]
}
qb.append(
    [RelaxWorkChain],
    filters=finished_filter
    )


# query = qb.all()
# for wc in query:
#     print(wc[0].pk, wc[0].exit_status)
#     # if wc[0].exit_status == 11: # will submit unconverged jobs
#     #     resubmit_timeout_structure(wc[0])
#     atoms = wc[0].inputs.structure.get_ase()
#     n_atoms = len(atoms)
#     pk = wc[0].pk

# resubmit_timeout_structure(load_node(4542))
resubmit_relaxation(load_node(4542))