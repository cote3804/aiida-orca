from aiida.orm import load_node
from aiida import load_profile
from ase.visualize import view

load_profile()

wk_node = load_node(3464)
out = wk_node.get_outgoing().all_nodes()
for i, n in enumerate(out):
    if n.is_finished_ok:
        in_struct = n.get_incoming().all_nodes()[-3]
        print(in_struct)
        view(in_struct.get_ase())
        struct = n.outputs.relaxed_structure
        # view(struct.get_ase())