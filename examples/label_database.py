# Script used to label all relevant publication calculations
# Going to take all the recent AssociationWorkChain2s (4977+) plus the recent 
# HomoLumoWorkChains (4910+) as well as all successful RamanSpectrumWorkChains.

from multiprocessing import Pool

from aiida.orm import (
    QueryBuilder, 
    CalcJobNode, 
    WorkChainNode, 
    Node, 
    load_node,
    StructureData,
    InstalledCode,
    Dict, 
    load_group
) 
from aiida.tools.visualization.graph import Graph
from WorkChains.AssociationWorkChain2 import AssociationWorkChain2
from aiida import load_profile

load_profile()

class NodeSet:
    """
    A set mock class that matches elements based on their 'pk' attribute when performing set operations.
    Elements must have a 'pk' attribute to be added to the set.

    Fully wripped this from Claude so may not work.
    """


    def __init__(self, iterable=None):
        self._items = []  # Using a list for internal storage
        if iterable:
            for item in iterable:
                self.add(item)
    
    def _find_index(self, item):
        """Find index of item with matching pk."""
        if not hasattr(item, 'pk'):
            raise AttributeError(f"Item {item} has no pk attribute")
        
        for idx, existing in enumerate(self._items):
            if existing.pk == item.pk:
                return idx
        return -1

    def add(self, item):
        """Add item if its pk isn't already present."""
        if not hasattr(item, 'pk'):
            raise AttributeError(f"Item {item} has no pk attribute")
            
        if self._find_index(item) == -1:
            self._items.append(item)
    
    def remove(self, item):
        """Remove item with matching pk."""
        idx = self._find_index(item)
        if idx == -1:
            raise KeyError(item)
        self._items.pop(idx)
    
    def discard(self, item):
        """Discard item with matching pk if found."""
        try:
            self.remove(item)
        except KeyError:
            pass
    
    def __contains__(self, item):
        """Check if item with matching pk exists."""
        return self._find_index(item) != -1
    
    def __iter__(self):
        """Iterate over items."""
        return iter(self._items)
    
    def __len__(self):
        """Return number of items."""
        return len(self._items)
    
    def union(self, other):
        """Return new NodeSet with items from both sets."""
        result = NodeSet(self)
        for item in other:
            result.add(item)
        return result
    
    def intersection(self, other):
        """Return new NodeSet with items common to both sets."""
        result = NodeSet()
        for item in self:
            if item in other:
                result.add(item)
        return result
    
    def difference(self, other):
        """Return new NodeSet with items in this set but not in other."""
        result = NodeSet()
        for item in self:
            if item not in other:
                result.add(item)
        return result
    
    def __repr__(self):
        return f"NodeSet({self._items})"

def recursive_find(pk:int, connected_nodes: list[Node]) -> list[Node]:
    """Find all connected nodes to specified pk recursively."""
    
    base_node = load_node(pk)
    incoming_nodes = []
    outgoing_nodes = []
    if len(base_node.base.links.get_incoming().all_nodes()) != 0:
        for inode in base_node.base.links.get_incoming():
            if inode.node not in connected_nodes:
                connected_nodes = recursive_find(inode.node.pk, connected_nodes)
            incoming_nodes.append(inode.node)
    if check_get_outputs(base_node):
        for inode in base_node.base.links.get_outgoing():
            if inode.node not in connected_nodes:
                connected_nodes = recursive_find(inode.node.pk, connected_nodes)
            outgoing_nodes.append(inode.node)

    connected_nodes = NodeSet(connected_nodes).union(NodeSet(incoming_nodes).union(NodeSet(outgoing_nodes)))

    return list(connected_nodes)

def check_get_outputs(node: Node):
    """Check whether to get input connections.
    
    This function encodes rules about which nodes are considered terminal
    so that more connections are not extracted to completely separate 
    workchains. For example, output connections should not be pulled for Code
    nodes because it would pull nearly everything in the database.

    Includes everything regardless of convergence.

    Don't want outputs of any StructureData or Dicts because they might be 
    connected to other workchains and their outputs will be reached either way
    by traversing up from the other nodes in the graph
    """
    unallowed_types = [
        InstalledCode,
        Dict,
        StructureData
    ]
    # check types
    if type(node) in unallowed_types:
        return False
    # check if there are no outputs
    if len(node.base.links.get_outgoing().all_nodes()) == 0:
        return False
    # otherwise return True
    return True
    


###### AssociationWorkChains ######
def get_association_nodes():
    qb = QueryBuilder()
    qb.append(
        WorkChainNode,
        filters={
            "attributes.process_label": "AssociationWorkChain2",
            "attributes.process_state": "finished",
            "id": {">=":4977},
        }
    )

    data = qb.all()
    return data
    # for d in data:
    #     n = d[0]
    #     all_nodes = recursive_find(n.pk, [])
    #     print(list(map(lambda d: d.pk, all_nodes)))
    #     print(len(all_nodes))

###### HomoLumoWorkChains
def get_homolumo_nodes():
    qb = QueryBuilder()
    qb.append(
        WorkChainNode,
        filters={
            "attributes.process_label": "HomoLumoWorkChain",
            "attributes.process_state": "finished",
        }
    )
    data = qb.iterall()
    for d in data:
        n = d[0]
        all_nodes = recursive_find(n, [])
        print(list(map(lambda d: d.pk, all_nodes)))
        print(len(all_nodes))

def get_ramanspectrum_nodes():
    qb = QueryBuilder()
    qb.append(
        WorkChainNode,
        filters={
            "attributes.process_label": "RamanSpectrumWorkChain",
            "attributes.process_state": "finished",
        }
    )
    data = qb.iterall()
    for d in data:
        n = d[0]
        all_nodes = recursive_find(n, [])
        print(list(map(lambda d: d.pk, all_nodes)))
        print(len(all_nodes))

def get_search_nodes():
    nodes = []
    qb = QueryBuilder()
    qb.append(
        WorkChainNode,
        filters={
            "attributes.process_label": "RamanSpectrumWorkChain",
            "attributes.process_state": "finished",
        }
    )
    nodes.extend(qb.all())
    qb = QueryBuilder()
    qb.append(
        WorkChainNode,
        filters={
            "attributes.process_label": "HomoLumoWorkChain",
            "attributes.process_state": "finished",
        }
    )
    nodes.extend(qb.all())
    qb = QueryBuilder()
    qb.append(
        WorkChainNode,
        filters={
            "attributes.process_label": "AssociationWorkChain2",
            "attributes.process_state": "finished",
            "id": {">=":4977},
        }
    )
    nodes.extend(qb.all())
    return [n[0].pk for n in nodes]


def find_connected_nodes(pk:int):
    """Use AiiDA's recursive node finding methods"""
    graph = Graph(node_id_type="pk")
    root_node = load_node(pk)
    graph.recurse_ancestors(
        root_node
    )
    graph.recurse_descendants(
        root_node
    )
    return graph.nodes

def label_nodes(pks: list[int]):
    for pk in pks:
        node = load_node(pk)
        node.group = "NaPub"


if __name__ == "__main__":
    search_nodes = get_search_nodes()
    nodes = set()
    nodes = find_connected_nodes(search_nodes[-1])
    for node in search_nodes:
        nodes = nodes.union(find_connected_nodes(node))
    print(nodes)
    print(len(nodes))
    
    real_nodes = [load_node(n) for n in nodes]
    group = load_group("NaPub")
    group.add_nodes(real_nodes)

    # print(search_nodes)
    # tasks = [(node, []) for node in search_nodes]
    # max_procs = 4
    # with Pool(processes=max_procs) as p:
    #     results = p.map(recursive_find, tasks)
    # results = get_association_nodes()
    # print(results)
    
    # n1 = load_node("4957da81")
    # n2 = load_node("208ca629")
    # ns1 = NodeSet([n1])
    # ns2 = NodeSet([n2, n1])
    # ns3 = ns1.union(ns2)

    # print(ns3)

    


