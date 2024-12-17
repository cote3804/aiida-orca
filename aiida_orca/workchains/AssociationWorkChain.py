from aiida.orm import StructureData
from ase.io import read
import os
from aiida.orm import StructureData, AbstractCode, Float, Dict, load_node
from aiida.engine import run, submit, WorkChain, calcfunction, ToContext, workfunction
from aiida import load_profile
from ase import Atoms, Atom
from ase.units import Hartree, kJ, J
from scipy.constants import R, Avogadro
import numpy as np
from aiida_orca.calculations import OrcaCalculation

mol_path = "/home/coopy/onedrive/Research/Batteries/molecules"
molecules = ["NaPF6", "NaBoPh", "NaBhfip", "NaBPh"]

num_procs = 8

parameters = {
    "charge": 0, "multiplicity": 1, 'input_keywords': ["wB97X-D4", "def2-TZVP", "OPT", f"PAL{num_procs}"],
    'extra_input_keywords': ["FREQ"],
    'input_blocks': {"cpcm": {"epsilon": 7.23, "refrac": 1.4097},
                     "method RunTyp Opt": {"end": ""},
                     "geom": {"MaxIter": 200}}
    }
nhours = 8 # hours
walltime = 60 * 60 * nhours # seconds
total_mem = 3.8 * num_procs # Gb

computer_resources = {
    "num_machines": 1, "num_mpiprocs_per_machine": num_procs, "max_wallclock_seconds": walltime,
    "max_memory_kb": int(total_mem * 1e6), "withmpi": False, "account": "ucb471_asc1"
    }

@calcfunction
def calculate_binding_affinity(pair_data: Dict, cation_data:Dict, anion_data:Dict):
    T = 298.15
    G_pair = pair_data["freeenergy"]
    G_cation = cation_data["freeenergy"]
    G_anion = anion_data["freeenergy"]
    dG = G_pair - (G_cation + G_anion) # Hartree
    # R is in J/(mol*K)
    dG = (dG * Hartree / J) * Avogadro # J/mol
    K = np.exp(-dG/(R*T))
    return Float(K)

@calcfunction
def extract_cation(structure:StructureData):
    atoms = structure.get_ase()
    for atom in atoms:
        if atom.symbol in ["Li", "Na"]:
            cation_atom = atom
            break
    cation_atoms = Atoms([cation_atom])
    cation_structure = StructureData(ase=cation_atoms)
    return cation_structure

@calcfunction
def extract_anion(structure:StructureData):
    atoms = structure.get_ase()
    for ia, atom in enumerate(atoms):
        if atom.symbol in ["Li", "Na"]:
            del atoms[ia]
    anion_atoms = atoms
    anion_structure = StructureData(ase=anion_atoms)
    return anion_structure

class AssociationWorkChain(WorkChain):

    nhours = 20 # hours
    walltime = 60 * 60 * nhours # seconds
    max_wallclock_seconds = walltime
    account = "ucb487_asc1"

    @classmethod
    def define(cls, spec):
        """ Oxidation/Reduction Potential WorkChain """
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('code', valid_type=AbstractCode)
        spec.input('parameters', valid_type=Dict)
        spec.input('resources', valid_type=Dict)
        spec.outline(
            cls.salt_pair_relaxation, # scan relaxation
            cls.salt_pair_calc,
            cls.anion_relaxation, # scan relaxation
            cls.anion_calc,
            cls.cation_calc,
            cls.calculate_binding_affinity,
        )
        spec.output('association_constant', valid_type=Float)
    
    def salt_pair_relaxation(self):
        structure = self.inputs.structure
        code = self.inputs.code
        builder = code.get_builder()
        builder.structure = structure
        parameters = self.inputs.parameters.get_dict()
        num_procs = self.inputs.resources.get_dict()["num_mpiprocs_per_machine"]
        parameters['input_keywords'] = ["r2SCAN-3c", "OPT"]
        builder.metadata.options.resources = self.inputs.resources.get_dict()
        builder.metadata.options.max_wallclock_seconds = self.max_wallclock_seconds
        max_memory_kb = int(3.8 * num_procs * 1e6) 
        builder.metadata.options.max_memory_kb = max_memory_kb
        builder.metadata.options.withmpi = False
        builder.metadata.options.account = self.account
        builder.parameters = parameters
        future = self.submit(builder)
        return ToContext(ion_pair_relaxation=future)

    def salt_pair_calc(self):
        structure = self.ctx.ion_pair_relaxation.outputs["relaxed_structure"]
        code = self.inputs.code
        builder = code.get_builder()
        builder.structure = structure
        parameters = self.inputs.parameters.get_dict()
        builder.metadata.options.resources = self.inputs.resources.get_dict()
        num_procs = self.inputs.resources.get_dict()["num_mpiprocs_per_machine"]
        builder.metadata.options.max_wallclock_seconds = self.max_wallclock_seconds
        max_memory_kb = int(3.8 * num_procs * 1e6) 
        builder.metadata.options.max_memory_kb = max_memory_kb
        builder.metadata.options.withmpi = False
        builder.metadata.options.account = self.account
        builder.parameters = parameters
        future = self.submit(builder)
        return ToContext(ion_pair_results=future)
    
    def cation_calc(self):
        structure = extract_cation(self.inputs.structure)
        code = self.inputs.code
        builder = code.get_builder()
        builder.structure = structure
        parameters = self.inputs.parameters.get_dict()
        parameters["charge"] = 1
        parameters["multiplicity"] = 1
        parameters["input_keywords"].remove("OPT")        
        builder.metadata.options.resources = self.inputs.resources.get_dict()
        num_procs = self.inputs.resources.get_dict()["num_mpiprocs_per_machine"]
        builder.metadata.options.max_wallclock_seconds = self.max_wallclock_seconds
        max_memory_kb = int(3.8 * num_procs * 1e6) 
        builder.metadata.options.max_memory_kb = max_memory_kb
        builder.metadata.options.withmpi = False
        builder.metadata.options.account = self.account
        builder.parameters = parameters
        future = self.submit(builder)
        return ToContext(cation_results=future)

    def anion_relaxation(self):
        structure = extract_anion(self.inputs.structure)
        code = self.inputs.code
        builder = code.get_builder()
        builder.structure = structure
        parameters = self.inputs.parameters.get_dict()
        num_procs = self.inputs.resources.get_dict()["num_mpiprocs_per_machine"]
        parameters['input_keywords'] = ["r2SCAN-3c", "OPT"]
        parameters["charge"] = -1
        parameters["multiplicity"] = 1        
        builder.metadata.options.resources = self.inputs.resources.get_dict()
        builder.metadata.options.max_wallclock_seconds = self.max_wallclock_seconds
        max_memory_kb = int(3.8 * num_procs * 1e6) 
        builder.metadata.options.max_memory_kb = max_memory_kb
        builder.metadata.options.withmpi = False
        builder.metadata.options.account = self.account
        builder.parameters = parameters
        future = self.submit(builder)
        return ToContext(anion_relaxation=future)  

    def anion_calc(self):
        structure = self.ctx.anion_relaxation.outputs["relaxed_structure"]
        code = self.inputs.code
        builder = code.get_builder()
        builder.structure = structure
        parameters = self.inputs.parameters.get_dict()
        parameters["charge"] = -1
        parameters["multiplicity"] = 1        
        builder.metadata.options.resources = self.inputs.resources.get_dict()
        num_procs = self.inputs.resources.get_dict()["num_mpiprocs_per_machine"]
        builder.metadata.options.max_wallclock_seconds = self.max_wallclock_seconds
        max_memory_kb = int(3.8 * num_procs * 1e6) 
        builder.metadata.options.max_memory_kb = max_memory_kb
        builder.metadata.options.withmpi = False
        builder.metadata.options.account = self.account
        builder.parameters = parameters
        future = self.submit(builder)
        return ToContext(anion_results=future)  

    def calculate_binding_affinity(self):
        pair_data = self.ctx.ion_pair_results.outputs["output_parameters"]
        cation_data = self.ctx.cation_results.outputs["output_parameters"]
        anion_data = self.ctx.anion_results.outputs.output_parameters
        association_constant = calculate_binding_affinity(pair_data, cation_data, anion_data)
        self.out('association_constant', association_constant)

if __name__ == "__main__":
    load_profile()
    code = load_node(92)
    atoms = read(os.path.join(mol_path, "NaPF6", "NaPF6.xyz"))
    structure = StructureData(ase=atoms)
    submit(AssociationWorkChain, {"code": code, "structure": structure})
    
