# Work chain to calculate the ionization energy of a molecule
from aiida.engine import ToContext, WorkChain, calcfunction, run_get_pk, while_, if_, return_
from aiida.orm import AbstractCode, Float, Str, Dict, StructureData, load_code, load_computer, load_node, Code
import click
import os
import sys
from aiida.engine import submit, run
from aiida import load_profile
from scipy.constants import physical_constants
from pymatgen.core import Structure
from OrcaWorkChain import OrcaWorkChain
Farad = physical_constants['Faraday constant'][0] # Faraday constant in C/mol

# Based on methods found here:
# https://perssongroup.lbl.gov/papers/compmatsci2015-electrolytegenome.pdf


walltime = 60 * 60 * 20 # 20 hours in seconds
##########################

def extract_energy(params: Dict):
    return params.get_dict()["scfenergies"][-1] + params.get_dict()["dispersionenergies"][-1]

@calcfunction
def assemble_structures(structures_dict):
    return Dict(structures_dict)

@calcfunction
def assemble_energies(energies_dict):
    return Dict(energies_dict)

class ComplexOptimizeWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        """ Oxidation/Reduction Potential WorkChain """
        super().define(spec)
        spec.input('structure_dict', valid_type=Dict)
        spec.input('code', valid_type=AbstractCode)
        spec.input('parameters', valid_type=Dict)
        spec.input('account', valid_type=Str)
        spec.outline(
            cls.initialize_structure_count,
            cls.initialize_complex_data,
            while_(cls.strucutre_count_less_than_number_of_structures)(
                cls.complex_relaxation,
                cls.store_data,
                cls.increment_structure_count,
            ),
            cls.return_data
        )
        spec.output('structures', valid_type=Dict)
        spec.output('energies', valid_type=Dict)
    
    def initialize_structure_count(self):
        self.ctx.structure_count = 0
    
    def store_data(self):
        current_struct_index = str(self.ctx.structure_count)
        current_result = self.ctx.complex_result
        output_params = current_result.outputs.output_parameters
        energy = extract_energy(output_params)
        structure = current_result.outputs.relaxed_structure.get_pymatgen()
        structure_d = structure.as_dict()
        self.ctx.complex_structures[current_struct_index] = structure_d
        self.ctx.complex_energies[current_struct_index] = energy

    def initialize_complex_data(self):
        self.ctx.complex_structures = {}
        self.ctx.complex_energies = {}


    def strucutre_count_less_than_number_of_structures(self):
        return self.ctx.structure_count < len(self.inputs.structure_dict.get_dict().keys())
    
    def increment_structure_count(self):
        self.ctx.structure_count += 1

    def complex_relaxation(self):
        struct_dict = self.inputs.structure_dict.get_dict()[str(self.ctx.structure_count)]
        structure = Structure.from_dict(struct_dict)
        structure = StructureData(pymatgen=structure)
        parameters = self.inputs.parameters
        nprocs = int(parameters.get_dict()["input_blocks"]["pal"]["nprocs"])
        parameters = Dict(parameters)
        inputs = {
            "parameters": self.inputs.parameters,
            "code": self.inputs.code,
            "structure": structure
        }
        future = self.submit(OrcaWorkChain, **inputs)
        return ToContext(complex_result = future)

    def return_data(self):
        struct_dict = assemble_structures(self.ctx.complex_structures)
        energies_dict = assemble_energies(self.ctx.complex_energies)
        self.out('structures', struct_dict)
        self.out('energies', energies_dict)