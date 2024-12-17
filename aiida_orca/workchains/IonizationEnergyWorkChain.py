# Work chain to calculate the ionization energy of a molecule
from aiida.engine import ToContext, WorkChain, calcfunction, run_get_pk
from aiida.orm import AbstractCode, Float, Str, Dict, StructureData, load_code, load_computer, load_node, Code
import click
import os
import sys
from aiida.engine import submit, run
from aiida import load_profile
from scipy.constants import physical_constants
Farad = physical_constants['Faraday constant'][0] # Faraday constant in C/mol

# Based on methods found here:
# https://perssongroup.lbl.gov/papers/compmatsci2015-electrolytegenome.pdf

#### INPUT PARAMETERS ####
neutral_parameters = {
    "charge": 0, "multiplicity": 1, 'input_keywords': ["B3LYP", "6-31+G*", "OPT"],
    'extra_input_keywords': ["LARGEPRINT", "CPCM(DMF)"]
    }

oxidized_parameters = {
    "charge": 1, "multiplicity": 2, 'input_keywords': ["B3LYP", "6-31+G*", "OPT"],
    'extra_input_keywords': ["LARGEPRINT", "CPCM(DMF)"]
    }

reduced_parameters = {
    "charge": -1, "multiplicity": 2, 'input_keywords': ["B3LYP", "6-31+G*", "OPT"],
    'extra_input_keywords': ["LARGEPRINT", "CPCM(DMF)"]
    }

computer_resources = {
    "num_machines": 1, "num_mpiprocs_per_machine": 1, 
    }

walltime = 60 * 60 * 8 # 8 hours in seconds
##########################

# Standard Li/Li+ absolute potential:
# U(SHE) = -4.44 V
# U(Li/Li+) = -3.05 V vs SHE
# U(Li/Li+) on an absolute scale = 4.44 - 3.05 = -1.39 V
absolute_potential = -4.44 - (-3.05)


@calcfunction
def get_oxidation_potential(neutral_data, ox_data):
    E_ox = ox_data['scfenergies'][-1] - neutral_data['scfenergies'][-1]
    # U_ox = -E_ox / Farad
    U_ox = -E_ox # leave as absolute potential
    return Float(U_ox)

@calcfunction
def get_reduction_potential(neutral_data, red_data):
    E_red = red_data['scfenergies'][-1] - neutral_data['scfenergies'][-1]
    # U_red = -E_red / Farad
    U_red = -E_red # leave as absolute potential
    return Float(U_red)


    
class IonizationEnergy(WorkChain):

    @classmethod
    def define(cls, spec):
        """ Oxidation/Reduction Potential WorkChain """
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('code', valid_type=AbstractCode)
        spec.outline(
            cls.neutral_calc,
            cls.ox_calc,
            cls.red_calc,
            cls.calculate_ionization_potentials,
        )
        spec.output('oxidation_potential', valid_type=Float)
        spec.output('reduction_potential', valid_type=Float)
    

    def neutral_calc(self):
        structure = self.inputs.structure
        code = self.inputs.code
        builder = code.get_builder()
        builder.structure = structure
        parameters = self.inputs.parameters
        computer_resources = {
        "num_machines": 1, "num_mpiprocs_per_machine": 8, 
        }
        parameters = Dict(parameters)
        parameters.description = "Input parameters for neutral molecule ORCA calculation"
        parameters.store()
        builder.parameters = parameters
        max_wallclock_seconds = walltime
        builder.metadata.options.resources = computer_resources
        builder.metadata.options.max_wallclock_seconds = max_wallclock_seconds
        builder.metadata.options.withmpi = False
        future = self.submit(builder)
        return ToContext(neutral_results=future)
    
    def ox_calc(self):
        structure = self.inputs.structure
        code = self.inputs.code
        builder = code.get_builder()
        builder.structure = structure
        parameters = self.inputs.parameters
        parameters["charge"] = 1
        computer_resources = {
        "num_machines": 1, "num_mpiprocs_per_machine": 8, 
        }
        parameters.description = "Input parameters for oxidized molecule ORCA calculation"
        builder.parameters = parameters
        max_wallclock_seconds = walltime
        builder.metadata.options.resources = computer_resources
        builder.metadata.options.max_wallclock_seconds = max_wallclock_seconds
        builder.metadata.options.withmpi = False
        future = self.submit(builder)
        return ToContext(ox_results=future)

    def red_calc(self):
        structure = self.inputs.structure
        code = self.inputs.code
        builder = code.get_builder()
        builder.structure = structure
        parameters = self.inputs.parameters
        parameters["charge"] = -1
        computer_resources = {
        "num_machines": 1, "num_mpiprocs_per_machine": 8, 
        }
        parameters.description = "Input parameters for reduced molecule ORCA calculation"
        builder.parameters = parameters
        max_wallclock_seconds = walltime
        builder.metadata.options.resources = computer_resources
        builder.metadata.options.max_wallclock_seconds = max_wallclock_seconds
        builder.metadata.options.withmpi = False
        future = self.submit(builder)
        return ToContext(red_results=future)
    
    def calculate_ionization_potentials(self):
        neutral_data = self.ctx.neutral_results.outputs.output_parameters
        ox_data = self.ctx.ox_results.outputs.output_parameters
        red_data = self.ctx.red_results.outputs.output_parameters
        oxidation_potential = get_oxidation_potential(neutral_data, ox_data)
        reduction_potential = get_reduction_potential(neutral_data, red_data)
        self.out('oxidation_potential', oxidation_potential)
        self.out('reduction_potential', reduction_potential)

class ParallelIonizationEnergy(WorkChain):
    @classmethod
    def define(cls, spec):
        """ Oxidation/Reduction Potential WorkChain """
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('code', valid_type=AbstractCode)
        spec.input('parameters', valid_type=Dict)
        spec.input('resources')
        spec.outline(
            cls.parallel_calcs,
            cls.calculate_ionization_potentials,
        )
        spec.output('oxidation_potential', valid_type=Float)
        spec.output('reduction_potential', valid_type=Float)
    
    def build_neutral_calc(self):
        structure = self.inputs.structure
        code = self.inputs.code
        builder = code.get_builder()
        builder.structure = structure
        parameters = self.inputs.parameters
        resources = self.inputs.resources
        # parameters.description = "Input parameters for neutral molecule ORCA calculation"
        # parameters.store()
        builder.parameters = parameters
        max_wallclock_seconds = walltime
        builder.metadata.options.resources = resources.get_dict()
        num_procs = resources.get_dict()["num_mpiprocs_per_machine"]
        builder.metadata.options.max_wallclock_seconds = max_wallclock_seconds
        builder.metadata.options.withmpi = False
        builder.metadata.options.max_memory_kb = int(3.8 * 1e6 * num_procs)
        return builder
    
    def build_ox_calc(self):
        structure = self.inputs.structure
        code = self.inputs.code
        builder = code.get_builder()
        builder.structure = structure
        parameters = self.inputs.parameters.get_dict()
        parameters["charge"] = 1
        parameters["multiplicity"] = 2
        paremeters = Dict(parameters)
        resources = self.inputs.resources
        # parameters.description = "Input parameters for oxidized molecule ORCA calculation"
        # parameters.store()
        builder.parameters = parameters
        max_wallclock_seconds = walltime
        builder.metadata.options.resources = resources.get_dict()
        num_procs = resources.get_dict()["num_mpiprocs_per_machine"]
        builder.metadata.options.max_wallclock_seconds = max_wallclock_seconds
        builder.metadata.options.withmpi = False
        builder.metadata.options.max_memory_kb = int(3.8 * 1e6 * num_procs)
        return builder

    def build_red_calc(self):
        structure = self.inputs.structure
        code = self.inputs.code
        builder = code.get_builder()
        builder.structure = structure
        parameters = self.inputs.parameters.get_dict()
        parameters["charge"] = -1
        parameters["multiplicity"] = 2
        paremeters = Dict(parameters)
        resources = self.inputs.resources
        # parameters.description = "Input parameters for reduced molecule ORCA calculation"
        # parameters.store()
        builder.parameters = parameters
        max_wallclock_seconds = walltime
        builder.metadata.options.resources = resources.get_dict()
        num_procs = resources.get_dict()["num_mpiprocs_per_machine"]
        builder.metadata.options.max_wallclock_seconds = max_wallclock_seconds
        builder.metadata.options.withmpi = False
        builder.metadata.options.max_memory_kb = int(3.8 * 1e6 * num_procs)
        return builder
    
    def parallel_calcs(self):
        neutral_calc = self.build_neutral_calc()
        neutral_results = self.submit(neutral_calc)
        ox_calc= self.build_ox_calc()
        ox_results = self.submit(ox_calc)
        red_calc = self.build_red_calc()
        red_results = self.submit(red_calc)
        return ToContext(neutral_results=neutral_results, ox_results=ox_results, red_results=red_results)
    
    def calculate_ionization_potentials(self):
        neutral_data = self.ctx.neutral_results.outputs.output_parameters
        ox_data = self.ctx.ox_results.outputs.output_parameters
        red_data = self.ctx.red_results.outputs.output_parameters
        oxidation_potential = get_oxidation_potential(neutral_data, ox_data)
        reduction_potential = get_reduction_potential(neutral_data, red_data)
        self.out('oxidation_potential', oxidation_potential)
        self.out('reduction_potential', reduction_potential)