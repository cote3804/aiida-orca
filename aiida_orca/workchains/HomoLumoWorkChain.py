from aiida.engine import ToContext, WorkChain, calcfunction, run_get_pk, BaseRestartWorkChain, process_handler, while_
from aiida.orm import AbstractCode, Int, Float, Str, Dict, StructureData, load_code, load_node, Code
from aiida.plugins.factories import CalculationFactory
from ase.io import read
from aiida import load_profile
import click
import os
import ase
import sys
# from aiida_orca.calculations import OrcaCalculation
from aiida.engine import submit, run

OrcaCalculation = CalculationFactory("orca.orca")

molecules_dir = "/home/cooper/Research/NaBat/molecules"

def extract_homo(data):
    homo_index = data["homos"][0]
    E_homo = data["moenergies"][0][homo_index]    
    # E_homo = Int(E_homo)
    return E_homo

def extract_lumo(data):
    homo_index = data["homos"][0]
    E_lumo = data["moenergies"][0][homo_index+1]
    # E_lumo = Int(E_lumo)
    return E_lumo

@calcfunction
def get_homo(outputdata):
    E_homo = extract_homo(outputdata)
    return Float(E_homo)

@calcfunction
def get_lumo(outputdata):
    E_lumo = extract_lumo(outputdata)
    return Float(E_lumo)

class HomoLumoWorkChain(BaseRestartWorkChain):

    _process_class = OrcaCalculation

    @classmethod
    def define(cls, spec):
        """ Homo Lumo Calculation WorkChain """
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('code', valid_type=AbstractCode)
        spec.input('parameters', valid_type=Dict)
        spec.input('resources', valid_type=Dict) # keep this a regular dictionary
        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.extract_homo_lumo,
            cls.results,
        )
        spec.expose_outputs(OrcaCalculation)
        spec.output('homo', valid_type=Float)
        spec.output('lumo', valid_type=Float)
    
    def setup(self):
        super().setup()
        wallclock_seconds = 60 * 60 * 20 # 8 hours
        nprocs = self.inputs.resources.get_dict()["num_mpiprocs_per_machine"]
        
        metadata = {
            "options": {
                "resources": self.inputs.resources.get_dict(),
                "max_wallclock_seconds": wallclock_seconds,
                "withmpi": False,
                "max_memory_kb": int(3.8 * 1e6 * nprocs),
                "account": "ucb487_asc1"
            }
        }
        self.ctx.inputs = {'structure': self.inputs.structure, 'code': self.inputs.code, 'parameters': self.inputs.parameters, "metadata": metadata}

    # def orca_calc(self):
    #     structure = self.inputs.structure
    #     code = self.inputs.code
    #     builder = code.get_builder()
    #     builder.structure = structure
    #     parameters = {
    #         "charge": 0, "multiplicity": 1, 'input_keywords': ["B3LYP", "DEF2-SVP"],
    #         'extra_input_keywords': ["LARGEPRINT"]
    #         }
    #     parameters = Dict(parameters)
    #     builder.parameters = parameters
    #     resources = {
    #         "num_machines": 1, "num_mpiprocs_per_machine": 1, 
    #         }
    #     max_wallclock_seconds = 60*30
    #     builder.metadata.options.resources = resources
    #     builder.metadata.options.max_wallclock_seconds = max_wallclock_seconds
    #     builder.metadata.options.withmpi = True
    #     future = self.submit(builder)
    #     return ToContext(calc_results=future)

    def extract_homo_lumo(self):
        last_calc = self.ctx.children[self.ctx.iteration - 1]
        outputs = self.exposed_outputs(last_calc, OrcaCalculation)
        # outputs = self.exposed_outputs(OrcaCalculation)
        # outputdata = self.ctx.output_parameters
        outputdata = outputs["output_parameters"]
        E_homo = get_homo(outputdata)
        E_lumo = get_lumo(outputdata)
        # E_lumo = extract_lumo(outputdata)
        self.out('homo', E_homo)
        self.out('lumo', E_lumo)

if __name__ == "__main__":

    load_profile()
    code = load_code(83)
    workchain = HomoLumoWorkChain
    # submitting two work chains

    # Work Chain 1:
    atoms1 = read(os.path.join(molecules_dir, "diglyme", "diglyme.xyz"), format="xyz")
    structure1 = StructureData(ase=atoms1)
    builder1 = workchain.get_builder()
    builder1.structure = structure1
    builder1.code = code
    submit(builder1)

    # Work Chain 2:
    atoms2 = read(os.path.join(molecules_dir, "EC", "EC.xyz"), format="xyz")
    structure2 = StructureData(ase=atoms2)
    builder2 = workchain.get_builder()
    builder2.structure = structure2
    builder2.code = code
    submit(builder2)

