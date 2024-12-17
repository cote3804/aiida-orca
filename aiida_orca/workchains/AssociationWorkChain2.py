from OrcaWorkChain import OrcaWorkChain
from aiida.engine import WorkChain, ToContext, calcfunction
from aiida.orm import StructureData, Float, Dict, AbstractCode, CalcJobNode
from ase.units import Hartree, J
from scipy.constants import R, Avogadro
import numpy as np

def free_energy(output_parameters: Dict):
    params = output_parameters.get_dict()
    free_energy = int(params["freeenergy"])
    return free_energy

@calcfunction
def calculate_association_constant(cat_params, an_params, pair_params):
    T = 298.15 # orca default
    G_cat = free_energy(cat_params)
    G_an = free_energy(an_params)
    G_pair = free_energy(pair_params)
    dG = G_pair - (G_cat + G_an) # Hartree
    # R is in J/(mol*K)
    dG = (dG / J) * Avogadro # J/mol
    K = np.exp(-dG/(R*T))
    return Float(K)

class AssociationWorkChain2(WorkChain): 

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            'pair_structure',
            valid_type=StructureData,
            required=True
        )
        spec.input(
            'anion_structure',
            valid_type=StructureData,
            required=True
        )
        spec.input(
            'cation_structure',
            valid_type=StructureData,
            required=True
        )
        spec.input(
            'parameters',
            valid_type=Dict,
            required=True
        )
        spec.input(
            'code',
            valid_type=AbstractCode,
            required=True
        )
        #TODO if user passes cation_calc, skip cation calc and take params from this calc.
        # check that calculation parameters are the same? At minimum it needs a freq calc to
        # get free energy
        spec.input( 
            'cation_calc',
            valid_type=CalcJobNode,
            required=False
        )
        spec.outline(
            cls.calculate_free_energies,
            cls.get_association_constant,
        )
        spec.output('Ka', valid_type=Float)
    
    def calculate_free_energies(self):
        pair_inputs = {
            "parameters": self.inputs.parameters,
            "code": self.inputs.code,
            "structure": self.inputs.pair_structure
        }
        pair_results = self.submit(OrcaWorkChain, **pair_inputs)

        anion_params = self.inputs.parameters.get_dict()
        anion_params["charge"] = -1
        anion_inputs = {
            "parameters": Dict(anion_params),
            "code": self.inputs.code,
            "structure": self.inputs.anion_structure
        }
        anion_results = self.submit(OrcaWorkChain, **anion_inputs)

        cation_params = self.inputs.parameters.get_dict()
        cation_params["charge"] = 1
        cation_inputs = {
            "parameters": Dict(cation_params),
            "code": self.inputs.code,
            "structure": self.inputs.cation_structure
        }
        cation_results = self.submit(OrcaWorkChain, **cation_inputs)

        return ToContext(cation_results=cation_results, anion_results=anion_results, pair_results=pair_results)
    
    def get_association_constant(self):
        an_res = self.ctx.anion_results.outputs.output_parameters
        cat_res = self.ctx.cation_results.outputs.output_parameters
        pair_res = self.ctx.pair_results.outputs.output_parameters
        association_constant = calculate_association_constant(cat_res, an_res, pair_res)
        self.out('Ka', association_constant)
