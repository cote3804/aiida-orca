from aiida.orm import StructureData, AbstractCode, Dict, Int
from aiida.common import AttributeDict
from aiida.common.exceptions import InputValidationError
from aiida.engine import BaseRestartWorkChain, ToContext, while_, process_handler, ProcessHandlerReport
from aiida.plugins import CalculationFactory
from aiida_orca.calculations import OrcaCalculation
from aiida_helpers import multiplicity_from_atoms
from ase.io import read
from io import StringIO
import numpy as np


class OrcaWorkChain(BaseRestartWorkChain):

    # _process_class = CalculationFactory('orca.orca')
    _process_class = OrcaCalculation

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            'structure', 
            valid_type=StructureData,
            required=True
            )
        spec.input(
            'code', 
            valid_type=AbstractCode,
            required=True
        )
        spec.input(
            'parameters', 
            valid_type=Dict,
            required=True
            )
        spec.input(
            'num_procs',
            valid_type=Int,
            required=False
        )
        spec.input(
            'walltime',
            valid_type=(float, int),
            required=False,
            default=20,
            help="Walltime in hours",
            non_db=True
        )
        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )
        spec.expose_outputs(cls._process_class)

    @process_handler
    def handle_failure_during_optimization(self, node):
        if node.exit_status == OrcaCalculation.exit_codes.ERROR_CALCULATION_UNSUCCESSFUL.status:
            if "OPT" in self.ctx.inputs["parameters"].get_dict()["input_keywords"]:
                # optimization calculation. Restart from partially optimized geoemtry
                xyz_string = node.outputs.retrieved.get_object_content("aiida.xyz")
                with StringIO(xyz_string) as f:
                    optimized_atoms = read(f, format="xyz")
                self.ctx.inputs["structure"] = StructureData(ase=optimized_atoms)
            return ProcessHandlerReport()

    def setup(self):
        """Call the `setup` of the `BaseRestartWorkChain` and then create the inputs dictionary in `self.ctx.inputs`.
        This `self.ctx.inputs` dictionary will be used by the `BaseRestartWorkChain` to submit the calculations in the
        internal loop."""
        super().setup()
        params = self.inputs.parameters.get_dict()
        atoms = self.inputs.structure.get_ase()
        if 'num_procs' in self.inputs:
            num_procs = self.inputs.num_procs
            params["input_blocks"]["pal"] = {"nprocs": num_procs}
        else:
            # check for PAL# in input_keywords list or in seperate %pal block in input_blocks
            if "pal" in params['input_blocks']:
                num_procs = Int(params['input_blocks']['pal']['nprocs'])
            else:
                for tag in params['input_keywords']:
                    if tag.startswith("PAL"):
                        num_procs = Int(tag.split("PAL")[-1])
        multiplicity = multiplicity_from_atoms(atoms, params["charge"])
        if multiplicity != int(params["multiplicity"]):
            raise InputValidationError("input charge and multiplicity are inconsistent")
        num_nodes = np.ceil(num_procs/64)
        num_procs_per_node = num_procs / num_nodes
        metadata = {
            "options": {
                "resources": {"num_machines": num_nodes, "num_mpiprocs_per_machine": num_procs_per_node},
                "max_wallclock_seconds": int(60 * 60 * self.inputs.walltime), # hours
                "withmpi": False,
                "max_memory_kb": int(3.8 * 1e6 * num_procs_per_node),
                "account": "ucb487_asc1"
            }
        }
        inputs = {
            "structure": self.inputs.structure,
            "code": self.inputs.code,
            "parameters": Dict(params),
            "metadata": metadata,
            }
        self.ctx.inputs = inputs