# page 298 of ORCA manual
from aiida.engine import BaseRestartWorkChain, while_
from aiida_orca.calculations import OrcaCalculation
from aiida.orm import Dict, AbstractCode, StructureData, Int
from aiida_helpers import multiplicity_from_atoms

class RamanSpectrumWorkChain(BaseRestartWorkChain):

    _process_class = OrcaCalculation

    parameters = {
        "charge": 0, "multiplicity": 1, 
        'input_keywords': ["wB97X-D4", "def2-TZVP", "def2/J", "OPT", "NumFreq", "TightSCF", "RIJCOSX", "LARGEPRINT"],
        'input_blocks': {
            "cpcm": {
                "epsilon": 7.23, "refrac": 1.4097
                },
            "pal":{
                    "nprocs": 0
                    },
            "geom": {
                "MaxIter": 500
                },
            "elprop": {
                "Polar": 1
            }
        }
    }

    @classmethod
    def define(cls, spec):
        """ Homo Lumo Calculation WorkChain """
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('code', valid_type=AbstractCode)
        spec.input('resources', valid_type=Dict) # keep this a regular dictionary
        spec.input('charge', valid_type=Int, required=False)
        spec.input('multiplicity', valid_type=Int, required=False)
        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )
        spec.expose_outputs(OrcaCalculation)

    def setup(self):
        super().setup()
        wallclock_seconds = 60 * 60 * 20 # 20 hours
        nprocs = self.inputs.resources.get_dict()["num_mpiprocs_per_machine"]
        parameters = self.parameters
        parameters["input_blocks"]["pal"]["nprocs"] = nprocs
        if 'charge' in self.inputs:
            parameters["charge"] = self.inputs.charge
        if 'multiplicity' in self.inputs:
            parameters["multiplicity"] = self.inputs.multiplicity
        metadata = {
            "options": {
                "resources": self.inputs.resources.get_dict(),
                "max_wallclock_seconds": wallclock_seconds,
                "withmpi": False,
                "max_memory_kb": int(3.8 * 1e6 * nprocs),
                "account": "ucb487_asc1"
            }
        }
        self.ctx.inputs = {'structure': self.inputs.structure, 'code': self.inputs.code, 'parameters': parameters, "metadata": metadata}