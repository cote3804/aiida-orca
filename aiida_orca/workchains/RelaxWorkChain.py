from OrcaWorkChain import OrcaWorkChain
from aiida.engine import WorkChain, ToContext
from aiida.orm import StructureData, Dict, AbstractCode

class RelaxWorkChain(WorkChain): 

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            'structure',
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
        spec.expose_outputs(OrcaWorkChain)
        spec.outline(
            cls.relax_molecule,
            cls.finalize
        )
    
    def relax_molecule(self):
        inputs = {
            "parameters": self.inputs.parameters,
            "code": self.inputs.code,
            "structure": self.inputs.structure
        }
        results = self.submit(OrcaWorkChain, **inputs)
        return ToContext(results=results)

    def finalize(self):
        self.out_many(self.exposed_outputs(self.ctx.results, OrcaWorkChain))