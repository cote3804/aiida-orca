from aiida.orm import StructureData
from ase.io import read
import os
from aiida.orm import StructureData, AbstractCode, Float, Dict, load_node
from aiida.engine import run, submit, WorkChain, calcfunction, ToContext, workfunction, BaseRestartWorkChain, while_
from ase import Atoms, Atom
import numpy as np
from aiida_orca.calculations import OrcaCalculation

class OrcaCalculationWorkChain(BaseRestartWorkChain):
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