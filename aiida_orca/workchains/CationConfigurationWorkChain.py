# work chain to sample cation placements relative to salt anion
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.visualize import view
from ase import Atoms, Atom
from ase.optimize import BFGS
from os.path import join as opj
from ase.data import vdw_radii, covalent_radii
from xtb.ase.calculator import XTB
import numpy as np
# from typing import Union, List
from functools import partial
from aiida.orm import StructureData, List, load_node, Int, Bool, Float, Dict
from aiida.engine import calcfunction, run, workfunction, WorkChain
import aiida
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.molecule_matcher import MoleculeMatcher

@calcfunction
def generate_configs(struct: StructureData, grid_density, optimize):
    atoms = struct.get_ase()
    placements = generate_placements(atoms, grid_density=int(grid_density))
    structures = generate_placed_atoms(placements=placements, atoms=atoms)
    opt_structs = Dict()
    for i, structure in enumerate(structures):
        if optimize == True:
            structure = xtb_optimize(structure)
        structure = AseAtomsAdaptor.get_molecule(structure)
        # structure = StructureData(ase=structure)
        opt_structs[str(i)] = structure.as_dict()
    return opt_structs


# import tkinter

### Successful ASE optimization using xtb forcefield ###
# atoms.calc = XTB(method="GFN-FF")
# dyn = BFGS(atoms, trajectory="test.traj")
# dyn.run(fmax=0.05)

# traj = Trajectory("test.traj")
# view(traj)

### ML ANI Attempt ###
# def atoms_to_ml_molecule(atoms:Atoms):
#     symbols = atoms.get_chemical_symbols()
#     positions = atoms.get_positions()
#     ml_atoms = [ml.data.atom(element_symbol=x, xyz_coordinates=y) for x, y in zip(symbols, positions)]
#     return ml.data.molecule(atoms=ml_atoms)

# ml_molecule = atoms_to_ml_molecule(atoms)
# calculator = ml.models.methods(method="ANI-1ccx")
# geomopt = ml.optimize_geometry(model=calculator, initial_molecule=ml_molecule)
# optmol = geomopt.optimized_molecule
# print(optmol.get_xyz_string())

def get_radius(atoms:Atoms):
    COM = atoms.get_center_of_mass()
    furthest_index = np.argmax(np.linalg.norm(atoms.positions - COM, axis=1))
    furthest_distance = np.linalg.norm(atoms.positions[furthest_index] - COM)
    return furthest_distance

def carts_from_polar(theta, phi, r):
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(phi)
    return x, y, z

def dist_sort(atom, ref_position):
    return np.linalg.norm(atom.position - ref_position)

def get_a(atoms:Atoms, placement_radius, placement_coordinates, intersection_atom:Atom, radius_scaling_factor):
    # get coefficient on COM vector
    COM = atoms.get_center_of_mass()
    r = (placement_radius + covalent_radii[intersection_atom.number]) * radius_scaling_factor
    v = (COM - placement_coordinates)/np.linalg.norm(COM - placement_coordinates)
    x = intersection_atom.position
    polynomial = [(1), (2*np.dot(COM, v) - 2*np.dot(x, v)), (np.dot(COM, COM) - 2*np.dot(COM,x) + np.dot(x,x) -r**2)]
    a = np.roots(polynomial)
    return a

def get_adjusted_placement(atoms:Atoms, placement_radius:float, placement_coordinates, intersection_atom:Atom, radius_scaling_factor=1.3):
    COM = atoms.get_center_of_mass()
    v = (COM - placement_coordinates)/np.linalg.norm(COM - placement_coordinates)
    # get coefficient on v such that COM + a * v = new placement point
    a = get_a(atoms, placement_radius, placement_coordinates, intersection_atom, radius_scaling_factor)
    idx = np.argmax(np.abs(a))
    y = COM + a[idx]*v
    return y

def find_intersection(placement_coords, placement_radius, atoms:Atoms):
    # atoms is the molecule being complexed
    # placement coords and radius correspond to the atom/molecule being placed
    COM = atoms.get_center_of_mass()
    COM_vec = COM - placement_coords
    cylinder_radius = placement_radius
    intersection_atoms = []
    for iatom in atoms:
        # angle between vector connection COM and placement atom and vector connecting COM and checked atom
        vdw_radius = covalent_radii[iatom.number]
        v = COM_vec/np.linalg.norm(COM_vec)
        m = np.dot(v, iatom.position) - np.dot(v, placement_coords)
        I = placement_coords + m*v
        L = I - iatom.position
        if np.linalg.norm(L) < np.linalg.norm(vdw_radius + cylinder_radius):
            intersection_atoms.append(iatom)
            # atoms.append(Atom(symbol="Li", position=iatom.position))
        else:
            continue
    sorted_atoms = sorted(intersection_atoms, key=partial(dist_sort, ref_position=placement_coords))
    intersection_atom = sorted_atoms[0]
    return intersection_atom

def remove_cation(atoms):
    if "Na" in atoms.get_chemical_symbols():
        symbols = atoms.get_chemical_symbols()
        cat_idx = symbols.index("Na")
        atoms.pop(cat_idx)
    return atoms

def calculate_molecule_radius(atoms:Atoms) -> float:
    COM = atoms.get_center_of_mass()
    average_radius = 0
    for atom in atoms:
        COM_dist = np.linalg.norm(atom.position - COM)
        atom_edge_dist = COM_dist + covalent_radii[atom.number]
        average_radius += atom_edge_dist/len(atoms)
    return average_radius


def generate_placement_coordinates(atoms:Atoms, grid_density, placement_atoms=None):
    # returns an Atoms object that has all the cation coordinates
    # atoms is the molecule being complexed with other molecules/atoms
    if placement_atoms == None:
        place_atom = Atom(symbol = "Na") 
        placement_radius = covalent_radii[place_atom.number]
    elif isinstance(placement_atoms, Atoms):
        placement_radius = calculate_molecule_radius(placement_atoms)
    else:
        raise Exception("placement_atoms must be None or an ASE.Atoms object")

    atoms = remove_cation(atoms)
    COM = atoms.get_center_of_mass()
    radius = get_radius(atoms) 
    circumference = 2 * np.pi * radius
    # if isinstance(grid_density, int):
        # theta - bounded from 0 to pi - angle from vector on xy plane to x axis
        # phi - bounded from 0 to 2*pi - angle from vector to z axis
    theta_vec = np.arange(0, np.pi, np.pi/grid_density)
    phi_vec = np.arange(0, 2*np.pi, np.pi/grid_density)
        # 2x as many phi points as theta points by default
    placement_coordinates = []
    for phi in phi_vec:
        # get the radius of the circle formed on the cross section of this phi angle
        crossection_radius = radius * np.sin(phi)
        crossection_circumference  = 2 * np.pi * crossection_radius
        # calculate number of points around crossection circle by multiplying
        # number of points along phi by the ratio of the circumference of the sphere
        # to the circumference of the crossection circle. This ensures point spacing is constant
        # for each crossection in the sphere
        num_points = np.floor(len(phi_vec) * crossection_circumference/circumference)
        theta_vec = np.arange(0, 2*np.pi, 2 * np.pi/num_points)
        for theta in theta_vec:
            x, y, z = carts_from_polar(theta, phi, radius) + COM
            placement_coords = np.array([x,y,z])
            intersection_atom = find_intersection(placement_coords, placement_radius, atoms)
            adjusted_position = get_adjusted_placement(atoms, placement_radius, placement_coords, intersection_atom, radius_scaling_factor=2)
            placement_coordinates.append(adjusted_position)
    return placement_coordinates

def xtb_optimize(atoms:Atoms) -> Atoms:
    print(type(atoms))
    atoms.calc = XTB(method="GFN-FF")
    dyn = BFGS(atoms)
    dyn.run(fmax=0.05)
    return atoms

def generate_placed_atoms(placements:np.ndarray, atoms:Atoms, placement_atoms:Atoms) -> Atoms:
    # takes placements from generate_placements and returns atoms objects
    # with each placement added to the "atoms" structure
    for position in placements:
        return_atoms = atoms.copy()
        COM = placement_atoms.get_center_of_mass()
        COM_translation = position - COM
        positions = placement_atoms.positions
        positions = positions + COM_translation
        placement_atoms.set_positions(positions)
        return_atoms.extend(placement_atoms)
        yield(return_atoms)

if __name__ == "__main__":
    molecules_root = "/home/coopy/ucb_research/Batteries/molecules"
    atoms = read(opj(molecules_root, "NaBoPh", "NaBoPh.xyz"))
    atoms = remove_cation(atoms)
    placement_atoms = read(opj(molecules_root, "EC", "EC.xyz"))
    cation_placements = generate_placement_coordinates(atoms, grid_density=4, placement_atoms=placement_atoms)
    for i, config in enumerate(generate_placed_atoms(cation_placements, atoms, placement_atoms)):
        config.calc = XTB(method="GFN-FF")
        dyn = BFGS(config, trajectory=f"EC_configs/atoms_{i}.traj")
        dyn.run(fmax=0.05)
        write(f"EC_configs/atoms_{i}.xyz", config)

    # AiiDA execution
    # aiida.load_profile(profile="cooper")
    # struct = load_node(31) # loading StructureData from provenance
    # atoms = read("/home/coopy/onedrive/Research/Batteries/molecules/NaBhfip/NaBhfip.xyz")
    # struct = StructureData(ase=atoms)
    # result = run(generate_configs, {"struct":struct, "grid_density":Int(4), "optimize": Bool(False)})