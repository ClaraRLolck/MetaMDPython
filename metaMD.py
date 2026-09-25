#!/groups/kemi/dpg158/miniconda3/envs/rdkit_env/bin/python
import os
import sys
import shutil
import subprocess
import tempfile
import textwrap
import hashlib
import random
import time
import traceback
import xyz2mol
import numpy as np
import pandas as pd
from rdkit.Chem.rdmolops import GetFormalCharge
from rdkit.Chem import rdmolops

from rdkit.Geometry import Point3D
from rdkit.Chem import rdDistGeom
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
from collections import deque


from tblite.ase import TBLite
from ase import units, Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import fs, Bohr, Hartree
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import write, read
from ase.md.md import MolecularDynamics
from RMSD_opt import compute_rmsd_and_grad
import re
from fairchem.core import pretrained_mlip, FAIRChemCalculator



def centerofmass(positions, mass):
    """Return the center of mass and total mass of a system given positions and masses"""
    mass = np.asarray(mass, dtype=float)
    positions = np.asarray(positions, dtype=float)
    totmass = mass.sum()
    if totmass == 0.0:
        com = np.zeros(3)
    else:
        com = (mass[:, None] * positions).sum(axis=0) / totmass
    return com, totmass

def rmrottr(positions, mass, vel):
    """Return velocities with overall translation and rotation removed, and positions shifted to center of mass frame"""
    pos = np.asarray(positions, dtype=float)
    vel = np.asarray(vel, dtype=float)
    mass = np.asarray(mass, dtype=float)

    com, totmass = centerofmass(pos, mass)
    r = pos - com

    angmon = (mass[:, None] * np.cross(r, vel)).sum(axis=0)

    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]
    I = np.array(
        [
            [(mass * (y * y + z * z)).sum(), -(mass * x * y).sum(), -(mass * x * z).sum()],
            [-(mass * x * y).sum(), (mass * (x * x + z * z)).sum(), -(mass * y * z).sum()],
            [-(mass * x * z).sum(), -(mass * y * z).sum(), (mass * (x * x + y * y)).sum()],
        ]
    )

    omega = np.linalg.solve(I, angmon)
    rlm = (mass[:, None] * vel).sum(axis=0)   # vector (3,)

    v_com = rlm / totmass 

    vel_corrected = vel - v_com - np.cross(omega, r)

    return vel_corrected, pos



class BiasCalc(Calculator):
    """Calculator that adds a bias potential to the base calculator."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, base_calc, bias_func, **kwargs):
        super().__init__(**kwargs)
        self.base = base_calc
        self.bias_func = bias_func

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        positions = atoms.get_positions()

        self.base.calculate(atoms, properties=("energy", "forces"), system_changes=system_changes)
        base_energy = self.base.results.get("energy", 0.0)
        base_forces = self.base.results.get("forces", np.zeros_like(positions))

        Ebias, Fbias = self.bias_func(positions)

        self.results["energy"] = base_energy + Ebias
        self.results["forces"] = base_forces + Fbias


class GXTBCommandlineCalc(Calculator):
    """Calculator that runs g-xTB via command line and extracts energy and forces."""
    implemented_properties = ["energy", "forces"]
    gradient_name = "gradient"
    energy_pattern = re.compile(
        r"(?:\|\s*)?TOTAL ENERGY\s+(-?\d+\.\d+(?:[Ee][+-]?\d+)?)",
        re.IGNORECASE,
    )

    def __init__(self, xtb_bin="xtb", charge=0, **kwargs):
        super().__init__(**kwargs)
        self.xtb_bin = xtb_bin
        self.charge = int(charge)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        os.makedirs("gxtb", exist_ok=True)
        run_dir =  "gxtb"
        input_file = os.path.join(run_dir, "input.xyz")
        write(input_file, atoms)

        p = subprocess.run(
            [self.xtb_bin, "input.xyz", "--gxtb", "--grad", "--chrg", str(self.charge)],
            cwd=run_dir,
            capture_output=True,
            text=True,
            check=True,
        )

        m = self.energy_pattern.search(p.stdout)
        gradfile = next(
            (os.path.join(run_dir, name) for name in [self.gradient_name] if os.path.exists(os.path.join(run_dir, name))),
            None,
        )

        vals = []
        with open(gradfile, "r") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) != 3:
                    continue
                vals.extend(float(x) for x in parts)


        vals = np.asarray(vals, dtype=float)
        self.results["energy"] = float(m.group(1)) * Hartree
        self.results["forces"] = -vals[: 3 * len(atoms)].reshape((len(atoms), 3)) * (Hartree / Bohr)

class BerendsenVelocity(MolecularDynamics):
    """Berendsen thermostat for velocity Verlet integration."""
    def __init__(self, atoms, temperature_K, taut, timestep, **kwargs):
        super().__init__(atoms, timestep, **kwargs)
        self.atoms = atoms
        self.target_temp = temperature_K
        self.tau = taut
        self.dt = timestep
        self._masses = self.atoms.get_masses().copy()
        self._inv_masses = 1.0 / self._masses
        self.nfreedom = max(1, 3 * len(self.atoms) - 6)

    def step(self):
        """Perform a single integration step using the Berendsen thermostat."""
        forces = self.atoms.get_forces()
        accelerations = forces * self._inv_masses[:, np.newaxis]
        xyzo = self.atoms.get_positions()
        velo = self.atoms.get_velocities()
        veln = velo + 0.5 * accelerations * self.dt

        ekin = 0.5 * (self._masses[:, np.newaxis] * (veln * veln)).sum()
        T = max((2 * ekin) / (self.nfreedom * units.kB), 1.0e-12)
        xlam2 = np.sqrt(1.0 + (self.dt / self.tau) * (self.target_temp / T - 1.0))
        vel = xlam2 * (velo + accelerations * self.dt)

        self.atoms.set_positions(xyzo + vel * self.dt)
        vel_mid = 0.5 * (velo + vel)
        ekin = 0.5 * (self._masses[:, np.newaxis] * (vel_mid * vel_mid)).sum()
        T = max((2 * ekin) / (self.nfreedom * units.kB), 1.0e-12)

        xlam2 = np.sqrt(1.0 + (self.dt / self.tau) * (self.target_temp / T - 1.0))

        self.atoms.set_velocities(vel)
        vel, c = rmrottr(self.atoms.get_positions(), self.atoms.get_masses(), self.atoms.get_velocities())
        self.atoms.set_velocities(vel)
        self.atoms.set_positions(c)

        

class ReactionDetected(Exception):
    pass
class MD:
    def __init__(
        self,
        atoms,
        s_factor=1.6,
        check_time=5,
        charge=0,
        dt=0.4,
        temp=300,
        k_hill_push=0.05,
        alpha=0.3,
        ramp=0.03,
        dump_fs=10.0,
        time_ps=100.0,
        #wall_k=0.019,
        h_mass = 2.0,
        wall_temp=6000.0, 
        method="g-xTB"
    ):
        self.atoms = atoms
        self.dt = float(dt)
        self.kT = float(temp)
        self.k_hill_push = float(k_hill_push) *27.211386245988  #Convert to eV
        self.alpha = float(alpha)/Bohr**2
        self.step = 0
        self.ref_atoms = self.atoms.copy()
        self.ref = self.atoms.get_positions().copy()
        self.current_bias = 0
        self.current_energy=0
        self.bias_list = deque(maxlen=100)
        self.ramp = float(ramp)
        self.dump_fs = float(dump_fs)
        self.time_ps = float(time_ps)
        #self.wall_k = float(wall_k)
        self.h_mass = None if h_mass is None else float(h_mass)
        self.R = 0
        self.s_factor = float(s_factor)
        self.checktime = float(check_time)
        self.charge = charge
        self.wall_temp=float(wall_temp)
        self.method = method

        masses = atoms.get_masses()  
        if self.h_mass is not None:
            masses = [self.h_mass if abs(x - 1.008) < 1.0e-6 else x for x in masses]
        atoms.set_masses(masses)
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=self.kT)
        base = self.set_base(self.method)

        atoms.calc = BiasCalc(base_calc=base, bias_func=self.compute_bias)
        self.integrator = BerendsenVelocity(self.atoms, timestep=self.dt *units.fs, temperature_K=self.kT, taut=500*units.fs)

        self._seed_initial_bias_reference()

        positions = atoms.get_positions()
        center = positions.mean(axis=0) 
        distances = np.linalg.norm(atoms.get_positions() - center, axis=1)
        self.R = distances.max()

    def _seed_initial_bias_reference(self, atom_displacement=1.0e-6):
        """Mimic xtb: initialize metadynamics with one tiny displaced reference."""
        if self.bias_list:
            return
        pos = self.atoms.get_positions().copy()
        disp = np.zeros_like(pos)
        for i in range(pos.shape[0]):
            while True:
                r = 2.0 * np.random.random(3) - 1.0
                nrm = np.linalg.norm(r)
                if nrm >= 1.0e-8:
                    disp[i] = (atom_displacement / nrm) * r
                    break
        self.bias_list.append((pos + disp, self.step))

    def set_base(self, method, xtb_bin="/software/kemi/xtb/6.7.1/bin/xtb"):
        """Set the base calculator based on the specified method."""
        if method == "g-xTB":
            base = GXTBCommandlineCalc(
                xtb_bin=xtb_bin,
                charge=self.charge,
            )
        elif method == "UMA":
            predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", inference_settings="turbo")
            base = FAIRChemCalculator(predictor, task_name="omol")
        return base

    def record_bias_pos(self):
        """Record the current positions and step number in the bias list."""
        pos = self.atoms.get_positions().copy()
        self.bias_list.append((pos, self.step))


    def compute_bias(self, current_pos, beta=10.0):
        """Compute the bias energy and forces based on the current positions and the history of biased positions."""
        Ebias = 0
        Fbias = np.zeros_like(current_pos)
        s = self.s_factor
        kwall = units.kB * self.wall_temp
        for idx, (pos, step) in enumerate(self.bias_list):
            if idx == len(self.bias_list)-1:
                dmp = 2.0 / (1.0 + np.exp(-self.ramp * (self.step-step))) - 1.0
            else:
                dmp = 1.0
            rmsd, grad = compute_rmsd_and_grad(current_pos, pos)
            hill_push = self.k_hill_push * np.exp(-self.alpha * rmsd**2)*dmp
            Ebias += hill_push
            dE_drmsd = - 2 * self.alpha * rmsd * hill_push
            Fbias +=  - dE_drmsd * grad

        center = current_pos.mean(axis=0)
        r = current_pos - center
        d = np.linalg.norm(r, axis=1)
        expo = np.exp(-beta * (self.R * s - d))
        frac = expo / (1.0 + expo)
        g_wall = kwall * beta * (frac / d)[:, None] * r
        wall_bias = np.log1p(expo).sum()

        Fbias -= g_wall
        Ebias += kwall * wall_bias

        return Ebias, Fbias
    
    def update_step(self):
        self.step += 1
        self.atoms.info['md_step'] = self.step
 
    def check_opt_xyz(self):
        """
        goes through the initially saved structures to see if they still (after
        optimization) correspond to a reaction haven taken place.
        """
        global n_steps_list_opt
        global reactions
        global canonical_reactants
        global canonical_products
        global smiles_list
        if self.step > 1:
            n_steps_list = split_trajectory('xtb.trj', self.charge, self.step)

            os.chdir('analyze_trajectory_'+str(self.step))

            xyz_files = [f for f in os.listdir() if f.endswith('.xyz')]
            for xyz_file in xyz_files:
                run_cmd("/software/kemi/xtb/6.7.1/bin/xtb {0} --opt tight --gxtb --gfn2 --chrg {1}".format(xyz_file, self.charge))
                if os.path.exists('xtbopt.xyz'):
                    os.rename('xtbopt.xyz', xyz_file[:-4]+'.optxyz')
           

            opt_files = [f for f in os.listdir(os.curdir) if f.endswith("optxyz")]
            opt_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            if not smiles_list:
                n_steps_list_opt.clear()
                reactions.clear()
                canonical_reactants.clear()
                canonical_products.clear()

            database_dir = os.path.join('../', 'structure_database')
            for i, optfile in enumerate(opt_files):
                try:
                    smiles, formal_charge, res_status = extract_smiles(optfile, self.charge,
                                                                    allow_charge=True,
                                                                    check_ac=True)
                    if formal_charge != self.charge:
                        smiles, formal_charge, res_status = extract_smiles(optfile, self.charge,
                                                                        allow_charge=False,
                                                                        check_ac=True)
                except:
                    try:
                        smiles, formal_charge, res_status = extract_smiles(optfile, self.charge,
                                                                        allow_charge=False,
                                                                        check_ac=True)
                    except:
                        continue
                if formal_charge != self.charge:
                    continue
                print(f"Optfile {optfile}: SMILES = {smiles}, Previous = {smiles_list[-1] if smiles_list else 'None'}")
                if not smiles_list:
                    smiles_list.append(smiles)
                    shutil.copy(optfile, 'optimized_structures')
                    save_structure_to_database(smiles, optfile, res_status,
                                            database_dir)

                elif smiles != smiles_list[-1]:
                    prev_smiles = smiles_list[-1]
                    current_can = canonicalize_smiles(smiles)
                    previous_can = canonicalize_smiles(prev_smiles)

                    if current_can == previous_can:
                        smiles_list.append(smiles)
                        continue
                    reactant = prev_smiles
                    reaction = reactant + '>>' + smiles
                    reactions.append(reaction)

                    canonical_reactants.append(previous_can)
                    canonical_products.append(current_can)

                    smiles_list.append(smiles)
                    n_steps_list_opt.append(n_steps_list[i])

                    shutil.copy(optfile, 'optimized_structures')
                    save_structure_to_database(smiles, optfile, res_status,
                                               database_dir)
            if len(smiles_list)>1:
                raise ReactionDetected(f"Reaction detected in MTD")
            os.chdir('../')
            self.s_factor = self.s_factor - 0.06
        return
    def trj_dump(self):
        with open("energies.txt", "a") as elog:
            elog.write(str(self.atoms.get_total_energy())+"\n")
        write('xtb.trj', self.atoms, append=True, format='xyz')

    def run(self):
        m = 1/self.dt
        self.integrator.attach(self.record_bias_pos, interval=1000*m)
        self.integrator.attach(self.update_step)
        self.integrator.attach(self.check_opt_xyz, interval=self.checktime*m*1000)
        self.integrator.attach(self.trj_dump, interval=10*m)
        try :
            self.integrator.run(100000*m)
        except ReactionDetected:
            raise # Re-raise to exit gracefully
        except:
            self.check_opt_xyz()
            raise


def run_cmd(cmd):
    """
    Run command line
    """
    cmd = cmd.split()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    output, err = p.communicate()
    return output.decode('utf-8')    


def reorder_atoms_to_map(mol):
    """
    Reorders the atoms in a mol objective to match that of the mapping
    """
    atom_map_order = np.zeros(mol.GetNumAtoms()).astype(int)
    for atom in mol.GetAtoms():
        map_number = atom.GetAtomMapNum()-1
        atom_map_order[map_number] = atom.GetIdx()
    mol = Chem.RenumberAtoms(mol, atom_map_order.tolist())
    return mol


def extract_last_structure(trj_file, last_structure_name):
    """
    Extracts the last structure in a trajectory file
    """
    with open(trj_file, 'r') as _file:
        line = _file.readline()
        n_lines = int(line.split()[0])+2
    count = 0
    input_file = open(trj_file, 'r')
    dest = None
    for line in input_file:
        if count % n_lines == 0:
            if dest:
                dest.close()
            dest = open(last_structure_name, "w")
        count += 1
        dest.write(line)

def embed_smiles_far(smiles):
    """
    create 3D conformer with atom order matching atom mapping. If more than one
    fragment present, they are moved 0.5*n_atoms from each other
    """
    embedded_fragments = []
    mol_sizes = []

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol = reorder_atoms_to_map(mol)

    fragments = Chem.GetMolFrags(mol, asMols=True)
    n_atoms = mol.GetNumAtoms()
    coordinates = np.zeros((n_atoms, 3))


    for fragment in fragments:
        Chem.SanitizeMol(fragment)
        rdmolops.AssignStereochemistry(fragment)
        status = AllChem.EmbedMolecule(fragment, maxAttempts=10000,
                                       randomSeed=RANDOM_SEED)
        if status == -1:
            print('fragment could not be embedded')
            sys.exit("Error: could not embed molecule")

        AllChem.MMFFOptimizeMolecule(fragment)
        embedded_fragments.append(fragment)
        dm = AllChem.Get3DDistanceMatrix(fragment)
        biggest_distance = np.amax(dm)
        mol_sizes.append(biggest_distance)

    np.random.seed(seed=RANDOM_SEED)


    for n_frag, fragment in enumerate(embedded_fragments):
        if n_frag == 0:
            random_vector = np.zeros(3)
        else:
            random_vector = np.random.rand(3)*2-1
            random_vector = random_vector / np.linalg.norm(random_vector)

        conformer = fragment.GetConformer()
        translation_distance = 0.5*(mol_sizes[n_frag]+mol_sizes[0])+2

        for i, atom in enumerate(fragment.GetAtoms()):
            atom_id = atom.GetAtomMapNum()-1
            coord =  conformer.GetAtomPosition(i)
            coord += Point3D(*(translation_distance*random_vector))
            coordinates[atom_id, :] = coord

    rdDistGeom.EmbedMolecule(mol)
    conf = mol.GetConformer()
    for i in range(n_atoms):
        x, y, z = coordinates[i, :]
        conf.SetAtomPosition(i, Point3D(x, y, z))

    return mol



def write_xyz_file(mol, file_name, smiles):
    """
    Embeds a mol object to get 3D coordinates which are written to an .xyz file
    """
    n_atoms = mol.GetNumAtoms()
    charge = Chem.GetFormalCharge(mol)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]

    Chem.SanitizeMol(mol)
    rdmolops.AssignStereochemistry(mol)

    mol = embed_smiles_far(smiles)

    with open(file_name, 'w') as _file:
        _file.write(str(n_atoms)+'\n\n')
        for atom, symbol in enumerate(symbols):
            coord = mol.GetConformers()[0].GetAtomPosition(atom)
            line = " ".join((symbol, str(coord.x), str(coord.y), str(coord.z),
                             "\n"))
            _file.write(line)
        if charge != 0:
            _file.write("$set\n")
            _file.write("chrg "+str(charge)+"\n")
            _file.write("$end")


def write_md_input(scale_factor, time=2):
    """
    Write the input file for an MD xTB calculation based on the input
    variables
    """
    md_file = """\
    $md
       time={1}
       step=0.4
       temp=300
       hmass=2
    $end
    $scc
       temp=300
    $end
    $cma
    $wall
       potential=logfermi
       beta=10.0
       temp=6000
       sphere: auto, all
       autoscale={0}
    $end
    """.format(scale_factor, time)

    with open('md.inp', 'w') as ofile:
        ofile.write(textwrap.dedent(md_file))


def calculate_md_relaxed_structure(smiles, scale_factor, ridx):
    """
    This function submits an md with a box with size scaled by scale_factor and
    extracts last structure of the trajectory file
    """

    os.mkdir('md')
    os.chdir('md')
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol = reorder_atoms_to_map(mol)
    n_atoms = mol.GetNumAtoms()
    charge = GetFormalCharge(mol)
    write_xyz_file(mol, str(ridx)+'.xyz', smiles)
    write_md_input(scale_factor)
    output = run_cmd("/software/kemi/xtb/6.7.1/bin/xtb {0} --omd --input md.inp --gxtb --gfn2 --chrg {1}".format(str(ridx)+'.xyz', charge))

    with open('md_out.log', 'w') as _file:
        _file.write(output)

    out_file = str(scale_factor)+'_md.xyz'
    extract_last_structure('xtb.trj', out_file)
    
    check_md_reaction(out_file, charge, smiles, str(ridx)+'.xyz')
    shutil.copy(out_file, '../')

    os.chdir('../')
    return charge, n_atoms, out_file


def chiral_tags(mol):
    """
    Tag methylene and methyl groups with a chiral tag priority defined
    from the atom index of the hydrogens
    """
    li_list = []
    smarts_ch2 = '[!#1][*]([#1])([#1])([!#1])'
    atom_sets = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts_ch2))
    for atoms in atom_sets:
        atoms = sorted(atoms[2:4])
        prioritized_H = atoms[-1]
        li_list.append(prioritized_H)
        mol.GetAtoms()[prioritized_H].SetAtomicNum(9)
    smarts_ch3 = '[!#1][*]([#1])([#1])([#1])'
    atom_sets = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts_ch3))
    for atoms in atom_sets:
        atoms = sorted(atoms[2:])
        H1 = atoms[-1]
        H2 = atoms[-2]
        li_list.append(H1)
        li_list.append(H2)
        mol.GetAtoms()[H1].SetAtomicNum(9)
        mol.GetAtoms()[H2].SetAtomicNum(9)

    Chem.AssignAtomChiralTagsFromStructure(mol, -1)
    rdmolops.AssignStereochemistry(mol)
    for atom_idx in li_list:
        mol.GetAtoms()[atom_idx].SetAtomicNum(1)

    return mol


def choose_resonance_structure(mol):
    """
    This function creates all resonance structures of the mol object, counts
    the number of rotatable bonds for each structure and chooses the one with
    fewest rotatable bonds (most 'locked' structure)
    """
    resonance_mols = rdchem.ResonanceMolSupplier(mol,
                                                 rdchem.ResonanceFlags.ALLOW_CHARGE_SEPARATION)
    res_status = True
    new_mol = None
    if not resonance_mols:
        print("using input mol")
        new_mol = mol
        res_status = False
    for res_mol in resonance_mols:
        Chem.SanitizeMol(res_mol)
        n_rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(res_mol)
        if new_mol is None:
            smallest_rot_bonds = n_rot_bonds
            new_mol = res_mol
        if n_rot_bonds < smallest_rot_bonds:
            smallest_rot_bonds = n_rot_bonds
            new_mol = res_mol

    Chem.DetectBondStereochemistry(new_mol, -1)
    rdmolops.AssignStereochemistry(new_mol, flagPossibleStereoCenters=True,
                                   force=True)
    Chem.AssignAtomChiralTagsFromStructure(new_mol, -1)
    return new_mol, res_status

def extract_smiles(xyz_file, charge, allow_charge=True, check_ac=False):
    """
    uses xyz2mol to extract smiles with as much 3d structural information as
    possible
    """
    atoms, _, xyz_coordinates = xyz2mol.read_xyz_file(xyz_file)
    try:
        input_mol = xyz2mol.xyz2mol(atoms, xyz_coordinates, charge=charge,
                                          use_graph=True,
                                          allow_charged_fragments=allow_charge,
                                          use_huckel=True, use_atom_maps=True,
                                          embed_chiral=True)
    except:
        input_mol = xyz2mol.xyz2mol(atoms, xyz_coordinates, charge=charge,
                                          use_graph=True,
                                          allow_charged_fragments=allow_charge,
                                          use_huckel=False, use_atom_maps=True,
                                          embed_chiral=True)
    input_mol = reorder_atoms_to_map(input_mol[0])
    structure_mol, res_status = choose_resonance_structure(input_mol)
    structure_mol = chiral_tags(structure_mol)
    rdmolops.AssignStereochemistry(structure_mol)
    structure_smiles = Chem.MolToSmiles(structure_mol)

    if check_ac:
        global AC_SAME
        ac = Chem.GetAdjacencyMatrix(input_mol)
        if not np.all(AC == ac):
            AC_SAME = False
            print("change in AC: stopping")

    return structure_smiles, GetFormalCharge(structure_mol), res_status


def canonicalize_smiles(structure_smiles):
    mol = Chem.MolFromSmiles(structure_smiles, sanitize=False)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
    return canonical_smiles

def get_smiles(xyz_file, charge, check_ac=False):
    """
    Try different things to extract sensible smiles using xyz2mol
    """
    smiles, formal_charge, res_status = extract_smiles(xyz_file, charge,
                                                       allow_charge=True,
                                                       check_ac=check_ac)
    if formal_charge != charge:
        smiles, formal_charge, res_status = extract_smiles(xyz_file, charge,
                                                           allow_charge=False,
                                                           check_ac=check_ac)
    #except:
    #    try:
    #        smiles, formal_charge, res_status = extract_smiles(xyz_file, charge,
    #                                                           allow_charge=False,
    #                                                           check_ac=check_ac)
    #    except:
    #        return None, None, None

    return smiles, formal_charge, res_status



def check_md_reaction(md_xyz, charge, in_smiles, in_xyz):
    """
    Checks if a reaction allready occurred in the MD run: indicating
    barrierless or very low barrier
    """

    smiles, formal_charge, res_status1 = get_smiles(md_xyz, charge,
                                                   check_ac=True)
    if not smiles:
        print("something's fishy with the MD structures")
        sys.exit()

    if formal_charge != charge:
        print("something's fishy with the MD structures charges")
        sys.exit()


    if smiles != in_smiles:
        print("smiles changed during MD")
        global n_steps_list_opt
        global reactions
        global canonical_reactants
        global canonical_products
        global smiles_list
        global MD_REACTION

        MD_REACTION=True
        database_dir = os.path.join('../', 'structure_database')

        r_canonical = canonicalize_smiles(in_smiles)
        canonical_reactants.append(r_canonical)

        optsmiles, formal_charge, res_status2 = get_smiles("xtbopt.xyz", charge,
                                                          check_ac=True)
        if optsmiles != in_smiles:
            print("warning: smiles changes during pre-MD optimization!!!")
            p_canonical = canonicalize_smiles(optsmiles)
            canonical_products.append(p_canonical)
            reactions.append(in_smiles+'>>'+optsmiles)
            save_structure_to_database(optsmiles, "xtbopt.xyz", res_status2,
                                       database_dir)
            smiles_list.append(in_smiles)
            smiles_list.append(optsmiles)


        else:
            reactions.append(in_smiles+'>>'+smiles)

            p_canonical = canonicalize_smiles(smiles)

            canonical_products.append(p_canonical)
            save_structure_to_database(smiles, md_xyz, res_status1, database_dir)

            smiles_list.append(in_smiles)
            smiles_list.append(smiles)

        save_structure_to_database(in_smiles, in_xyz, np.nan, database_dir)

        n_steps_list_opt.append(0)

        if AC_SAME == False:
            global DF
            DF['Reactions'] = reactions
            DF['Reactants_canonical'] = canonical_reactants
            DF['Products_canonical'] = canonical_products
            DF['N_steps'] = n_steps_list_opt

            DF.to_pickle('../dataframe.pkl')


def split_trajectory(trajectory_file, charge, step, n_check=10):
    """
    Check trajectory for reactions every n_check points
    """
    _dir = 'analyze_trajectory_'+str(step)
    os.mkdir(_dir)
    smiles_list = []
    n_steps_list = []
    count = 0
    saved_struc = 1
    global n_steps
    n_steps = 0
    with open(trajectory_file, 'r') as _file:
        line = _file.readline()
        n_lines = int(line.split()[0])+2
        while line:
            if count % (n_lines*n_check) == 0:
#                print(count, saved_struc)
                file_name = _dir+'/'+str(saved_struc)+'.xyz'
                with open(file_name, 'w') as xyz_file:
                    for _ in range(n_lines):
                        xyz_file.write(line)
                        line = _file.readline()
                        count += 1
                try:
                    smiles, formal_charge, _ = extract_smiles(file_name,
                                                              charge, 
                                                              allow_charge=True, 
                                                              check_ac=False)
                    #print(smiles)
                    if formal_charge != charge:
                        #print(formal_charge, charge)
                        smiles, formal_charge, _ = extract_smiles(file_name,
                                                                  charge, 
                                                                  allow_charge=False,
                                                                  check_ac=False)
                        if formal_charge != charge:
                            continue
                    if not smiles_list and formal_charge == charge:
                        smiles_list.append(smiles)
                        print('smiles =', smiles)
                        saved_struc += 1
                        n_steps_list.append(n_steps)
                    elif smiles != smiles_list[-1] and formal_charge == charge:
                        smiles_list.append(smiles)
                        print('smiles = ', smiles)
                        saved_struc += 1
                        n_steps_list.append(n_steps)
                except:
                    print("error reading smiles")
                    pass
                n_steps += n_check
            else:
                line = _file.readline()
                count += 1

    n_steps_list.append(n_steps)

    return n_steps_list

def save_structure_to_database(smiles, xyzfile, res_status, database_dir):
    """
    The saved SMILES is represented by its SHA1 code. Then the database of
    optimized structures is checked for that hash code. If not present in the
    database: save the xyzfile with the hash name and add entrance to the .csv
    file for the structure database
    """
    #database_dir = os.path.join('../../', 'structure_database')
    print("Saving structure to database: ", os.getcwd())

    hashobject = hashlib.sha1(smiles.encode())
    hashcode = hashobject.hexdigest()

    database_path = os.path.join(database_dir, 'structure_database.csv')
    database = pd.read_csv(database_path, index_col=0)

    if hashcode not in database.index:
        database.loc[hashcode, 'smiles'] = smiles
        database.loc[hashcode, 'resonance_run'] = res_status
        structure_path = os.path.join(database_dir, 'xyz_files',
                                      str(hashcode)+'.xyz')
        shutil.copy(xyzfile, structure_path)

    database.to_csv(database_path)


def do_metadynamics_calculation(atoms, time, k_push, alp, scale_factor,
                                charge, method, structure_file=None):
    """
    runs a meta-dynamics calculation with the given input variables
    """

    try:
        md = MD(atoms, 
            check_time=time,
            s_factor=scale_factor,
            k_hill_push=k_push,
            alpha=alp,
            charge = charge,
            method=method)
        md.run()
    except ReactionDetected:
        pass
    except Exception as e:
        print("Error during MTD: ", e)
        



if __name__ == "__main__":
    os.environ["XTBHOME"] = "/software/kemi/xtb/6.7.1/bin/xtb"
    os.environ["OMP_STACKSIZE"] = '2G'
    os.environ["OMP_NUM_THREADS"] = '1'
    os.environ["MKL_NUM_THREADS"] = '1'
    os.system('ulimit -s unlimited')
    start_time = time.time()
    MD_REACTION=False
    SMILES_IDX = sys.argv[1]
    RUN_NR = sys.argv[2]
    #SMILES = '[N:1](=[c:2]1\\[n:3][n:4][o:5][n:6][c:7]1[H:9])\\[H:8]'
    SMILES = sys.argv[3]
    SCALE_FACTOR = sys.argv[4]
    TIME = sys.argv[5]  #time in ps
    K_PUSH = sys.argv[6]
    ALP = sys.argv[7]
    global RANDOM_SEED
    RANDOM_SEED = int(sys.argv[8])
    METHOD = sys.argv[9]
    WITH_PRODUCTS = sys.argv[10] == "True"
#    print(WITH_PRODUCTS)
#    print(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    os.mkdir('run'+str(RUN_NR))
    if WITH_PRODUCTS:
        #shutil.copy(str(SMILES_IDX)+'_initial_structures.xyz', 'run'+str(RUN_NR))
        STRUCTURE_FILE = str(SMILES_IDX)+'_initial_structures.xyz'
    else:
        STRUCTURE_FILE = None
    #STRUCTURE_FILE = str(SMILES_IDX)+'_initial_structures.xyz'
#    print(STRUCTURE_FILE)
    os.chdir('run'+str(RUN_NR))
    os.mkdir("structure_database")
    os.mkdir("structure_database/xyz_files")
    pd.DataFrame().to_csv("structure_database/structure_database.csv")

    MOL = Chem.MolFromSmiles(SMILES, sanitize=False)
    MOL = reorder_atoms_to_map(MOL)
    AC = Chem.GetAdjacencyMatrix(MOL)
    DF = pd.DataFrame()
    AC_SAME = True

    DF = pd.DataFrame()
    AC_SAME = True
    reactions = []
    canonical_reactants = []
    canonical_products = []
    n_steps_list_opt = []
    smiles_list = []

    CHARGE, N_ATOMS, MD_file = calculate_md_relaxed_structure(SMILES, SCALE_FACTOR,
                                                              SMILES_IDX)
    
    atoms = read(MD_file)

    do_metadynamics_calculation(atoms, TIME, K_PUSH, ALP, SCALE_FACTOR,
                                CHARGE, METHOD,
                                structure_file=STRUCTURE_FILE)
    STRUCTURE_FILE = str(SMILES_IDX)+'_initial_structures.xyz'

    print("Ending, saving pkl: ", os.getcwd())
    end_time = time.time()
    with open("timing.txt", "w") as tfile:
        tfile.write("Total time: "+str((end_time-start_time)/60)+" minutes\n")

    DF = pd.DataFrame({
        "Reactions": reactions,
        "Reactants_canonical": canonical_reactants,
        "Products_canonical": canonical_products,
        "N_steps": n_steps_list_opt,
    })

    DF.to_pickle('../dataframe.pkl')
