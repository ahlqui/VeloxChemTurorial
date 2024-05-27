import numpy as np
from pathlib import Path
from sys import stdout
from time import time
import xml.etree.ElementTree as ET
from xml.dom import minidom

import openmm as mm
import openmm.app as app
import openmm.unit as unit
from veloxchem import Molecule
import veloxchem as vlx

class OpenMMDynamics:
    """
    Wrapper class for molecular dynamics simulations using OpenMM.
    """
    
    def __init__(self):
        """
        Initializes the class with default simulation parameters.
        """
        self.platform = 'reference'
        self.ensemble = 'NVE'
        self.temperature = 298.15 
        self.friction = 1.0 
        self.timestep = 2.0 
        self.nsteps = 1000
        self.parent_ff = 'amber03.xml'
        self.water_ff = 'spce.xml'
        self.box_size = 3.0 
        self.padding = 3.0
        self.cutoff = 1.0

        self.system = None
        self.integrator = None
        self.simulation = None
        self.pdb = None
        self.modeller = None
        self.labels = []

        # VeloxChem objects
        self.molecule = None
        self.unique_residues = []
        self.unique_molecules = []

        # QM Region parameters
        self.qm_driver = None
        self.grad_driver = None
        self.qm_atoms = None
        self.qm_force_index = None


    # Load and analize a PDB file


    def load_system_PDB(self, filename):
        """
        Loads a system from a PDB file, extracting unique residues and their details.

        :param filename: Path to the PDB file.
        :raises FileNotFoundError: If the specified file does not exist.
        """
        pdb_path = Path(filename)
        if not pdb_path.is_file():
            raise FileNotFoundError(f"{filename} does not exist.")
        
        pdbstr = pdb_path.read_text()
        residues = {}

        # Formatting issues flags
        label_guess_warning = False
        conect_warning = False

        self.unique_residues = []  # Initialize unique_residues earlier

        for line in pdbstr.strip().splitlines():
            if line.startswith(('ATOM', 'HETATM')):
                atom_label = line[76:78].strip()
                if not atom_label:
                    label_guess_warning = True
                    atom_name = line[12:16].strip()
                    atom_label = atom_name[0]

                residue = line[17:20].strip()
                residue_number = int(line[22:26])
                coordinates = [float(line[i:i+8]) for i in range(30, 54, 8)]

                residue_identifier = (residue, residue_number)

                if residue_identifier not in residues:
                    residues[residue_identifier] = {
                        'labels': [],
                        'coordinates': []
                    }
                
                residues[residue_identifier]['labels'].append(atom_label)
                residues[residue_identifier]['coordinates'].append(coordinates)

            # Check if the CONECT records are present and skip them
            # If they are not present activate the warning flag
            if line.startswith('CONECT'):
                conect_warning = False
            else:
                conect_warning = True

        # Overwrite the PDB file with the guessed atom labels in columns 77-78
        # if the labels are missing
        if label_guess_warning:
            with open(filename, 'w') as f:
                for line in pdbstr.strip().splitlines():
                    if line.startswith(('ATOM', 'HETATM')):
                        atom_name = line[12:16].strip()
                        atom_label = atom_name[0]
                        new_line = line[:76] + f"{atom_label:>2}" + line[78:] + '\n'
                        f.write(new_line)
                    else:
                        f.write(line + '\n')

        if conect_warning:
            print('Warning: CONECT records not found in the PDB file.')
            # Create a molecule from the PDB file
            molecule = Molecule.read_PDB_file(filename)
            connectivity_matrix = molecule.get_connectivity_matrix()
            # Determine all the bonds in the molecule
            with open(filename, 'a') as f:
                for i in range(connectivity_matrix.shape[0]):
                    for j in range(i + 1, connectivity_matrix.shape[1]):
                        if connectivity_matrix[i, j] == 1:
                            # Convert indices to 1-based index for PDB format and ensure proper column alignment
                            i_index = i + 1
                            j_index = j + 1
                            # Align to the right 
                            con_string = "{:6s}{:>5d}{:>5d}".format('CONECT', i_index, j_index)
                            f.write(con_string + '\n')
       
            print('Warning: Atom labels were guessed based on atom names (first character).')
            print(f'Please verify the atom labels in the {filename} PDB file.')
            print('The CONECT records were not found in the PDB file.')
            print('The connectivity matrix was used to determine the bonds.')

        # Create VeloxChem Molecule objects for each unique residue

        molecules = []
        unq_residues = []

        for (residue, number), data in residues.items():
            coordinates_array = np.array(data['coordinates'])
            mol = Molecule(data['labels'], coordinates_array, "angstrom")
            molecules.append(mol)
            unq_residues.append((residue, number))


        # Initialize a set to track the first occurrence of each residue name
        seen_residue_names = set()

        # Lists to store the filtered residues and molecules
        self.unique_residues = []
        self.unique_molecules = []

        for index, (residue_name, number) in enumerate(unq_residues):
            if residue_name not in seen_residue_names:
                seen_residue_names.add(residue_name)
                self.unique_residues.append((residue_name, number))
                self.unique_molecules.append(molecules[index])

        # Print results
        print("Unique Residues:", self.unique_residues, "saved as molecules.")
            

    # Method to generate OpenMM system from VeloxChem objects
    def system_from_molecule(self, 
                             molecule, 
                             ff_gen, 
                             phase='gas', 
                             qm_atoms=None, 
                             filename='residue', 
                             residue_name='MOL'):
        """
        Generates an OpenMM system from a VeloxChem molecule and a forcefield generator.
        
        :param molecule:
            VeloxChem molecule object.
        :param ff_gen:
            VeloxChem forcefield generator object.
        :param phase:
            Phase of the system ('gas', 'water', or 'periodic'). Default is 'gas'.
        :param qm_atoms:
            Options: None, 'all', or list of atom indices for QM region.
        :param filename:
            Base name for the generated files. Default is 'residue'.
        :param residue_name:
            Name of the residue. Default is 'MOL'.
        """

        from openmm import NonbondedForce

        # Store the molecule object and generate OpenMM compatible files
        self.molecule = molecule
        self.positions = molecule.get_coordinates_in_angstrom()
        self.labels = molecule.get_labels()

        if qm_atoms:
            if qm_atoms == 'all':
                qm_atoms = list(range(len(self.labels)))
            self._create_QM_residue(ff_gen)
            ff_gen.write_pdb('qm_region.pdb', 'QMR')
            filename = 'qm_region'

        elif qm_atoms is None:
            ff_gen.write_openmm_files(filename, residue_name)
            
        self.pdb = app.PDBFile(f'{filename}.pdb')
        
        # Common forcefield loading, modified according to phase specifics
        forcefield_files = [f'{filename}.xml']
        if phase == 'water':
            forcefield_files.append(self.water_ff)
        
        forcefield = app.ForceField(*forcefield_files)

        # Handling different phases
        if phase == 'gas':
            self.system = forcefield.createSystem(self.pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
        else:
            self.modeller = app.Modeller(self.pdb.topology, self.pdb.positions)
            if phase == 'water':
                self.modeller.addSolvent(forcefield, model='tip3p', padding=self.padding * unit.nanometer)   
            elif phase == 'periodic':
                self.modeller.topology.setUnitCellDimensions(mm.Vec3(self.box_size, self.box_size, self.box_size))

            self.system = forcefield.createSystem(self.modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=self.cutoff * unit.nanometer, constraints=app.HBonds)
        
        # Options for QM/MM simulations
        if qm_atoms:

            # Set the QM/MM Interaction Groups
            total_atoms = self.system.getNumParticles()
            qm_group = set(qm_atoms)
            mm_group = set(range(total_atoms)) - qm_group

            self.qm_atoms = qm_atoms

            # Add custom hessian forces
            force_expression = mm.CustomExternalForce("-fx*x-fy*y-fz*z")
            self.system.addForce(force_expression)
            force_expression.addPerParticleParameter("fx")
            force_expression.addPerParticleParameter("fy")
            force_expression.addPerParticleParameter("fz")

            for i in self.qm_atoms:
                force_expression.addParticle(i, [0, 0, 0])

            # QM Hessian Force Group
            force_expression.setForceGroup(0)

            # If a MM region is present define the interactions
            if mm_group:
                
                # CustomNonbondedForce for QM/MM interactions
                vdw = mm.CustomNonbondedForce("4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)")
                if phase in ['water', 'periodic']:
                    vdw.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
                    vdw.setCutoffDistance(self.cutoff)
                vdw.addPerParticleParameter("sigma")
                vdw.addPerParticleParameter("epsilon")

                coulomb = mm.CustomNonbondedForce("138.935456*charge1*charge2/r;")

                if phase in ['water', 'periodic']:
                    coulomb.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
                    coulomb.setCutoffDistance(self.cutoff)
                coulomb.addPerParticleParameter("charge")

                self.system.addForce(vdw)
                self.system.addForce(coulomb)

                # Add particles to the custom forces
                # QM region
                for i in qm_group:
                    vdw.addParticle([ff_gen.atoms[i]['sigma'], ff_gen.atoms[i]['epsilon']])
                    coulomb.addParticle([ff_gen.atoms[i]['charge']])

                # MM region
                nonbonded_force = None
                custom_nonbonded_forces = []
                for force in self.system.getForces():
                    if isinstance(force, NonbondedForce):
                        nonbonded_force = force
                    elif isinstance(force, mm.CustomNonbondedForce):
                        custom_nonbonded_forces.append(force)

                if nonbonded_force:
                    for i in range(nonbonded_force.getNumExceptions()):
                        p1, p2, chargeProd, sigma, epsilon = nonbonded_force.getExceptionParameters(i)
                        for custom_force in custom_nonbonded_forces:
                            custom_force.addExclusion(p1, p2)

                for i in mm_group:
                    # The charges, sigmas, and epsilons are taken from the system
                    charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
                    vdw.addParticle([sigma, epsilon])
                    coulomb.addParticle([charge])

                qm_group = set(qm_atoms)
                mm_group = set(range(total_atoms)) - qm_group

                vdw.addInteractionGroup(qm_group, mm_group)
                coulomb.addInteractionGroup(qm_group, mm_group)

                # Set force groups
                vdw.setForceGroup(1)
                coulomb.setForceGroup(2)

                # Set a force group for the MM region
                for force in self.system.getForces():
                    # Non bonded MM
                    if isinstance(force, mm.NonbondedForce):
                        force.setForceGroup(3)
                    # Bonded
                    elif isinstance(force, mm.HarmonicBondForce):
                        force.setForceGroup(4)
                    elif isinstance(force, mm.HarmonicAngleForce):
                        force.setForceGroup(5)
                    elif isinstance(force, mm.PeriodicTorsionForce):
                        force.setForceGroup(6)
                    elif isinstance(force, mm.CMMotionRemover):
                        force.setForceGroup(7)
                    
            # Determine the force index for the QM region
            # it is an instance of openmm.openmm.CustomExternalForce
            for i, force in enumerate(self.system.getForces()):
                if isinstance(force, mm.CustomExternalForce):
                    self.qm_force_index = i
                    break
        
        # Write the system to a xml file (for debugging purposes)
        with open(f'{filename}_system.xml', 'w') as f:
            f.write(mm.XmlSerializer.serialize(self.system))
            print(f'System parameters written to {filename}_system.xml')

        # Write the system to a pdb file (for debugging purposes)
        if phase == 'gas':
            app.PDBFile.writeFile(self.pdb.topology, self.positions, open(f'{filename}_system.pdb', 'w'))
            print(f'System coordinates written to {filename}_system.pdb')
        elif phase in ['water', 'periodic']:
            app.PDBFile.writeFile(self.modeller.topology, self.modeller.positions, open(f'{filename}_system.pdb', 'w'))
            print(f'System coordinates written to {filename}_system.pdb')

        self.phase = phase

    # Method to build a custom system from a PDB file and custom XML files
    def build_custom_system(self, 
                            system_pdb, 
                            xml_file, 
                            phase='gas', 
                            nonbondedMethod=app.PME, 
                            nonbondedCutoff=1*unit.nanometer, 
                            constraints=app.HBonds,
                            topology_pdb=None):
        """
        Builds a system from a PDB file containing multiple residues and custom XML files.
        This method assumes that the user has already created the XML files and the PDB
        contains all the residues and PBC conditions.

        :param system_pdb:
            PDB file containing the system. Or a list of PDB files.
        :param xml_file:
            XML file containing the forcefield parameters. Or a list of XML files.
            The parent forcefield is taken from the instance variable parent_ff.
        :param box_size:
            Size of the box. Default is (3.0, 3.0, 3.0) nm
        :param nonbondedMethod:
            Nonbonded method. Default is PME
        :param nonbondedCutoff:
            Nonbonded cutoff. Default is 1 nm
        :param constraints:
            Constraints. Default is app.HBonds
        :param topology_pdb:
            PDB file containing the topology of the system. Default is None.
        """

        # Check if the system_pdb is a list
        if isinstance(system_pdb, list):
            # Create a list of PDB objects
            pdb_files = [app.PDBFile(pdb) for pdb in system_pdb]
            # Add the pdb files to the modeller
            self.modeller = app.Modeller(pdb_files[0].topology, pdb_files[0].positions)
            for pdb in pdb_files[1:]:
                self.modeller.add(pdb.topology, pdb.positions)
            # Save a pdb file of the system
            app.PDBFile.writeFile(self.modeller.topology, self.modeller.positions, open('system.pdb', 'w'))
            self.molecule = Molecule.read_PDB_file('system.pdb')
            if phase == 'periodic':
                self.modeller = app.PDBFile('system.pdb')
                self.modeller.topology.setUnitCellDimensions((self.box_size, self.box_size, self.box_size))

        else:
            self.modeller = app.PDBFile(system_pdb)
            self.molecule = Molecule.read_PDB_file(system_pdb)
            if phase == 'periodic':
                self.modeller = app.PDBFile(system_pdb)
                self.modeller.topology.setUnitCellDimensions((self.box_size, self.box_size, self.box_size))

        if topology_pdb is not None:
            print('Topology PDB file provided')
            if phase == 'periodic':
                pdb = app.PDBFile(topology_pdb)
                pdb.topology.setUnitCellDimensions((self.box_size, self.box_size, self.box_size))
                topology = pdb.topology
            else:
                topology = app.PDBFile(topology_pdb).topology


        # Check if the xml_files is a list
        if isinstance(xml_file, list):

            if phase == 'water':
                forcefield = app.ForceField(self.parent_ff, *xml_file, self.water_ff)
            elif phase == 'gas':
                forcefield = app.ForceField(self.parent_ff, *xml_file)
            elif phase == 'periodic':
                forcefield = app.ForceField(self.parent_ff, *xml_file)
        else:
            if phase == 'water':
                forcefield = app.ForceField(self.parent_ff, xml_file, self.water_ff)
            elif phase == 'gas':
                forcefield = app.ForceField(self.parent_ff, xml_file)
            elif phase == 'periodic':
                forcefield = app.ForceField(self.parent_ff, xml_file)

        # Create the system
        if phase == 'water':
            box = mm.Vec3(self.box_size, self.box_size, self.box_size)
            self.modeller.addSolvent(forcefield, model='tip3p', boxSize=box)
            self.system = forcefield.createSystem(self.modeller.topology, nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff, constraints=constraints)
        elif phase == 'gas':
            if topology_pdb is not None:
                self.system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=constraints)
            else:
                self.system = forcefield.createSystem(self.modeller.topology, nonbondedMethod=app.NoCutoff, constraints=constraints)
        elif phase == 'periodic':
            if topology_pdb is not None:
                self.system = forcefield.createSystem(topology, nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff, constraints=constraints)
            else:
                self.system = forcefield.createSystem(self.modeller.topology, nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff, constraints=constraints)

        self.phase = 'custom'
 

    # Simulation methods
    def energy_minimization(self, max_iter=0, tol=10.0):
        """
        Minimizes the energy of the system using the specified parameters.

        Args:
            max_iter (int): Maximum number of iterations for the minimization. Default is 0 (no limit).
            tol (float): Tolerance for the energy minimization, in kJ/mol. Default is 10.0.

        Raises:
            RuntimeError: If the system has not been created prior to the call.

        Returns:
            float: The minimized potential energy of the system.
            str: XYZ format string of the relaxed coordinates.
        """
        if self.system is None:
            raise RuntimeError('System has not been created!')
        atom_labels = self.molecule.get_labels()

        # Create an integrator and simulation object
        self.integrator = self._create_integrator()

        self.simulation = app.Simulation(self.modeller.topology if 
                                         self.phase in ['water', 'custom', 'periodic'] else
                                         self.pdb.topology,
                                        self.system, self.integrator)
        
        self.simulation.context.setPositions(self.modeller.positions if 
                                             self.phase in ['water', 'custom', 'periodic'] else
                                             self.pdb.positions)

        # Perform energy minimization
        self.simulation.minimizeEnergy(tolerance=tol * unit.kilojoules_per_mole / unit.nanometer, maxIterations=max_iter)
        
        # Retrieve and process the final state of the system
        state = self.simulation.context.getState(getEnergy=True, getPositions=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        coordinates = np.array(state.getPositions().value_in_unit(unit.nanometer)) * 10  # Convert nm to Angstroms

        # Construct XYZ format string for the coordinates
        xyz = str(len(atom_labels)) + '\n\n'
        for i in range(len(atom_labels)):
            xyz += atom_labels[i] + ' ' + str(coordinates[i][0]) + ' ' + str(coordinates[i][1]) + ' ' + str(coordinates[i][2]) + '\n'
        print('Geometry optimized')    
        return energy, xyz
#        xyz = f"{len(self.labels)}\n\n"
#        xyz += "\n".join(f"{label} {x} {y} {z}" for label, (x, y, z) in zip(self.labels, coordinates))
        
#        return energy, xyz
    
    def steepest_descent(self, max_iter=100000, learning_rate=0.01, convergence_threshold=1e-3):
        """
        Performs a steepest descent minimization on the system.
        """

        if self.system is None:
            raise RuntimeError('System has not been created!')
        atom_labels = self.molecule.get_labels()

        # Create an integrator and simulation object
        self.integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)

        self.simulation = app.Simulation(self.modeller.topology if
                                         self.phase in ['water', 'custom', 'periodic'] else
                                         self.pdb.topology,
                                        self.system, self.integrator)

        # Get initial positions
        positions = self.modeller.positions if self.phase in ['water', 'custom', 'periodic'] else self.pdb.positions
        positions = np.array(positions.value_in_unit(unit.nanometer))

        # Define the energy and gradient functions
        def compute_energy_and_gradient(positions):

            self.simulation.context.setPositions(positions)
            state = self.simulation.context.getState(
                getPositions=True,
                getEnergy=True,
                getForces=True)

            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            gradient = -1.0 * state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)

            return energy, gradient

        for iteration in range(max_iter):
            energy, gradients = compute_energy_and_gradient(positions)
#            print("Gradient norm:", np.linalg.norm(gradients))

            # Optional: Normalize gradients
            norm = np.linalg.norm(gradients)
            if norm > 0:
                gradients /= norm

            # Update positions
            new_positions = positions - learning_rate * gradients

            # Adaptive learning rate adjustment
            if iteration > 0 and energy > previous_energy:
                learning_rate *= 0.5
#                print("Reducing learning rate to:", learning_rate)

            previous_energy = energy

            # Check for convergence
            if np.linalg.norm(new_positions - positions) < convergence_threshold:
                print(f"Convergence reached after {iteration+1} iterations.")
                break

            positions = new_positions
 #           print(f"Iteration {iteration+1}: Energy = {energy}")

       # Once converged, return the final energy and positions
       # The positions shall be written in XYZ format in angstroms
        final_positions = positions * 10
#        xyz = f"{len(self.labels)}\n\n"
#        for label, (x, y, z) in zip(self.labels, final_positions):
#            xyz += f"{label} {x} {y} {z}\n"

#        return xyz, energy
        xyz = str(len(atom_labels)) + '\n\n'
        for i in range(len(atom_labels)):
            xyz += atom_labels[i] + ' ' + str(final_positions[i][0]) + ' ' + str(final_positions[i][1]) + ' ' + str(final_positions[i][2]) + '\n'
        print('Geometry optimized')    
        return energy, xyz
     
    def conformational_sampling(self, ensemble='NVT', temperature=700, timestep=2.0, nsteps=1000, snapshots=10):

        """
        Runs a high-temperature MD simulation to sample conformations and minimize the energy of these conformations.

        Args:
            ensemble (str): Type of ensemble ('NVE', 'NVT', 'NPT'). Default is 'NVT'.
            temperature (float): Temperature of the system in Kelvin. Default is 700 K.
            timestep (float): Timestep of the simulation in femtoseconds. Default is 2.0 fs.
            nsteps (int): Number of steps in the simulation. Default is 1000.
            snapshots (int): Frequency of saving snapshots. Default is every 100 steps.
            out_file (str): File path to save the conformation snapshots. Default is 'conformations.xyz'.

        Returns:
            tuple: Tuple containing a list of energies and a list of XYZ format strings of the relaxed coordinates.

        Raises:
            RuntimeError: If the system has not been created or if the molecule object is not defined.
        """
        if self.system is None:
            raise RuntimeError('System has not been created!')
        if self.molecule is None:
            raise RuntimeError('Molecule object does not exist!')

        self.ensemble = ensemble
        self.temperature = temperature * unit.kelvin
        self.timestep = timestep * unit.femtosecond
        self.nsteps = nsteps

        self.integrator = self._create_integrator()
        topology = self.modeller.topology if self.phase in ['water', 'custom', 'periodic'] else self.pdb.topology
        self.positions = self.modeller.positions if self.phase in ['water', 'custom', 'periodic'] else self.pdb.positions

        self.simulation = app.Simulation(topology, self.system, self.integrator)
        self.simulation.context.setPositions(self.positions)
        self.simulation.context.setVelocitiesToTemperature(self.temperature)

        save_freq = nsteps // snapshots if snapshots else nsteps
        energies = []
        opt_coordinates = []

        for step in range(nsteps):
            self.simulation.step(1)
            if step % save_freq == 0:
                state = self.simulation.context.getState(getPositions=True, getEnergy=True)
                energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                coordinates = state.getPositions()
                self.simulation.context.setPositions(coordinates)  # Re-setting positions for potential energy minimization

                print(f'Step: {step}, Potential energy: {energy}')
                self.simulation.minimizeEnergy()
                minimized_state = self.simulation.context.getState(getPositions=True, getEnergy=True)
                minimized_energy = minimized_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                minimized_coordinates = minimized_state.getPositions()

                energies.append(minimized_energy)
                print(f'Minimized energy: {minimized_energy}')
                xyz = f"{len(self.labels)}\n\n"
                for label, coord in zip(self.labels, minimized_coordinates):
                    xyz += f"{label} {coord.x * 10} {coord.y * 10} {coord.z * 10}\n"  # Convert nm to Angstroms
                print('Saved coordinates for step', step)
                opt_coordinates.append(xyz)

        print('Conformational sampling completed!')
        print(f'Number of conformations: {len(opt_coordinates)}')

        return energies, opt_coordinates

    def run_md(self, ensemble='NVE', temperature=298.15, pressure=1.0, friction=1.0,
            timestep=2.0, nsteps=1000, snapshots=100, out_file='trajectory.pdb'):
        """
        Runs an MD simulation using OpenMM, storing the trajectory and simulation data.

        Args:
            ensemble (str): Type of ensemble. Options are 'NVE', 'NVT', 'NPT'. Default is 'NVE'.
            temperature (float): Temperature of the system in Kelvin. Default is 298.15 K.
            pressure (float): Pressure of the system in atmospheres. Default is 1.0 atm.
            friction (float): Friction coefficient in 1/ps. Default is 1.0.
            timestep (float): Timestep of the simulation in femtoseconds. Default is 2.0 fs.
            nsteps (int): Number of steps in the simulation. Default is 1000.
            snapshots (int): Frequency of snapshots. Default is 100.
            out_file (str): Output file name for the trajectory. Default is 'trajectory.pdb'.

        Raises:
            RuntimeError: If the system has not been previously created.

        Notes:
            This method initializes the simulation environment, sets initial conditions, 
            and runs the simulation, outputting the results to specified files.
        """
        if self.system is None:
            raise RuntimeError('System has not been created!')

        self.ensemble = ensemble
        self.temperature = temperature * unit.kelvin
        self.friction = friction / unit.picosecond
        self.timestep = timestep * unit.femtoseconds
        self.nsteps = nsteps

        # Create or update the integrator
        new_integrator = self._create_integrator()

        topology = self.modeller.topology if self.phase in ['water', 'custom', 'periodic'] else self.pdb.topology
        positions = self.modeller.positions if self.phase in ['water', 'custom', 'periodic'] else self.pdb.positions
        self.simulation = app.Simulation(topology, self.system, new_integrator)
        self.simulation.context.setPositions(positions)
        self.simulation.minimizeEnergy()

        self.simulation.context.setVelocitiesToTemperature(self.temperature)

        # Set up reporting
        save_freq = max(nsteps // snapshots, 1)
        self.simulation.reporters.clear()  
        self.simulation.reporters.append(app.PDBReporter(out_file, save_freq))
        self.simulation.reporters.append(app.StateDataReporter(stdout, save_freq, step=True,
                                                            potentialEnergy=True, temperature=True, density=True))

        print('Running simulation...')
        print('=' * 50)
        print('Ensemble:', ensemble)
        if ensemble in ['NVT', 'NPT']:
            print('Temperature:', temperature, 'K')
            if ensemble == 'NPT':
                print('Pressure:', pressure, 'atm')
        print('Friction:', friction, '1/ps')
        print('Timestep:', timestep, 'fs')
        print('Total simulation time in ns:', nsteps * timestep / 1e6)
        print('=' * 50)

        start_time = time()
        self.simulation.step(nsteps)
        end_time = time()
        elapsed_time = end_time - start_time
        elapsed_time_days = elapsed_time / (24 * 3600)
        performance = (nsteps * timestep / 1e6) / elapsed_time_days  # Convert fs to ns and seconds to days for ns/day

        print('Simulation completed!')
        print(f'Elapsed time: {int(elapsed_time // 60)} minutes, {int(elapsed_time % 60)} seconds')
        print(f'Performance: {performance:.2f} ns/day')
        print(f'Trajectory saved as {out_file}')

    def restart_md(self,
                   ensemble='NVE',
                   temperature=298.15,
                   pressure=1.0,
                   friction=1.0,
                   timestep=2.0,
                   nsteps=1000,
                   snapshots=100,
                   out_file='trajectory.pdb'):
        
        """
        Restarts an MD simulation using OpenMM, storing the trajectory and simulation data.

        """

        if self.system is None:
            raise RuntimeError('System has not been created!')
        
        self.ensemble = ensemble
        self.temperature = temperature * unit.kelvin
        self.friction = friction / unit.picosecond
        self.timestep = timestep * unit.femtoseconds
        self.nsteps = nsteps

        # In order to restart a simulation, a previous simulation object must exist
        if self.simulation is None:
            raise RuntimeError('No previous simulation object found. Please run a simulation first.')
        
        # Create or update the integrator
        new_integrator = self._create_integrator()

        self.simulation.integrator = new_integrator

        self.simulation.context.reinitialize(preserveState=True)

        # Set initial velocities if the ensemble is NVT or NPT
        if self.ensemble in ['NVT', 'NPT']:
            self.simulation.context.setVelocitiesToTemperature(self.temperature)

        # Set up reporting
        self.simulation.reporters.clear()
        self.simulation.reporters.append(app.PDBReporter(out_file, nsteps // snapshots))
        self.simulation.reporters.append(app.StateDataReporter(stdout, nsteps // snapshots, step=True,
                                                            potentialEnergy=True, temperature=True, density=True))
        
        print('Restarting simulation...')
        print('=' * 50)
        print('Ensemble:', ensemble)
        if ensemble in ['NVT', 'NPT']:
            print('Temperature:', temperature, 'K')
            if ensemble == 'NPT':
                print('Pressure:', pressure, 'atm')
        print('Friction:', friction, '1/ps')
        print('Timestep:', timestep, 'fs')
        print('Total simulation time in ns:', nsteps * timestep / 1e6)
        print('=' * 50)

        start_time = time()
        self.simulation.step(nsteps)
        end_time = time()
        elapsed_time = end_time - start_time
        elapsed_time_days = elapsed_time / (24 * 3600)
        performance = (nsteps * timestep / 1e6) / elapsed_time_days  # Convert fs to ns and seconds to days for ns/day

        print('Simulation completed!')
        print('-' * 60)
        print(f'Elapsed time: {int(elapsed_time // 60)} minutes, {int(elapsed_time % 60)} seconds')
        print(f'Performance: {performance:.2f} ns/day')
        print(f'Trajectory saved as {out_file}')



    def run_qmmm(self, qm_driver, grad_driver, ensemble='NVE', temperature=298.15, pressure=1.0, friction=1.0,
                    timestep=2.0, nsteps=1000, snapshots=100, out_file='trajectory.pdb'):
        """
        Runs a QM/MM simulation using OpenMM, storing the trajectory and simulation data.

        :param ensemble:
            Type of ensemble. Options are 'NVE', 'NVT', 'NPT'. Default is 'NVE'.
        :param temperature:
            Temperature of the system in Kelvin. Default is 298.15 K.
        :param pressure:
            Pressure of the system in atmospheres. Default is 1.0 atm.
        :param friction:
            Friction coefficient in 1/ps. Default is 1.0.
        :param timestep:
            Timestep of the simulation in femtoseconds. Default is 2.0 fs.
        :param nsteps:
            Number of steps in the simulation. Default is 1000.
        :param snapshots:
            Frequency of snapshots. Default is 100.
        :param out_file:
            Output file name for the trajectory. Default is 'trajectory.pdb'.
        """
        if self.system is None:
            raise RuntimeError('System has not been created!')
        
        self.ensemble = ensemble
        self.temperature = temperature * unit.kelvin
        self.friction = friction / unit.picosecond
        self.timestep = timestep * unit.femtoseconds
        self.nsteps = nsteps

        self.qm_driver = qm_driver
        self.grad_driver = grad_driver
        self.qm_driver.ostream.mute()

        qm_potential = []
        qm_mm_interaction_energies = []
        mm_potential = []
        total_potential = []
        kinetic_energy = []
        self.total_energy = []
        
        save_freq = nsteps // snapshots if snapshots else nsteps

        # Create or update the integrator
        new_integrator = self._create_integrator()

        self.topology = (self.modeller.topology if 
                         self.phase in ['water', 'custom', 'periodic'] else 
                         self.pdb.topology)
        
        self.positions = (self.modeller.positions if 
                     self.phase in ['water', 'custom', 'periodic'] else 
                     self.pdb.positions)
        
        self.simulation = app.Simulation(self.topology, self.system, new_integrator)
        self.simulation.context.setPositions(self.positions)

        # Set initial velocities if the ensemble is NVT or NPT
        if self.ensemble in ['NVT', 'NPT']:
            self.simulation.context.setVelocitiesToTemperature(self.temperature)

        # Set up reporting
        self.simulation.reporters.clear()
        self.simulation.reporters.append(app.PDBReporter(out_file, save_freq))
        
        for step in range(nsteps):

            self.update_forces(self.simulation.context)

            # Potential energies
            # QM region
            qm = self.get_qm_potential_energy(self.simulation.context)
            qm_potential.append(qm)

            # QM/MM interactions
            qm_mm = self.simulation.context.getState(getEnergy=True, groups={1,2}).getPotentialEnergy()
            qm_mm_interaction_energies.append(qm_mm.value_in_unit(unit.kilojoules_per_mole))

            # MM region 
            mm = self.simulation.context.getState(getEnergy=True, groups={3,4,5,6,7}).getPotentialEnergy()
            mm_potential.append(mm.value_in_unit(unit.kilojoules_per_mole))

            # Total potential energy
            pot = qm * unit.kilojoules_per_mole + qm_mm + mm
            total_potential.append(pot.value_in_unit(unit.kilojoules_per_mole))

            # Kinetic energy
            kinetic = self.simulation.context.getState(getEnergy=True).getKineticEnergy()
            kinetic_energy.append(kinetic.value_in_unit(unit.kilojoules_per_mole))

            # Total energy
            total = pot + kinetic
            self.total_energy.append(total.value_in_unit(unit.kilojoules_per_mole))


            # Information output
            if step % save_freq == 0:
                
                print(f"Step: {step} / {nsteps} Time: {round(step * timestep, 2)} ps")
                print('Potential Energy QM region:', qm, 'kJ/mol')
                print('Potential Energy MM region:', mm)
                print('QM/MM Interaction Energy:', qm_mm)
                print('Total Potential Energy:', pot)
                print('Kinetic Energy:', kinetic)
                print('Total Energy:', total)
                print('-' * 60)

            self.simulation.step(1)

        print('QM/MM simulation completed!')
        print(f'Number of steps: {nsteps}')
        print(f'Trajectory saved as {out_file}')
        print('=' * 60)
        print('Total energy averages:')
        total_energy_avg = np.mean(self.total_energy)
        std_dev = np.std(self.total_energy)
        print(f'Total Energy: {total_energy_avg} ± {std_dev} kJ/mol')

    # Post-simulation analysis methods
    def visualize_trajectory(self, trajectory_file='trajectory.pdb', interval=1):
        """
        Visualizes the trajectory of the simulation using py3Dmol.

        :param trajectory_file:
            Path to the PDB file containing the trajectory. Default is 'trajectory.pdb'.
        """
        try:
            import py3Dmol

        except ImportError:
            raise ImportError("py3Dmol is not installed. Please install it using `pip install py3Dmol`.")
        
        viewer = py3Dmol.view(width=800, height=600)

        viewer.addModelsAsFrames(open(trajectory_file, 'r').read(),'pdb', {'keepH': True})
        viewer.animate({'interval': interval, 'loop': "forward", 'reps': 10})
        viewer.setStyle({"stick":{},"sphere": {"scale":0.25}})
        viewer.zoomTo()


        viewer.show()


    # Private methods

    def _format_PDB_file(self, pdb_file):
        """
        Formats a PDB file by removing water molecules and adding CONECT records.

        Args:
            pdb_file (str): Path to the PDB file to format.

        Returns:
            str: Path to the formatted PDB file.
        """
        with open(pdb_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            if line.startswith('HETATM'):
                continue
            if line.startswith('CONECT'):
                continue
            new_lines.append(line)

        with open('formatted.pdb', 'w') as f:
            f.writelines(new_lines)

        return 'formatted.pdb'

    def _create_integrator(self):
        """
        Creates an OpenMM integrator object based on the specified ensemble type.

        Returns:
            OpenMM Integrator: Configured integrator for the simulation.
        """
        # Common parameters for Langevin integrators
        if self.ensemble in ['NVT', 'NPT']:
            integrator = mm.LangevinIntegrator(self.temperature, self.friction, self.timestep)
            integrator.setConstraintTolerance(1e-5)
            if self.ensemble == 'NPT':
                # Add a barostat for pressure control in NPT ensemble
                barostat = mm.MonteCarloBarostat(1 * unit.atmospheres, self.temperature, 25)
                self.system.addForce(barostat)
        elif self.ensemble == 'NVE':
            # Use Verlet integrator for the NVE ensemble (energy conservation)
            integrator = mm.VerletIntegrator(self.timestep)
        else:
            raise ValueError("Unsupported ensemble type. Please choose 'NVE', 'NVT', or 'NPT'.")

        return integrator
    
    # Method to create a QM region in the system
    def _create_QM_residue(self, ff_gen, filename='qm_region', residue_name='QMR'):
        """
        This method creates an xml file for a QM region.
        The xml file only contains atomtypes, residue, and nonbonded parameters.

        :param molecule:
            VeloxChem molecule object
        :param ff_gen:
            VeloxChem forcefield generator object
        :param filename:
            Name of the files to be generated. Default is 'qm_region'
        :param residue_name:
            Name of the residue. Default is 'QMR'
        """

        atoms = ff_gen.atoms
        bonds = ff_gen.bonds

        # Create the root element of the XML file
        ForceField = ET.Element("ForceField")
        
        # AtomTypes section
        AtomTypes = ET.SubElement(ForceField, "AtomTypes")

        for i, atom in atoms.items():
            element = ''.join([i for i in atom['name'] if not i.isdigit()])  
            attributes = {
                # Name is the atom type_molname
                "name": atom['name'] + '_' + residue_name,
                "class": str(i + 1),
                "element": element,
                "mass": str(atom['mass']) 
            }
            ET.SubElement(AtomTypes, "Type", **attributes)

        # Residues section
        Residues = ET.SubElement(ForceField, "Residues")
        Residue = ET.SubElement(Residues, "Residue", name=residue_name)
        for atom_id, atom_data in atoms.items():
            ET.SubElement(Residue, "Atom", name=atom_data['name'], type=atom_data['name'] + '_' + residue_name, charge=str(atom_data['charge']))
        for bond_id, bond_data in bonds.items():
            ET.SubElement(Residue, "Bond", atomName1=atoms[bond_id[0]]['name'], atomName2=atoms[bond_id[1]]['name'])

        # NonbondedForce section
        NonbondedForce = ET.SubElement(ForceField, "NonbondedForce", coulomb14scale=str(ff_gen.fudgeQQ), lj14scale=str(ff_gen.fudgeLJ))
        for atom_id, atom_data in atoms.items():
            attributes = {
                "type": atom_data['name'] + '_' + residue_name,
                "charge": str(0.0),
                "sigma": str(0.0),
                #"sigma": str(atom_data['sigma']), # Sigma is used by OpenMM to determine the vdW radius of the atom
                "epsilon": str(0.0)
            }
            ET.SubElement(NonbondedForce, "Atom", **attributes)

        # Generate the tree and write to file
        tree = ET.ElementTree(ForceField)
        rough_string = ET.tostring(ForceField, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        indented_string = reparsed.toprettyxml(indent="    ")  

        with open(filename + '.xml', 'w') as output_file:
            output_file.write(indented_string)

        print(f'QM region parameters written to {filename}.xml')


    def update_gradient_and_energy(self, new_positions):
        """
        Updates and returns the gradient and potential energy of the QM region.

        :param new_positions:
            The new positions of the atoms in the QM region.
        :return:
            The gradient and potential energy of the QM region.
        """

        positions_ang = (new_positions) * 10 
        # Atom labels for the QM region
        atom_labels = [atom.element.symbol for atom in self.topology.atoms()]
        qm_atom_labels = [atom_labels[i] for i in self.qm_atoms]

        new_molecule = vlx.Molecule(qm_atom_labels, positions_ang, units="angstrom")
        self.qm_driver.compute(new_molecule)
        self.grad_driver.compute(new_molecule)

        # Potential energy is in Hartree, convert to kJ/mol
        potential_kjmol = self.qm_driver.get_energy() * vlx.hartree_in_kcalpermol() * 4.184

        return self.grad_driver.get_gradient(), potential_kjmol

    def update_gradient(self, new_positions):
        """
        Updates and returns the gradient of the QM region.

        :param new_positions:
            The new positions of the atoms in the QM region.
        :return:
            The gradient of the QM region.
        """
        gradient, _ = self.update_gradient_and_energy(new_positions)

        return gradient
    
    def update_potential_energy(self, new_positions):
        """
        Updates and returns the potential energy of the QM region.

        :param new_positions:
            The new positions of the atoms in the QM region.
        :return:
            The potential energy of the QM region.
        """
        _, potential_energy = self.update_gradient_and_energy(new_positions)

        return potential_energy

    def update_forces(self, context):
        """
        Updates the forces in the system based on a new gradient.

        Args:
            context: The OpenMM context object.
        """

        conversion_factor = (4.184 * vlx.hartree_in_kcalpermol() * 10.0 / vlx.bohr_in_angstrom()) * unit.kilojoule_per_mole / unit.nanometer
        new_positions = context.getState(getPositions=True).getPositions()

        # Update the forces of the QM region
        qm_positions = np.array([new_positions[i].value_in_unit(unit.nanometer) for i in self.qm_atoms])

        gradient = self.update_gradient(qm_positions)
        force = -np.array(gradient) * conversion_factor

        for i, atom_idx in enumerate(self.qm_atoms):
            self.system.getForce(self.qm_force_index).setParticleParameters(atom_idx, atom_idx, force[i])
        self.system.getForce(self.qm_force_index).updateParametersInContext(context)
    
    def get_qm_potential_energy(self, context):
        """
        Returns the potential energy of the QM region.

        Args:
            context: The OpenMM context object.
        Returns:
            The potential energy of the QM region.
        """

        positions = context.getState(getPositions=True).getPositions()
        qm_positions = np.array([positions[i].value_in_unit(unit.nanometer) for i in self.qm_atoms])
        potential_energy = self.update_potential_energy(qm_positions)

        return potential_energy
    
    def get_qm_mm_interaction_energy(self, context):
        """
        Returns the potential energy contribution of QM/MM interactions.

        Args:
            context: The OpenMM context object.
        Returns:
            The potential energy of the QM/MM interaction.
        """
        # Get the state of the system with energy for specific force groups
        state_vdw = context.getState(getEnergy=True, groups={1})
        state_coulomb = context.getState(getEnergy=True, groups={2})

        # Extract the energies
        energy_vdw = state_vdw.getPotentialEnergy()
        energy_coulomb = state_coulomb.getPotentialEnergy()

        # Calculate the total QM/MM interaction energy
        qm_mm_interaction_energy = energy_vdw + energy_coulomb

        return qm_mm_interaction_energy.value_in_unit(unit.kilojoules_per_mole)


