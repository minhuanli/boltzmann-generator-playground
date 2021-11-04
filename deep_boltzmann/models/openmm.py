import numpy as np
import tensorflow as tf
from simtk import unit
from simtk import openmm

class OpenMMEnergy(object):
        
    def __init__(self, openmm_system, openmm_integrator, length_scale, n_atoms=None, openmm_integrator_args=None, n_steps=0):
        self._length_scale = length_scale
        self._openmm_integrator = openmm_integrator(*openmm_integrator_args)

        self._openmm_context = openmm.Context(openmm_system, self._openmm_integrator)
        
        kB_NA = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        self._unit_reciprocal = 1. / (self._openmm_integrator.getTemperature() * kB_NA)

        self.energy_tf = wrap_energy_as_tf_op(self.__call__)
        
        self.n_steps = n_steps

        self.dim = 3*n_atoms

        self.natoms = n_atoms
        self.atom_indices = np.arange(self.dim).reshape(self.natoms, 3).astype(np.int32)

    def _reduce_units(self, x):
        return x * self._unit_reciprocal
    
    def _assign_openmm_positions(self, configuration):
        positions = openmm.unit.Quantity(
            value=configuration.reshape(-1, 3), 
            unit=self._length_scale)
        self._openmm_context.setPositions(positions)
    
    def _get_energy_from_openmm_state(self, state):
        energy_quantity = state.getPotentialEnergy()
        return self._reduce_units(energy_quantity)
    
    def _get_gradient_from_openmm_state(self, state):
        forces_quantity = state.getForces(asNumpy=True)
        return -1. * np.ravel(self._reduce_units(forces_quantity) * self._length_scale)
    
    def _simulate(self, n_steps):
        self._openmm_integrator.step(n_steps)

    def _get_state(self, **kwargs):
        return self._openmm_context.getState(**kwargs)

    def __call__(self, batch, n_steps=0):
        """batch: (B, N*D) """
        
        gradients = np.zeros_like(batch, dtype=batch.dtype)
        energies = np.zeros((batch.shape[0], 1), dtype=batch.dtype)    
        
        # force `np.float64` for OpenMM
        batch_ = batch.astype(np.float64)
        
        for batch_idx, configuration in enumerate(batch_):
            if np.all(np.isfinite(configuration)):
                self._assign_openmm_positions(configuration)
                if n_steps > 0:
                    self._simulate(n_steps)
                state = self._get_state(getForces=True, getEnergy=True)
                energies[batch_idx] = self._get_energy_from_openmm_state(state)
                # zero out gradients for non-finite energies 
                if np.isfinite(energies[batch_idx]):
                    gradients[batch_idx] = self._get_gradient_from_openmm_state(state)
                    
        return energies, gradients
    
    def energy(self, batch):
        """batch: (B, N*D) """
        
        energies = np.zeros(batch.shape[0], dtype=batch.dtype)    
        
        # force `np.float64` for OpenMM
        batch_ = batch.astype(np.float64)
        
        for batch_idx, configuration in enumerate(batch_):
            if np.all(np.isfinite(configuration)):
                self._assign_openmm_positions(configuration)
                if self.n_steps > 0:
                    self._simulate(self.n_steps)
                state = self._get_state(getEnergy=True)
                energies[batch_idx] = self._get_energy_from_openmm_state(state)

        return energies

    

def wrap_energy_as_tf_op(compute_energy, n_steps=0):
    """Wraps an energy evaluator in a tensorflow op that returns gradients
        
            `compute_energy`:    Callable that takes a (B, N*D) batch of `configuration` and returns the total energy (scalar)
                                 over all batches (unaveraged) and the (B*N, D) tensor of all gradients wrt to the batch
                                 of configurations.
    """
    
    @tf.custom_gradient
    def _energy(configuration):
        """Actual tf op that is evaluated in the `tf.Graph()` built by `keras.Model.compile()`
           
               `configuration`: (B, D*N) tensor containing the B batches of D*N dimensional configurations.
            
            Returns
                        `energy`:   Scalar containg the average energy of the whole batch
                        `grad_fun`: Function returning the gradients wrt configuration given gradient wrt output  according to the chain rule
        """
        n_batch, n_system_dim = configuration.get_shape().as_list()
        dtype = configuration.dtype
        
        batch_size, ndims = configuration.shape

        # here we can call our python function using the `tf.py_func` wrapper
        # important to note: this has to be executed on the master node (only important for distributed computing)

        potential_energy, gradients = tf.numpy_function(func=compute_energy, inp=[configuration], Tout=[dtype, dtype])
        potential_energy.set_shape((n_batch, 1))
        gradients.set_shape((n_batch, n_system_dim))

        
        def _grad_fn(grad_out):
            """Function returing the gradeint wrt configuration given the gradient wrt output according to the chain rule:
            
                    takes `dL/df`
                    and returns `dL/dx = dL/df * df/dx`
            """
            # enforce (B, 1) for scalar outputs
            if len(grad_out.get_shape().as_list()) < 2:
                grad_out = tf.expand_dims(grad_out, axis=-1)
            gradients_in = grad_out * gradients
            return gradients_in
        return potential_energy, _grad_fn
    return _energy


def setup_protein(pdbmodel_dir, temp, implicit_solvent=True):
    """ Sets up protein Topology and Energy Model

    Parameters
    ----------
    pdbmodel_dir: str
        path str to the model PDB file

    temp: int or float
        Temperature of the system in Kelvin

    implicit_solvnet: Boolean, default True
        Choose which force field file to use. If True, use the implicit solvent one; else use the explicit solvent model

    Returns
    -------
    top : mdtraj Topology object
        Protein topology
    energy : Energy object
        Energy model
    """
    INTEGRATOR_ARGS = (temp*unit.kelvin, 1.0 /
                       unit.picoseconds, 2.0*unit.femtoseconds)

    from simtk.openmm import app
    from simtk.openmm import LangevinIntegrator
    import mdtraj as md

    pdb = app.PDBFile(pdbmodel_dir)
    if implicit_solvent:
        forcefield = app.ForceField(
            'amber99sb.xml', 'amber99_obc.xml')  # implicit Solvent
    else:
        forcefield = app.ForceField(
            'amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')  # explicit Solvent

    system = forcefield.createSystem(pdb.topology, removeCMMotion=False,
                                     nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=1.0*unit.nanometers,
                                     constraints=None, rigidWater=True)

    protein_omm_energy = OpenMMEnergy(system,
                                      LangevinIntegrator,
                                      unit.nanometers,
                                      n_atoms=md.Topology().from_openmm(pdb.topology).n_atoms,
                                      openmm_integrator_args=INTEGRATOR_ARGS)

    mdtraj_topology = md.Topology().from_openmm(pdb.topology)
    return mdtraj_topology, protein_omm_energy

