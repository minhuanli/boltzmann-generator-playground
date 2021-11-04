import matplotlib.pyplot as plt

import numpy as np
import mdtraj
import deep_boltzmann
from deep_boltzmann.networks.training import MLTrainer, FlexibleTrainer
from deep_boltzmann.networks.invertible import invnet, EnergyInvNet, InvNet
from deep_boltzmann.openmmutils import save_latent_samples_as_trajectory
from deep_boltzmann.models.openmm import OpenMMEnergy
import mdtraj as md
from simtk import openmm, unit

import sys, os, shutil
import tensorflow as tf

# Check if it is Running on GPU
print("Tensorflow Version:", tf.__version__)
print(tf.config.list_physical_devices('GPU'))
sys.stdout.flush()

# Read-in PDB model and MD data
pdb_model = mdtraj.load_pdb('pdz3_rat_apo_fixed.pdb')
traj = mdtraj.load("implicit_traj.h5")
sim_x = traj.xyz
print(sim_x.shape)
sys.stdout.flush()

# Utility Functions
def setup_protein(pdbmodel_dir):
    """ Sets up protein Topology and Energy Model

    Returns
    -------
    top : mdtraj Topology object
        Protein topology
    energy : Energy object
        Energy model

    """
    # 4C in KIW's experiment
    INTEGRATOR_ARGS = (277*unit.kelvin, 1.0/unit.picoseconds, 2.0*unit.femtoseconds)

    from simtk.openmm import app
    pdb = app.PDBFile(pdbmodel_dir)
    forcefield = openmm.app.ForceField('amber99sb.xml', 'amber99_obc.xml') #implicit Solvent

    system = forcefield.createSystem(pdb.topology,removeCMMotion=False,
                                     nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=1.0*unit.nanometers,
                                     constraints=None, rigidWater=True)
    integrator = openmm.LangevinIntegrator(277*unit.kelvin, 1.0/unit.picoseconds, 2.0*unit.femtoseconds)
    simulation = openmm.app.Simulation(pdb.topology, system, integrator)

    protein_omm_energy = OpenMMEnergy(system,
                                   openmm.LangevinIntegrator,
                                   unit.nanometers,
                                   n_atoms=md.Topology().from_openmm(simulation.topology).n_atoms,
                                   openmm_integrator_args=INTEGRATOR_ARGS)

    mdtraj_topology = md.Topology().from_openmm(pdb.topology)
    return mdtraj_topology, protein_omm_energy

def get_indices(top, cartesian_CYS=True):
    """ Returns Cartesian and IC indices

    Returns
    -------
    cart : array
        Cartesian atom selection
    Z : array
        Z index matrix

    """
    from deep_boltzmann.models.proteins import mdtraj2Z
    cartesian = ['CA', 'C', 'N']
    cart = top.select(' '.join(["name " + s for s in cartesian]))
    if cartesian_CYS:
        Z_, _carts = mdtraj2Z(top,  cartesian="resname CYS and mass>2 and sidechain")
        Z_ = np.array(Z_)
        cart = np.sort(np.concatenate((cart,_carts)))
    else:
        Z_ = np.array(mdtraj2Z(top))
    return cart, Z_

def train_ML(bg, xtrain, epochs, batch_sizes, prefix):
    trainer_ML = MLTrainer(bg, lr=0.001)
    for batch_size in batch_sizes:
        trainer_ML.train(xtrain, epochs=epochs, batch_size=batch_size)
        bg.save(prefix+"_model.pkl")
        np.save(prefix+'_loss.npy', np.array(trainer_ML.loss_train))
    return trainer_ML
        
def train_KL(bg, xtrain, epochs, high_energies, w_KLs, prefix,
             stage=0, rc_dims=None, rc_func=None, rc_min=None, 
             rc_max=None, w_RC=0.,w_L2_angle=0.,
             loss_track=[]):            
    for current_stage in range(stage, len(epochs)):
        print('-----------------------')
        print(high_energies[current_stage], w_KLs[current_stage])
        sys.stdout.flush()
        flextrainer = FlexibleTrainer(bg, lr=0.0001, batch_size=2000,
                                      high_energy=high_energies[current_stage], max_energy=1e20,
                                      w_KL=w_KLs[current_stage], w_ML=1, weigh_ML=False, w_RC=w_RC,
                                      rc_func=rc_func, rc_min=np.array(rc_min), rc_max=np.array(rc_max),
                                      w_L2_angle=w_L2_angle, 
                                      rc_dims=rc_dims)
        flextrainer.train(xtrain, epochs=epochs[current_stage])
        loss_track.extend(flextrainer.loss_train)

        # Analyze
        samples_z = np.random.randn(2000, bg.dim)
        samples_x = bg.Tzx.predict(samples_z)
        samples_e = bg.energy_model.energy(samples_x)
        energy_violations = [np.count_nonzero(samples_e > E) for E in high_energies]
        print('Energy violations:')
        for E, V in zip(high_energies, energy_violations):
            print(V, '\t>\t', E)
        sys.stdout.flush()

        # SAVE
        bg.save(prefix+'_model.pkl')
        np.save(prefix+'_loss.npy', np.array(loss_track))
        saveconfig = {}
        saveconfig['stage'] = current_stage
        #np.savez_compressed('config_save.npz', **saveconfig)
        print('Intermediate result saved')
        sys.stdout.flush()
    return loss_track

# Set up a BG network
top, mm_pdz = setup_protein('pdz3_rat_apo_fixed.pdb')

# Superimpose frames and reshuffle 
nframes = sim_x.shape[0]
dim = sim_x.shape[1]*sim_x.shape[2]
sim_x_traj = mdtraj.Trajectory(sim_x.reshape((nframes, int(dim/3), 3)), top)
sim_x_traj = sim_x_traj.superpose(sim_x_traj[0], atom_indices=top.select('backbone'))
sim_x = sim_x_traj.xyz.reshape((nframes, -1))
np.random.shuffle(sim_x)

# Get indices for dimensions represented by cartesian and internal coordinates respectively
CartIndices, ZIndices = get_indices(top, cartesian_CYS=False)

# Construct Boltzmann Generator object 
bg = invnet(dim, 'R'*8, energy_model=mm_pdz,
            ic=np.asarray(ZIndices,dtype=np.int), ic_cart=np.asarray(CartIndices,dtype=np.int), ic_norm=sim_x,
            nl_layers=4, nl_hidden=[256, 128, 256], nl_activation='relu', nl_activation_scale='tanh')
print('BG constructed\n')
sys.stdout.flush()

# Perform ML training -- 4 epochs with increasing batch sizes
epochs_ML = 2000
batch_sizes_ML = [256, 512, 1024, 1024, 1024]
trainer_ML = train_ML(bg, sim_x, epochs_ML, batch_sizes_ML, 'BG_ImplicitSolvent_MLtrain')
print('ML training done\n')
sys.stdout.flush()

# Perform Flexible Train
bg = EnergyInvNet.load('BG_ImplicitSolvent_MLtrain_model.pkl', mm_pdz)
saveconfig = {}
saveconfig['stage'] = 0
epochs_KL     = [  15,   15,   15,   15,   15,   15,  20,  20,  30, 50, 500]
high_energies = [1e10,  1e9,  1e8,  1e7,  1e6,  1e5,  1e5,  1e5,  5e4,  5e4,  2e4]
w_KLs         = [1e-12, 1e-6, 1e-5, 1e-4, 1e-3, 1e-3, 5e-3, 1e-3, 5e-3, 5e-3, 0.01]
loss_track = train_KL(bg,  sim_x, epochs_KL, high_energies, w_KLs, prefix="BG_ImplicitSolvent_KLtrain", stage = saveconfig['stage'],w_L2_angle=1.0, loss_track=[])





