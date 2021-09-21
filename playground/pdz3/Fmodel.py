'''
Calculate Structural Factor from an atomic model: F_model = k_total * (F_calc + k_mask * F_mask)

Note:
1. We use direct summation for the F_calc
2. Now we only include f_0, no f' or f'', so no anomalous scattering

Written in Tensorflow 2.0 to fit in the general pipeline. Also necessary for efficient optimization

May, 2021
'''

__author__ = "Minhuan Li"
__email__ = "minhuanli@g.harvard.edu"

import gemmi
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import reciprocalspaceship as rs


ccp4_hkl_asu = [
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 7, 6, 7, 6, 7, 7, 7,
    6, 7, 6, 7, 7, 6, 6, 7, 7, 7, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9
]

asu_cases = {
    0: lambda h, k, l: (l > 0) | ((l == 0) & ((h > 0) | ((h == 0) & (k >= 0)))),
    1: lambda h, k, l: (k >= 0) & ((l > 0) | ((l == 0) & (h >= 0))),
    2: lambda h, k, l: (h >= 0) & (k >= 0) & (l >= 0),
    3: lambda h, k, l: (l >= 0) & (((h >= 0) & (k > 0)) | ((h == 0) & (k == 0))),
    4: lambda h, k, l: (h >= k) & (k >= 0) & (l >= 0),
    5: lambda h, k, l: ((h >= 0) & (k > 0)) | ((h == 0) & (k == 0) & (l >= 0)),
    6: lambda h, k, l: (h >= k) & (k >= 0) & ((k > 0) | (l >= 0)),
    7: lambda h, k, l: (h >= k) & (k >= 0) & ((h > k) | (l >= 0)),
    8: lambda h, k, l: (h >= 0) & (((l >= h) & (k > h)) | ((l == h) & (k == h))),
    9: lambda h, k, l: (k >= l) & (l >= h) & (h >= 0),
}


def to_pos_idx(idx, x):
    '''
    Utils function to make negative slice index postive, becasue tensorflow is stupid.
    For example, for a tensor with size [160,160,160], index [-1,-1,-1] means index [159,159,159],
    But tensorflow doesn't support negative index in the function tf.scatter_update or tf.gather, so
    we have to do this convert manually.

    Parameters:
    -----------
    idx: array-like stacked nd indices
    x: The tensor you want to apply the index to.

    Return:
    -------
    The tensor with same shape as the idx array, but now all elements are postive
    '''
    idx_p = tf.convert_to_tensor(idx)
    s = tf.shape(x)[:tf.size(idx_p)]
    idx_p = tf.where(idx_p < 0, s + idx_p, idx_p)
    return idx_p


def diff_array(a, b):
    '''
    Return the elements in a but not in b, when a and b are array-like object

    Parameters
    ----------
    a: array-like
       Can be N Dimensional

    b: array-like

    return_diff: boolean
       return the set difference or not

    Return
    ------
    Difference Elements
    '''
    tuplelist_a = list(map(tuple, a))
    tuplelist_b = list(map(tuple, b))
    set_a = set(tuplelist_a)
    set_b = set(tuplelist_b)
    return set_a - set_b


def reciprocal_asu(cell, spacegroup, dmin):
    '''
    Generate the miller indices of the reflections in the reciprocal ASU, with unit cell, space group info and the resolution cutoff

    I vectorize the generate_reciprocal_asu function from reciprocal spaceship:
    https://github.com/Hekstra-Lab/reciprocalspaceship/blob/main/reciprocalspaceship/utils/asu.py

    Parameters:
    -----------
    cell: gemmi.UnitCell
        A gemmi cell object
    spacegroup: gemmi.SpaceGroup
        A gemmi spacegroup object
    dmin: float
        Maximum resolution of the data in Å
    '''
    p1_hkl = rs.utils.generate_reciprocal_cell(cell, dmin)
    hkl = p1_hkl[~rs.utils.is_absent(p1_hkl, spacegroup)]
    group_ops = spacegroup.operations()
    num_ops = len(group_ops)
    basis_op = spacegroup.basisop
    op_list = []
    for i, op in enumerate(group_ops):
        op_list.append(np.array(op.rot)/op.DEN)
        op_list.append(-np.array(op.rot)/op.DEN)
    Rot_stack = np.stack(op_list).astype(int)
    Rot_stack_b = np.transpose(
        np.matmul(Rot_stack, np.array(basis_op.rot)/basis_op.DEN), [2, 1, 0])
    Rot_stack = np.transpose(Rot_stack, [2, 1, 0])
    R = np.transpose(np.matmul(hkl, Rot_stack), [1, 0, 2])
    R_b = np.matmul(hkl, Rot_stack_b)
    asu_case_index = ccp4_hkl_asu[spacegroup.number-1]
    in_asu = asu_cases[asu_case_index]
    idx = in_asu(*R_b)
    idx[np.cumsum(idx, -1) > 1] = False
    H_asu = R.swapaxes(1, 2)[idx].astype(int)
    hasu = np.unique(H_asu, axis=0)
    return hasu


def expand_to_p1(spacegroup, Hasu_array, Fasu_tensor):
    '''
    Expand the reciprocal ASU array to a complete p1 unit cell, with phase shift on the Complex Structure Factor
    In a fully differentiable manner (to Fasu_tensor), with tensorflow

    Parameters:
    -----------
    spacegroup: gemmi.SpaceGroup
        A gemmi spacegroup object

    Hasu_array: np.int32 array
        The HKL list of reciprocal ASU

    Fasu_tensor: tf.complex64 tensor
        Corresponding structural factor tensor

    Return:
    -------
    Hp1_array, Fp1_tensor
        HKL list in p1 unit cell and corresponding complex structural factor tensor
    '''
    groupops = spacegroup.operations()
    allops = [op for op in groupops]

    Fp1_tensor = Fasu_tensor
    len_asu = len(Hasu_array)

    ds = pd.DataFrame()
    ds["H"] = Hasu_array[:, 0]
    ds["K"] = Hasu_array[:, 1]
    ds["L"] = Hasu_array[:, 2]
    ds["index"] = np.arange(len_asu)

    for i, op in enumerate(allops):
        if i == 0:
            continue
        rot_temp = np.array(op.rot)/op.DEN
        tran_temp = tf.constant(np.array(op.tran)/op.DEN, dtype=tf.float32)
        H_temp = np.matmul(Hasu_array, rot_temp).astype(np.int32)
        ds_temp = pd.DataFrame()
        ds_temp["H"] = H_temp[:, 0]
        ds_temp["K"] = H_temp[:, 1]
        ds_temp["L"] = H_temp[:, 2]
        ds_temp["index"] = np.arange(len_asu)+len_asu*i
        # exp(-2*pi*j*h*T)
        Hasu_tensor = tf.constant(Hasu_array, dtype=tf.float32)
        phaseshift_temp = tf.exp(tf.complex(
            0., -2*np.pi*tf.tensordot(Hasu_tensor, tran_temp, axes=1)))
        Fcalc_temp = Fasu_tensor * phaseshift_temp
        ds = ds.append(ds_temp)
        Fp1_tensor = tf.concat((Fp1_tensor, Fcalc_temp), axis=0)

    # Friedel Pair
    ds_friedel = ds.copy()
    ds_friedel["H"] = -ds_friedel["H"]
    ds_friedel["K"] = -ds_friedel["K"]
    ds_friedel["L"] = -ds_friedel["L"]
    ds_friedel["index"] = ds_friedel["index"] + len(ds)
    F_friedel_tensor = tf.math.conj(Fp1_tensor)

    # Combine
    ds = ds.append(ds_friedel)
    Fp1_tensor = tf.concat((Fp1_tensor, F_friedel_tensor), axis=0)

    ds = ds.drop_duplicates(subset=["H", "K", "L"])

    HKL_1 = ds[["H", "K", "L"]].values
    idx_1 = tf.constant(ds["index"].values, dtype=tf.int32)
    Fp1_tensor1 = tf.gather(Fp1_tensor, idx_1)
    in_asu = asu_cases[0]  # p1 symmetry
    idx_2 = in_asu(*HKL_1.T)
    HKL_2 = HKL_1[idx_2]
    Fp1_tensor2 = Fp1_tensor1[idx_2]
    return HKL_2, Fp1_tensor2


def reciprocal_grid(Hp1_array, Fp1_tensor, gridsize):
    '''
    Construct a reciprocal grid in reciprocal unit cell with HKL array and Structural Factor tensor
    Fully differentiable with tensorflow. It is stupid that tensorflow doesn't support dynamic assignment

    Parameters
    ----------
    Hp1_array: np.int32 array
        The HKL list in the p1 unit cell. Usually the output of expand_to_p1

    Fp1_tensor: tf.complex64 tensor
        Corresponding structural factor tensor. Usually the output of expand_to_p1

    gridsize: array-like, int
        The size of the grid you want to create, the only requirement is "big enough"

    Return:
    Reciprocal space unit cell grid, as a tf.complex64 tensor
    '''
    grid = tf.zeros(gridsize, dtype=tf.complex64)
    postive_index = to_pos_idx(Hp1_array, grid)
    grid = tf.tensor_scatter_nd_update(grid, postive_index, Fp1_tensor)
    return grid


def rsgrid2realmask(rs_grid, solvent_percent=50.0, scale=50):
    '''
    Convert reciprocal space grid to real space solvent mask grid, in a
    fully differentiable way with tensorflow.

    Parameters:
    -----------
    rs_grid: tf.complex64 tensor
        Reciprocal space unit cell grid. Usually the output of reciprocal_grid

    solvent_percent: float
        The approximate volume percentage of solvent in the system, to generate the cutoff

    scale: int/float
        The scale used in sigmoid function, to make the distribution binary

    Return:
    -------
    tf.float32 tensor
    The solvent mask grid in real space, solvent voxels have value close to 1, while protein voxels have value close to 0.
    '''
    real_grid = tf.math.real(tf.signal.fft3d(rs_grid))
    real_grid_norm = (real_grid - tf.reduce_mean(real_grid)) / \
        tf.math.reduce_std(real_grid)
    CUTOFF = tfp.stats.percentile(real_grid_norm, 50.0)
    real_grid_mask = tf.sigmoid((CUTOFF-real_grid_norm)*scale)
    return real_grid_mask


def realmask2Fmask(real_grid_mask, H_array):
    '''
    Convert real space solvent mask grid to mask structural factor, in a fully differentiable
    manner, with tensorflow.

    Parameters:
    -----------
    real_grid_mask: tf.float32 tensor
        The solvent massk grid in real space unit cell. Usually the output of rsgrid2realmask

    H_array: array-like, int
        The HKL list we are interested in to assign structural factors

    Return:
    -------
    tf.complex64 tensor
    Solvent mask structural factor corresponding to the HKL list in H_array
    '''
    Fmask_grid = tf.math.conj(tf.signal.fft3d(tf.complex(real_grid_mask, 0.)))
    positive_index = to_pos_idx(H_array, Fmask_grid)
    Fmask = tf.gather_nd(Fmask_grid, positive_index)
    return Fmask


def DWF_iso(b_iso, dr2_array):
    '''
    Calculate Debye_Waller Factor with Isotropic B Factor
    DWF_iso = exp(-B_iso * dr^2/4), Rupp P640, dr is dhkl in reciprocal space

    Parameters:
    -----------
    b_iso: float or tensor float32
        Isotropic B factor

    dr2_array: numpy 1D array or 1D tensor
        Reciprocal d*(hkl)^2 array, corresponding to the HKL_array

    Return:
    -------
    A 1D float32 tensor with DWF corresponding to different HKL
    '''
    return tf.cast(tf.exp(-b_iso*dr2_array/4), dtype=tf.float32)


def DWF_aniso(b_aniso, reciprocal_cell_paras, HKL_array):
    '''
    Calculate Debye_Waller Factor with anisotropic B Factor, Rupp P641
    DWF_aniso = exp[-2 * pi^2 * (U11*h^2*ar^2 + U22*k^2*br^2 + U33*l^2cr^2
                                 + 2U12*h*k*ar*br*cos(gamma_r)
                                 + 2U13*h*l*ar*cr*cos(beta_r)
                                 + 2U23*k*l*br*cr*cos(alpha_r))]

    Parameters:
    -----------
    b_aniso: list of float or tensor float
        Anisotropic B factor U, [U11, U22, U33, U12, U13, U23]

    reciprocal_cell_paras: list of float or tensor float
        Necessary info of Reciprocal unit cell, [ar, br, cr, cos(alpha_r), cos(beta_r), cos(gamma_r)

    HKL_array: array of HKL index, [N,3]

    Return:
    -------
    A 1D float32 tensor with DWF corresponding to different HKL
    '''
    U11, U22, U33, U12, U13, U23 = b_aniso
    ar, br, cr, cos_alphar, cos_betar, cos_gammar = reciprocal_cell_paras
    h, k, l = HKL_array.T
    log_value = -2 * np.pi**2 * (U11 * h**2 * ar**2
                                 + U22 * k**2 * br**2
                                 + U33 * l**2 * cr**2
                                 + 2*U12*h*k*ar*br*cos_gammar
                                 + 2*U13*h*l*ar*cr*cos_betar
                                 + 2*U23*k*l*br*cr*cos_alphar)
    return tf.cast(tf.exp(log_value), dtype=tf.float32)


def F_protein(HKL_array, dr2_array, full_atomic_sf, atom_name, reciprocal_cell_paras,
              R_G_tensor_stack,
              T_G_tensor_stack,
              atom_pos_frac,
              atom_b_iso,
              atom_b_aniso,
              atom_occ):
    '''
    Calculate Protein Structural Factor from an atomic model
    '''
    # TODO: Assert they are all tensor
    # F_calc = sum_Gsum_j{ [f0_sj*DWF*exp(2*pi*i*(h,k,l)*(R_G*(x1,x2,x3)+T_G))]} fractional postion, Rupp's Book P279
    # G is symmetry operations of the spacegroup and j is the atoms
    # DWF is the Debye-Waller Factor, has isotropic and anisotropic version, based on the PDB file input, Rupp's Book P641
    Hasu_tensor = tf.constant(HKL_array, dtype=tf.float32)
    F_calc = 0
    for atom_str, pos_frac, b_iso, b_aniso, occ in zip(atom_name, atom_pos_frac, atom_b_iso, atom_b_aniso, atom_occ):
        if (b_aniso == [0.]*6).numpy().all():
            # Isotropic case
            DWF = DWF_iso(b_iso, dr2_array)
        else:
            # Anisotropic case
            DWF = DWF_aniso(b_aniso, reciprocal_cell_paras, HKL_array)
        magnitude_j = occ * full_atomic_sf[atom_str] * DWF  # Shape [N_HKL,]
        sym_oped_pos_frac = tf.tensordot(
            R_G_tensor_stack, pos_frac, 1) + T_G_tensor_stack  # Shape [N_ops, 3]
        phase_j = 2*tf.constant(np.pi) * \
            tf.tensordot(Hasu_tensor, tf.transpose(
                sym_oped_pos_frac), 1)  # Shape [N_HKL, N_ops]
        cos_phase_j = tf.reduce_sum(tf.cos(phase_j), axis=1)  # Shape [N_HKL,]
        sin_phase_j = tf.reduce_sum(tf.sin(phase_j), axis=1)
        F_calc_j = tf.complex(magnitude_j*cos_phase_j,
                              magnitude_j*sin_phase_j)
        F_calc += F_calc_j
    return F_calc

class SFcalculator(object):
    '''
    A class to formalize the structural factor calculation.
    '''

    def __init__(self, PDBfile_dir, mtzfile_dir=None, dmin=None):
        # TODO: Add override option for spacegroup, cell, atoms_name, atoms_postion, atoms_Bfactor, actoms_occuracy
        '''
        Initialize with necessary reusable information, like spacegroup, unit cell info, HKL_list, et.c.

        Parameters:
        -----------
        model_dir: path, str
            path to the PDB model file, will use its unit cell info, space group info, atom name info,
            atom position info, atoms B-factor info and atoms occupancy info to initialize the instance.

        mtz_file_dir: path, str, default None
            path to the mtz_file_dir, will use the HKL list in the mtz instead, override dmin with an inference

        dmin: float, default None
            highest resolution in the map in Angstrom, to generate Miller indices in recirpocal ASU
        '''
        structure = gemmi.read_pdb(PDBfile_dir)  # gemmi.Structure object
        self.unit_cell = structure.cell  # gemmi.UnitCell object
        self.space_group = gemmi.SpaceGroup(
            structure.spacegroup_hm)  # gemmi.SpaceGroup object
        self.operations = self.space_group.operations()  # gemmi.GroupOps object
        self.R_G_tensor_stack = tf.stack([tf.constant(
            sym_op.rot, dtype=tf.float32)/sym_op.DEN for sym_op in self.operations], axis=0)
        self.T_G_tensor_stack = tf.stack([tf.constant(
            sym_op.tran, dtype=tf.float32)/sym_op.DEN for sym_op in self.operations], axis=0)

        self.reciprocal_cell = self.unit_cell.reciprocal()  # gemmi.UnitCell object
        # [ar, br, cr, cos(alpha_r), cos(beta_r), cos(gamma_r)]
        self.reciprocal_cell_paras = [self.reciprocal_cell.a,
                                      self.reciprocal_cell.b,
                                      self.reciprocal_cell.c,
                                      np.cos(np.deg2rad(
                                          self.reciprocal_cell.alpha)),
                                      np.cos(np.deg2rad(
                                          self.reciprocal_cell.beta)),
                                      np.cos(np.deg2rad(
                                          self.reciprocal_cell.gamma))
                                      ]
        # Generate ASU HKL array and Corresponding d*^2 array
        if mtzfile_dir:
            mtz_reference = gemmi.read_mtz_file(mtzfile_dir)
            # HKL array from the reference mtz file, [N,3]
            self.HKL_array = mtz_reference.make_miller_array()
            self.dmin = mtz_reference.resolution_high()
            self.Hasu_array = rs.utils.generate_reciprocal_asu(
                self.unit_cell, self.space_group, self.dmin)
            assert diff_array(self.HKL_array, self.Hasu_array) == set(
            ), "HKL_array should be equal or subset of the Hasu_array!"
            self.asu2HKL_index = tf.constant([np.argwhere(np.all(self.Hasu_array == hkl, axis=1))[
                                             0, 0] for hkl in self.HKL_array], dtype=tf.int32)
            # d*^2 array according to the HKL list, [N]
            self.dr2asu_array = self.unit_cell.calculate_1_d2_array(
                self.Hasu_array)
            self.dr2HKL_array = self.unit_cell.calculate_1_d2_array(
                self.HKL_array)
        else:
            if not dmin:
                raise ValueError(
                    "high_resolution dmin OR a reference mtz file should be provided!")
            else:
                self.dmin = dmin
                self.Hasu_array = rs.utils.generate_reciprocal_asu(
                    self.unit_cell, self.space_group, self.dmin)
                self.dr2asu_array = self.unit_cell.calculate_1_d2_array(
                    self.Hasu_array)
                self.HKL_array = None

        self.atom_name = []
        self.atom_pos_orth = []
        self.atom_pos_frac = []
        self.atom_b_aniso = []
        self.atom_b_iso = []
        self.atom_occ = []
        model = structure[0]  # gemmi.Model object
        for chain in model:
            for res in chain:
                for atom in res:
                    # A list of atom name like ['O','C','N','C', ...], [Nc]
                    self.atom_name.append(atom.element.name)
                    # A list of atom's Positions in orthogonal space, [Nc,3]
                    self.atom_pos_orth.append(atom.pos.tolist())
                    # A list of atom's Positions in fractional space, [Nc,3]
                    self.atom_pos_frac.append(
                        self.unit_cell.fractionalize(atom.pos).tolist())
                    # A list of anisotropic B Factor [[U11,U22,U33,U12,U13,U23],..], [Nc,6]
                    self.atom_b_aniso.append(atom.aniso.elements())
                    # A list of isotropic B Factor [B1,B2,...], [Nc]
                    self.atom_b_iso.append(atom.b_iso)
                    # A list of occupancy [P1,P2,....], [Nc]
                    self.atom_occ.append(atom.occ)

        self.atom_pos_orth = tf.convert_to_tensor(self.atom_pos_orth)
        self.atom_pos_frac = tf.convert_to_tensor(self.atom_pos_frac)
        self.atom_b_aniso = tf.convert_to_tensor(self.atom_b_aniso)
        self.atom_b_iso = tf.convert_to_tensor(self.atom_b_iso)
        self.atom_occ = tf.convert_to_tensor(self.atom_occ)

        self.unique_atom = list(set(self.atom_name))
        self.orth2frac_tensor = tf.constant(
            self.unit_cell.fractionalization_matrix.tolist())

        # A dictionary of atomic structural factor f0_sj of different atom types at different HKL Rupp's Book P280
        # f0_sj = [sum_{i=1}^4 {a_ij*exp(-b_ij* d*^2/4)} ] + c_j
        self.full_atomic_sf_asu = {}
        for atom_type in self.unique_atom:
            element = gemmi.Element(atom_type)
            self.full_atomic_sf_asu[atom_type] = tf.constant([
                element.it92.calculate_sf(dr2/4.) for dr2 in self.dr2asu_array], dtype=tf.float32)

    def Calc_Fprotein(self, atoms_position_tensor=None,
                      atoms_biso_tensor=None,
                      atoms_baniso_tensor=None,
                      atoms_occ_tensor=None):
        # Read and tensorfy necessary inforamtion
        if atoms_position_tensor:
            assert len(atoms_position_tensor) == len(
                self.atom_name), "Atoms in atoms_positions_tensor should be consistent with atom names in PDB model!"
            self.atom_pos_frac = tf.tensordot(
                atoms_position_tensor, tf.transpose(self.orth2frac_tensor), 1)

        if atoms_baniso_tensor:
            assert len(atoms_baniso_tensor) == len(
                self.atom_name), "Atoms in atoms_baniso_tensor should be consistent with atom names in PDB model!"
            self.atom_b_aniso = atoms_baniso_tensor

        if atoms_biso_tensor:
            assert len(atoms_biso_tensor) == len(
                self.atom_name), "Atoms in atoms_biso_tensor should be consistent with atom names in PDB model!"
            self.atom_b_iso = atoms_biso_tensor

        if atoms_occ_tensor:
            assert len(atoms_occ_tensor) == len(
                self.atom_name), "Atoms in atoms_occ_tensor should be consistent with atom names in PDB model!"
            self.atom_occ = atoms_occ_tensor

        self.Fprotein_asu = F_protein(self.Hasu_array, self.dr2asu_array,
                                      self.full_atomic_sf_asu,
                                      self.atom_name,
                                      self.reciprocal_cell_paras,
                                      self.R_G_tensor_stack, self.T_G_tensor_stack,
                                      self.atom_pos_frac,
                                      self.atom_b_iso, self.atom_b_aniso, self.atom_occ)
        if self.HKL_array:
            self.Fprotein_HKL = tf.gather(
                self.Fprotein_asu, self.asu2HKL_index)
            return self.Fprotein_HKL
        else:
            return self.Fprotein_asu

    # TODO Check if Calc_Fprotein has been run
    def Calc_Fsolvent(self, gridsize=[160, 160, 160]):
        Hp1_array, Fp1_tensor = expand_to_p1(
            self.space_group, self.Hasu_array, self.Fprotein_asu)
        rs_grid = reciprocal_grid(Hp1_array, Fp1_tensor, gridsize)
        self.real_grid_mask = rsgrid2realmask(rs_grid)
        if self.HKL_array:
            self.Fmask_HKL = realmask2Fmask(
                self.real_grid_mask, self.HKL_array)
            return self.Fmask_HKL
        else:
            self.Fmask_asu = realmask2Fmask(
                self.real_grid_mask, self.Hasu_array)
            return self.Fmask_asu

    # TODO add anisotropic overall scale
    def Calc_Ftotal(self, kall=tf.constant(1.0), ksol=tf.constant(0.35), bsol=tf.constant(50.0)):
        if self.HKL_array:
            dr2_complex_tensor = tf.constant(
                self.dr2HKL_array, dtype=tf.complex64)
            scaled_Fmask = tf.complex(
                ksol, 0.0)*self.Fmask_HKL*tf.exp(-tf.complex(bsol, 0.0)*dr2_complex_tensor/4.)
            self.Ftotal_HKL = tf.complex(
                kall, 0.0)*(self.Fprotein_HKL+scaled_Fmask)
            return self.Ftotal_HKL
        else:
            dr2_complex_tensor = tf.constant(
                self.dr2asu_array, dtype=tf.complex64)
            scaled_Fmask = tf.complex(
                ksol, 0.0)*self.Fmask_asu*tf.exp(-tf.complex(bsol, 0.0)*dr2_complex_tensor/4.)
            self.Ftotal_asu = tf.complex(
                kall, 0.0)*(self.Fprotein_asu+scaled_Fmask)
            return self.Ftotal_asu

    def prepare_DataSet(self, HKL_attr, F_attr):
        F_out = getattr(self, F_attr)
        HKL_out = getattr(self, HKL_attr)
        assert len(F_out) == len(
            HKL_out), "HKL and structural factor should have same length!"
        F_out_mag = tf.abs(self.F_out)
        PI_on_180 = 0.017453292519943295
        F_out_phase = tf.math.angle(F_out) / PI_on_180
        dataset = rs.DataSet(
            spacegroup=self.space_group, cell=self.unit_cell)
        dataset["H"] = HKL_out[:, 0]
        dataset["K"] = HKL_out[:, 1]
        dataset["L"] = HKL_out[:, 2]
        dataset["FMODEL"] = F_out_mag.numpy()
        dataset["PHIFMODEL"] = F_out_phase.numpy()
        dataset["FMODEL_COMPLEX"] = F_out.numpy()
        dataset.set_index(["H", "K", "L"], inplace=True)
        return dataset