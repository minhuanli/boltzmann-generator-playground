import tensorflow as tf
import numpy as np
from deep_boltzmann.util import linlogcut
from DeepRefine.utils import construct_SO3, r_factor
from DeepRefine.Fmodel import DWF_aniso



class MLlossNormal:
    """
    The maximum-likelihood loss (or forward KL loss) of the MD data
    U(Txz(x)) - log_det_Jxz, U(z) = (0.5/std**2) * \sum z**2
    """
    def __init__(self, std_z=1.0):
        self.std_z = std_z
    
    def __call__(self, args):
        output_z, log_det_Jxz = args[0], args[1]
        energy_z = (0.5/self.std_z**2) * tf.reduce_sum(output_z**2, axis=1, keepdims=True)
        return tf.reduce_mean(energy_z - log_det_Jxz)

class KLloss:
    """
    The reverse KL loss of the normalizing flow
    U(F(zx(z))) - log_det_Jzx, U(x) is given by the energy function
    """
    def __init__(self, energy_function, high_energy, max_energy, temperature=1.0):
        '''
        energy_function: bg.energy_model.energy_tf
        '''
        self.energy_function = energy_function
        self.Ehigh = high_energy
        self.Emax = max_energy
        self.temperature = temperature
    
    def __call__(self, args):
        output_x, log_det_Jzx = args[0], args[1] 
        E = self.energy_function(output_x) / self.temperature
        Ereg = linlogcut(E, self.Ehigh, self.Emax, tf=True)
        return tf.reduce_mean(Ereg - log_det_Jzx)

class XstalLoss(tf.keras.models.Model):
    '''
    args: sfcalculator, checkpoint_file
    '''
    def __init__(self, sfcalculator, checkpoint_file):
        super().__init__()
        self.sfcalculator = sfcalculator
        self.Fo = sfcalculator.Fo
        self.SigF = sfcalculator.SigF
        self.rwork_id = sfcalculator.rwork_id
        self.rfree_id = sfcalculator.rfree_id
        self.dr2_complex_tensor = tf.constant(sfcalculator.dr2HKL_array, dtype=tf.complex64)
        self.reciprocal_cell_paras = sfcalculator.reciprocal_cell_paras
        self.HKL_array = sfcalculator.HKL_array
        self.load_args(checkpoint_file)
        
    def save_args(self, prefix):
        record_dict = {}
        for arg in self.trainable_weights:
            record_dict[arg.name.split(":")[0]] = arg.numpy()
        np.save(prefix+"_xstalargs_checkpoint.npy", record_dict)
        
    def load_args(self, file_path):
        
        checkpoint = np.load(file_path, allow_pickle=True)
        self.v1 = tf.Variable(
            checkpoint.item()["rot_v1"], name='rot_v1', dtype=tf.float32)
        self.v2 = tf.Variable(
            checkpoint.item()["rot_v2"], name='rot_v2', dtype=tf.float32)
        self.trans_vec = tf.Variable(
            checkpoint.item()["trans_vec"], name='trans_vec', dtype=tf.float32)
        self.kall = tf.Variable(
            checkpoint.item()["kall"], name='kall', dtype=tf.float32)
        self.ksol = tf.Variable(
            checkpoint.item()["ksol"], name='ksol', dtype=tf.float32)
        self.bsol = tf.Variable(
            checkpoint.item()["bsol"], name='bsol', dtype=tf.float32)

        try:
            self.log_kaniso_diag = tf.Variable(
                checkpoint.item()["log_kaniso_diag"], name='log_kaniso_diag', dtype=tf.float32)
            self.k_aniso_nondiag = tf.Variable(
                checkpoint.item()["k_aniso_nondiag"], name='k_aniso_nondiag', dtype=tf.float32)
        except:
            print("The k aniso is not correctly parameterized in the checkpoint!")

    def x2Fcalc(self, output_x, batchsize, rotate = True):

        if rotate: 
            R = construct_SO3(self.v1,self.v2)
            rot_x_samples = tf.matmul(tf.reshape(output_x, [-1, self.sfcalculator.n_atoms, 3]), R) + self.trans_vec
        else:
            rot_x_samples = tf.reshape(output_x, [-1, self.sfcalculator.n_atoms, 3])
        Fprotein = self.sfcalculator.Calc_Fprotein_batch(atoms_position_batch=rot_x_samples*10, batchsize=batchsize)
        Fmask= self.sfcalculator.Calc_Fsolvent_batch(batchsize=batchsize)
        scaled_Fmask = tf.complex(self.ksol, 0.0)*Fmask*tf.exp(-tf.complex(self.bsol, 0.0)*self.dr2_complex_tensor/4.)
        kaniso = tf.concat((tf.exp(self.log_kaniso_diag), self.k_aniso_nondiag), axis=0)
        F_total = tf.complex(self.kall, 0.0)*tf.complex(DWF_aniso(kaniso[None, ...], self.reciprocal_cell_paras, self.HKL_array)[0], 0.0)*(Fprotein+scaled_Fmask)
        return F_total
        
    def __call__(self, output_x, batchsize, rotate=True):
        F_total = self.x2Fcalc(output_x, batchsize, rotate=rotate) # Shape [N_batch, N_HKLs]
        F_model = tf.abs(tf.reduce_mean(F_total, axis=0)) # Shape [N_HKLs,]
        Z_work = tf.gather(self.Fo, self.rwork_id) - tf.gather(F_model,self.rwork_id) # Shape [N_HKLs,]
        Z_free = tf.gather(self.Fo, self.rfree_id) - tf.gather(F_model,self.rfree_id)
        loss_work = tf.reduce_sum(Z_work**2) 
        loss_free = tf.reduce_sum(Z_free**2)
        r_work, r_free = r_factor(self.Fo, F_model, self.rwork_id, self.rfree_id)    
        return loss_work, loss_free, r_work, r_free

def loss_L2_angle_penalization(bg):
    losses = []
    for layer in bg.layers:
        if hasattr(layer, "angle_loss"):
            losses.append(layer.angle_loss)
    loss = sum(losses)
    return tf.reduce_mean(loss)



