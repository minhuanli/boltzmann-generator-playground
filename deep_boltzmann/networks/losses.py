import tensorflow as tf
import numpy as np
from deep_boltzmann.util import linlogcut


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
        return energy_z - log_det_Jxz 

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
        return Ereg - log_det_Jzx

def loss_L2_angle_penalization(bg):
    losses = []
    for layer in bg.layers:
        if hasattr(layer, "angle_loss"):
            losses.append(layer.angle_loss)
    loss = sum(losses)
    return loss[..., None]



