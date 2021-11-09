import sys


from deep_boltzmann.networks import nonlinear_transform
from deep_boltzmann.util import ensure_traj
from deep_boltzmann.networks.invertible_layers import *
from deep_boltzmann.networks.invertible_coordinate_transforms import *

import deep_boltzmann.networks.losses as losses


class InvNet(object):
    def __init__(self, dim, layers, prior='normal'):
        """
        Parameters
        ----------
        dim : int
            Dimension
        layers : list
            list of invertible layers
        prior : str
            Type of prior, 'normal', 'lognormal'
        """

        """ Stack of invertible layers """
        self.dim = dim
        self.layers = layers
        self.prior = prior
        self.connect_layers()

    @classmethod
    def load(cls, filename, clear_session=True):
        """ Loads parameters into model. Careful: this clears the whole TF session!!
        """
        from deep_boltzmann.util import load_obj
        if clear_session:
            tf.keras.backend.clear_session()

        D = load_obj(filename)
        prior = D['prior']
        layerdicts = D['layers']
        layers = [eval(d['type']).from_dict(d) for d in layerdicts]
        return InvNet(D['dim'], layers, prior=prior)

    def save(self, filename):
        from deep_boltzmann.util import save_obj
        D = {}
        D['dim'] = self.dim
        D['prior'] = self.prior
        layerdicts = []
        for l in self.layers:
            d = l.to_dict()
            d['type'] = l.__class__.__name__
            layerdicts.append(d)
        D['layers'] = layerdicts
        save_obj(D, filename)

    def connect_xz(self, x):
        z = None
        for i in range(len(self.layers)):
            #print(' layer', i)
            z = self.layers[i].connect_xz(x)  # connect
            x = z  # rename output
        return z

    def connect_zx(self, z):
        x = None
        for i in range(len(self.layers)-1, -1, -1):
            #print(' layer', i)
            x = self.layers[i].connect_zx(z)  # connect
            z = x  # rename output to next input
        return x

    def connect_layers(self):
        print('Before connect')
        # X -> Z
        self.input_x = tf.keras.layers.Input(shape=(self.dim,))
        self.output_z = self.connect_xz(self.input_x)
        print('Done xz')

        # Z -> X
        self.input_z = tf.keras.layers.Input(shape=(self.dim,))
        self.output_x = self.connect_zx(self.input_z)
        print('Done zx')

        # build networks
        self.Txz = tf.keras.models.Model(
            inputs=self.input_x, outputs=self.output_z)
        self.Tzx = tf.keras.models.Model(
            inputs=self.input_z, outputs=self.output_x)

        # Full x -> z transformation with Jacobian
        log_det_Jxzs = []
        for layer in self.layers:
            if hasattr(layer, 'log_det_Jxz'):
                log_det_Jxzs.append(layer.log_det_Jxz)
        if len(log_det_Jxzs) == 0:
            self.TxzJ = None
            self.log_det_Jxz = tf.ones((self.output_z.shape[0],))
        else:
            # name of the layer with log(det(J))
            log_det_name = "log_det_Jxz"
            if len(log_det_Jxzs) == 1:
                self.log_det_Jxz = tf.keras.layers.Lambda(lambda x: x, name=log_det_name)(
                    log_det_Jxzs[0]
                )
            else:
                self.log_det_Jxz = tf.keras.layers.Add(
                    name=log_det_name)(log_det_Jxzs)
            self.TxzJ = tf.keras.Model(
                inputs=self.input_x,
                outputs=[self.output_z, self.log_det_Jxz],
                name="TxzJ"
            )
        
        # Full z -> x transformation with Jacobian
        log_det_Jzxs = []
        for layer in self.layers:
            if hasattr(layer, 'log_det_Jzx'):
                log_det_Jzxs.append(layer.log_det_Jzx)
        if len(log_det_Jzxs) == 0:
            self.TzxJ = None
            self.log_det_Jzx = tf.ones((self.output_x.shape[0],))
        else:
            log_det_name = "log_det_Jzx"    # name of the layer with log(det(J))
            if len(log_det_Jzxs) == 1:
                self.log_det_Jzx = tf.keras.layers.Lambda(lambda x: x, name=log_det_name)(
                    log_det_Jzxs[0]
                )
            else:
                self.log_det_Jzx = tf.keras.layers.Add(name=log_det_name)(log_det_Jzxs)
            self.TzxJ = tf.keras.Model(
                inputs=self.input_z,
                outputs=[self.output_x, self.log_det_Jzx],
                name="TzxJ"
            )

    def transform_xz(self, x):
        return self.Txz.predict(ensure_traj(x))

    def transform_xzJ(self, x):
        x = ensure_traj(x)
        z, J = self.TxzJ.predict(x)
        return z, J[:, 0]

    def transform_zx(self, z):
        return self.Tzx.predict(ensure_traj(z))

    def transform_zxJ(self, z):
        z = ensure_traj(z)
        x, J = self.TzxJ.predict(z)
        return x, J[:, 0]

    def energy_z(self, z, temperature=1.0):
        if self.prior == 'normal':
            E = self.dim * np.log(np.sqrt(temperature)) + \
                np.sum(z**2 / (2*temperature), axis=1)
        elif self.prior == 'lognormal':
            sample_z_normal = np.log(z)
            E = np.sum(sample_z_normal**2 / (2*temperature),
                       axis=1) + np.sum(sample_z_normal, axis=1)
        elif self.prior == 'cauchy':
            E = np.sum(np.log(1 + (z/temperature)**2), axis=1)
        return E

    def sample_z(self, std=1.0, nsample=100000, return_energy=False):
        """ Samples from prior distribution in z and produces generated x configurations
        Parameters:
        -----------
        temperature : float
            Relative temperature. Equal to the variance of the isotropic Gaussian sampled in z-space.
        nsample : int
            Number of samples
        Returns:
        --------
        sample_z : array
            Samples in z-space
        energy_z : array
            Energies of z samples (optional)
        """
        sample_z = None
        energy_z = None
        if self.prior == 'normal':
            sample_z = std * np.random.randn(nsample, self.dim)
        elif self.prior == 'lognormal':
            sample_z_normal = std * np.random.randn(nsample, self.dim)
            sample_z = np.exp(sample_z_normal)
        elif self.prior == 'cauchy':
            from scipy.stats import cauchy
            sample_z = cauchy(loc=0, scale=std **
                              2).rvs(size=(nsample, self.dim))
        else:
            raise NotImplementedError(
                'Sampling for prior ' + self.prior + ' is not implemented.')

        if return_energy:
            energy_z = self.energy_z(sample_z)
            return sample_z, energy_z
        else:
            return sample_z

class EnergyInvNet(InvNet):
    def __init__(self, energy_model, layers, prior='normal'):
        """ Invertible net where we have an energy function that defines p(x) """
        for attribute in ("dim", "energy", "energy_tf"):
            if not hasattr(energy_model, attribute):
                raise AttributeError(
                    f"Provided energy model does not have attribute '{attribute}'"
                )
        self.energy_model = energy_model
        super().__init__(energy_model.dim, layers, prior=prior)

    @classmethod
    def load(cls, filename, energy_model, clear_session=True):
        """ Loads parameters into model. Careful: this clears the whole TF session!!
        """
        from deep_boltzmann.util import load_obj
        if clear_session:
            tf.keras.backend.clear_session()
        D = load_obj(filename)
        prior = D['prior']
        layerdicts = D['layers']
        layers = [eval(d['type']).from_dict(d) for d in layerdicts]
        return EnergyInvNet(energy_model, layers, prior=prior)

    def sample(self, std=1.0, temperature=1.0, nsample=100000):
        """ Samples from prior distribution in x and produces generated x configurations

        Parameters:
        -----------
        temperature : float
            Relative temperature. Equal to the variance of the isotropic Gaussian sampled in z-space.

        nsample : int
            Number of samples

        Returns:
        --------
        sample_z : array
            Samples in z-space

        sample_x : array
            Samples in x-space

        energy_z : array
            Energies of z samples

        energy_x : array
            Energies of x samples

        log_w : array
            Log weight of samples
        """

        sample_z, energy_z = self.sample_z(
            std=std, nsample=nsample, return_energy=True)
        sample_x, Jzx = self.transform_zxJ(sample_z)
        energy_x = self.energy_model.energy(sample_x) / temperature
        logw = -energy_x + energy_z + Jzx

        return sample_z, sample_x, energy_z, energy_x, logw


def invnet(dim, layer_types, energy_model=None, channels=None,
           nl_layers=2, nl_hidden=100, nl_layers_scale=None, nl_hidden_scale=None,
           nl_activation='relu', nl_activation_scale='tanh', scale=None, prior='normal',
           permute_atomwise=False,
           whiten=None, whiten_keepdims=None,
           ic=None, ic_cart=None, ic_norm=None, torsion_cut=None, ic_jacobian_regularizer=1e-10,
           rg_splitfrac=0.5,
           **layer_args):
    """
    layer_types : str
        String describing the sequence of layers. Usage:
            N NICER layer
            n NICER layer, share parameters with last layer
            R RealNVP layer
            r RealNVP layer, share parameters with last layer
            S Scaling layer
            W Whiten layer
            P Permute layer
            Z Split dimensions off to latent space, leads to a merge and 3-way split.
        Splitting and merging layers will be added automatically

    energy_model : Energy model class
        Class with energy() and dim

    channels : array or None
        Assignment of dimensions to channels (0/1 array of length ndim)

    nl_layers : int
        Number of hidden layers in the nonlinear transformations

    nl_hidden : int
        Number of hidden units in each nonlinear layer

    nl_activation : str
        Hidden-neuron activation functions used in the nonlinear layers

    nl_activation_scale : str
        Hidden-neuron activation functions used in scaling networks. If None, nl_activation will be used.

    scale : None or float
        If a scaling layer is used, fix the scale to this number. If None, scaling layers are trainable

    prior : str
        Form of the prior distribution

    whiten : None or array
        If not None, compute a whitening transformation with respect togiven coordinates

    whiten_keepdims : None or int
        Number of largest-variance dimensions to keep after whitening.

    ic : None or array
        If not None, compute internal coordinates using this Z index matrix. Do not mix with whitening.

    ic_cart : None or array
        If not None, use cartesian coordinates and whitening for these atoms.

    ic_norm : None or array
        If not None, these x coordinates will be used to compute the IC mean and std. These will be used
        for normalization

    torsion_cut : None or aray
        If given defines where the torsions are cut

    rg_splitfrac : float
        Splitting fraction for Z layers
    """

    # fix channels
    channels, indices_split, indices_merge = split_merge_indices(
        dim, nchannels=2, channels=channels)

    # augment layer types with split and merge layers
    split = False
    tmp = ''
    if whiten is not None:
        tmp += 'W'

    if ic is not None:
        tmp += 'I'

    for ltype in layer_types:
        if (ltype == 'S' or ltype == 'P') and split:
            tmp += '>'
            split = False
        if (ltype == 'N' or ltype == 'R') and not split:
            tmp += '<'
            split = True
        tmp += ltype
    if split:
        tmp += '>'
    layer_types = tmp
    print(layer_types)
    # prepare layers

    layers = []
    if nl_activation_scale is None:
        nl_activation_scale = nl_activation
    if nl_layers_scale is None:
        nl_layers_scale = nl_layers
    if nl_hidden_scale is None:
        nl_hidden_scale = nl_hidden

    # number of dimensions left in the signal. The remaining dimensions are going to latent space directly
    dim_L = dim
    dim_R = 0
    dim_Z = 0

    # translate and scale layers
    T1 = None
    T2 = None
    S1 = None
    S2 = None
    for ltype in layer_types:
        print(ltype, dim_L, dim_R, dim_Z)
        if ltype == '<':
            if dim_R > 0:
                raise RuntimeError('Already split. Cannot invoke split layer.')

            channels_cur = np.concatenate(
                [channels[:dim_L], np.tile([2], dim_Z)])
            dim_L = np.count_nonzero(channels_cur == 0)
            dim_R = np.count_nonzero(channels_cur == 1)
            layers.append(SplitChannels(dim, channels=channels_cur))

        elif ltype == '>':
            if dim_R == 0:
                raise RuntimeError('Not split. Cannot invoke merge layer.')
            channels_cur = np.concatenate(
                [channels[:(dim_L+dim_R)], np.tile([2], dim_Z)])
            dim_L += dim_R
            dim_R = 0
            layers.append(MergeChannels(dim, channels=channels_cur))

        elif ltype == 'P':
            if permute_atomwise:
                order_atomwise = np.arange(dim).reshape((dim//3, 3))
                permut_ = np.random.choice(dim//3, dim//3, replace=False)
                layers.append(
                    Permute(dim, order=order_atomwise[permut_, :].flatten()))
            else:
                layers.append(Permute(dim))

        elif ltype == 'N':
            if dim_R == 0:
                raise RuntimeError('Not split. Cannot invoke NICE layer.')
            T1 = nonlinear_transform(dim_R, nlayers=nl_layers, nhidden=nl_hidden,
                                     activation=nl_activation, **layer_args)
            T2 = nonlinear_transform(dim_L, nlayers=nl_layers, nhidden=nl_hidden,
                                     activation=nl_activation, **layer_args)
            layers.append(NICER([T1, T2]))

        elif ltype == 'n':
            if dim_R == 0:
                raise RuntimeError('Not split. Cannot invoke NICE layer.')
            layers.append(NICER([T1, T2]))

        elif ltype == 'R':
            if dim_R == 0:
                raise RuntimeError('Not split. Cannot invoke RealNVP layer.')
            S1 = nonlinear_transform(dim_R, nlayers=nl_layers_scale, nhidden=nl_hidden_scale,
                                     activation=nl_activation_scale, init_outputs=0, **layer_args)
            T1 = nonlinear_transform(dim_R, nlayers=nl_layers, nhidden=nl_hidden,
                                     activation=nl_activation, **layer_args)
            S2 = nonlinear_transform(dim_L, nlayers=nl_layers_scale, nhidden=nl_hidden_scale,
                                     activation=nl_activation_scale, init_outputs=0, **layer_args)
            T2 = nonlinear_transform(dim_L, nlayers=nl_layers, nhidden=nl_hidden,
                                     activation=nl_activation, **layer_args)
            layers.append(RealNVP([S1, T1, S2, T2]))

        elif ltype == 'r':
            if dim_R == 0:
                raise RuntimeError('Not split. Cannot invoke RealNVP layer.')
            layers.append(RealNVP([S1, T1, S2, T2]))

        elif ltype == 'S':
            if dim_R > 0:
                raise RuntimeError(
                    'Not merged. Cannot invoke constant scaling layer.')
            # scaling layer
            if scale is None:
                scaling_factors = None
            else:
                scaling_factors = scale * np.ones((1, dim))
            layers.append(
                Scaling(dim, scaling_factors=scaling_factors, trainable=(scale is None)))

        elif ltype == 'I':
            if dim_R > 0:
                raise RuntimeError('Already split. Cannot invoke IC layer.')
            dim_L = dim_L - 6
            dim_R = 0
            dim_Z = 6
            if ic_cart is None:
                layer = InternalCoordinatesTransformation(
                    ic, Xnorm=ic_norm, torsion_cut=torsion_cut)
            else:
                layer = MixedCoordinatesTransformation(
                    ic_cart, ic, X0=ic_norm, torsion_cut=torsion_cut, jacobian_regularizer=ic_jacobian_regularizer)
            layers.append(layer)

        elif ltype == 'W':
            if dim_R > 0:
                raise RuntimeError(
                    'Not merged. Cannot invoke whitening layer.')
            W = FixedWhiten(whiten, keepdims=whiten_keepdims)
            dim_L = W.keepdims
            dim_Z = dim-W.keepdims
            layers.append(W)

        elif ltype == 'Z':
            if dim_R == 0:
                raise RuntimeError('Not split. Cannot invoke Z layer.')
            if dim_L + dim_R <= 1:  # nothing left to split, so we ignore this layer and move on
                break
            # merge the current pattern
            channels_cur = np.concatenate(
                [channels[:(dim_L+dim_R)], np.tile([2], dim_Z)])
            dim_L += dim_R
            dim_R = 0
            layers.append(MergeChannels(dim, channels=channels_cur))

            # 3-way split
            split_to_z = int((dim_L + dim_R) * rg_splitfrac)
            split_to_z = max(1, split_to_z)  # split at least 1 dimension
            dim_Z += split_to_z
            dim_L -= split_to_z
            channels_cur = np.concatenate(
                [channels[:dim_L], np.tile([2], dim_Z)])
            dim_L = np.count_nonzero(channels_cur == 0)
            dim_R = np.count_nonzero(channels_cur == 1)
            layers.append(SplitChannels(dim, channels=channels_cur))

    if energy_model is None:
        return InvNet(dim, layers, prior=prior)

    else:
        return EnergyInvNet(energy_model, layers, prior=prior)
