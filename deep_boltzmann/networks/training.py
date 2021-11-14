import sys
import numpy as np
import tensorflow as tf
import deep_boltzmann.networks.losses as losses
import time


class MLTrainer(object):

    def __init__(self, bg, optimizer=None, lr=0.001, clipnorm=None,
                 std_z=1.0, save_test_energies=False):

        self.bg = bg
        self.save_test_energies = save_test_energies
        self.std_z = std_z

        if optimizer is None:
            if clipnorm is None:
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            else:
                self.optimizer = tf.keras.optimizers.Adam(
                    learning_rate=lr, clipnorm=clipnorm)

        self.loss_train = []
        self.loss_val = []
        if save_test_energies:
            self.energies_x_val = []
            self.energies_z_val = []

        inputs = []
        outputs = []

        # Create inputs and outputs
        inputs.append(self.bg.input_x)
        outputs.append(self.bg.output_z)
        outputs.append(self.bg.log_det_Jxz)

        # Define ML loss function
        ml_loss = tf.keras.layers.Lambda(losses.MLlossNormal(
            std_z=self.std_z), name="ML_loss_layer")([self.bg.output_z, self.bg.log_det_Jxz])

        # Construct the model
        self.ML_model = tf.keras.models.Model(
            inputs=inputs, outputs=outputs, name="ML_model")
        self.ML_model.add_loss(ml_loss)
        self.ML_model.add_metric(ml_loss, name="ML_loss")
        self.ML_model.compile(self.optimizer)

    def train(self, x_train, x_val=None, epochs=2000, batch_size=1024, verbose=1, record_time=False):

        N = x_train.shape[0]
        I = np.arange(N)

        if x_val is not None:
            Nt = x_val.shape[0]
            It = np.arange(Nt)

        @tf.function
        def steptrain(model_and_data):
            model, data = model_and_data
            return model.train_step((data,))

        @tf.function
        def steptest(model_and_data):
            model, data = model_and_data
            return model.test_step((data,))

        for e in range(epochs):
            if record_time:
                start_time = time.time()

            # sample batch
            x_batch = x_train[np.random.choice(
                I, size=batch_size, replace=True)]

            # single step train
            losses_for_this_iteration = steptrain((self.ML_model, [x_batch]))
            self.loss_train.append(float(losses_for_this_iteration["ML_loss"]))

            if record_time:
                time_this_round = round(time.time() - start_time, 3)

            # validate
            if x_val is not None:
                xval_batch = x_val[np.random.choice(
                    It, size=batch_size, replace=True)]
                val_losses_for_this_iteration = steptest(
                    (self.ML_model, [xval_batch]))
                self.loss_val.append(
                    val_losses_for_this_iteration["ML_loss"])

                if self.save_test_energies:
                    z = self.bg.sample_z(nsample=batch_size)
                    xout = self.bg.transform_zx(z)
                    self.energies_x_val.append(
                        self.bg.energy_model.energy(xout))
                    zout = self.bg.transform_xz(xval_batch)
                    self.energies_z_val.append(self.bg.energy_z(zout))

            # print
            if verbose > 0:
                str_ = 'Epoch ' + str(e) + '/' + str(epochs) + ' '
                str_ += 'ML Loss Value' + ' '
                str_ += '{:.4f}'.format(self.loss_train[-1]) + ' '
                if x_val is not None:
                    str_ += 'Testset {:.4f}'.format(self.loss_val[-1]) + ' '
                if record_time:
                    str_ += "Time: " + str(time_this_round)
                print(str_)
                sys.stdout.flush()


# TODO: RC function, Crysatllography likelihood
class FlexibleTrainer(object):

    def __init__(self, bg, optimizer=None, lr=0.001, clipnorm=None,
                 high_energy=100, max_energy=1e10, std_z=1.0, temperature=1.0,
                 w_KL=1.0, w_ML=1.0, w_L2_angle=0.0, w_xstal=0.0):
        """
        Parameters:
        -----------
        """

        self.bg = bg
        self.lr = lr
        self.high_energy = high_energy
        self.max_energy = max_energy
        self.std_z = std_z
        self.temperature = temperature

        self.w_ML = w_ML
        self.w_KL = w_KL
        self.w_L2_angle = w_L2_angle
        self.w_xstal = w_xstal

        if optimizer is None:
            if clipnorm is None:
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            else:
                self.optimizer = tf.keras.optimizers.Adam(
                    learning_rate=lr, clipnorm=clipnorm)

        self.loss_train = []

        inputs = []
        outputs = []
        applied_loss = []
        loss_for_metric = []

        if self.w_ML > 0.0:
            inputs.append(self.bg.input_x)

            outputs.append(self.bg.output_z)
            outputs.append(self.bg.log_det_Jxz)

            ml_loss = tf.keras.layers.Lambda(losses.MLlossNormal(
                std_z=self.std_z), name="ML_loss_layer")([self.bg.output_z, self.bg.log_det_Jxz])

            applied_loss.append(ml_loss*self.w_ML)
            loss_for_metric.append((ml_loss, "ML_loss"))

        if self.w_KL > 0.0:
            inputs.append(self.bg.input_z)

            outputs.append(self.bg.output_x)
            outputs.append(self.bg.log_det_Jzx)

            kl_loss_instance = losses.KLloss(
                self.bg.energy_model.energy_tf, self.high_energy, self.max_energy, temperature=self.temperature)

            kl_loss = tf.keras.layers.Lambda(kl_loss_instance,
                                             name="KL_loss_layer")([self.bg.output_x, self.bg.log_det_Jzx])

            applied_loss.append(kl_loss*self.w_KL)
            loss_for_metric.append((kl_loss, "KL_loss"))

        if self.w_L2_angle > 0.0:
            if self.w_KL <= 0.0:
                raise ValueError(
                    "Please set an nonzero w_kl when using L2 penalizaiton!")

            l2_loss = tf.keras.layers.Lambda(
                losses.loss_L2_angle_penalization, name="l2_loss")(self.bg)
            applied_loss.append(l2_loss*self.w_L2_angle)
            loss_for_metric.append((l2_loss, "L2_angle_Loss"))

        if self.w_xstal > 0.0:
            raise NotImplementedError

        # Construct model
        self.dual_model = tf.keras.models.Model(
            inputs=inputs, outputs=outputs, name="Dual_Model")
        self.dual_model.add_loss(applied_loss)
        for loss, loss_name in loss_for_metric:
            self.dual_model.add_metric(loss, name=loss_name)
        self.dual_model.compile(optimizer=self.optimizer)

    def train(self, x_train, epochs=2000,
              batchsize_ML=2000, batchsize_KL=None, batchsize_xstal=None,
              verbose=1, samplez_std=None, record_time=False):

        I = np.arange(x_train.shape[0])
        if samplez_std is None:
            samplez_std = self.std_z
        
        if batchsize_KL is None:
            batchsize_KL = batchsize_ML
        
        if batchsize_xstal is None:
            batchsize_xstal = 5

        @tf.function
        def steptrain(model_and_data):
            model, data = model_and_data
            return model.train_step((data,))

        for e in range(epochs):

            if record_time:
                start_time = time.time()

            input_for_training = []

            if self.w_ML > 0.0:
                # sample batch
                Isel = np.random.choice(I, size=batchsize_ML, replace=True)
                x_batch = x_train[Isel]
                input_for_training.append(x_batch)

            if self.w_KL > 0.0 or self.w_xstal > 0.0:
                z_batch = samplez_std * \
                    np.random.randn(batchsize_KL, self.bg.dim)
                input_for_training.append(z_batch)

            # Single step train
            losses_for_this_iteration = steptrain(
                [self.dual_model, input_for_training])

            loss_record_this_step = []
            str_ = 'Epoch ' + str(e) + '/' + str(epochs) + ' '
            for loss_name, loss_value in losses_for_this_iteration.items():
                loss_record_this_step.append(round(float(loss_value), 4))
                str_ += loss_name + ' '
                str_ += '{:.4f}'.format(float(loss_value)) + ' '

            self.loss_train.append(loss_record_this_step)

            if record_time:
                time_this_round = round(time.time() - start_time, 3)
                str_ += "Time: " + str(time_this_round)

            if verbose > 0:
                print(str_)
                sys.stdout.flush()
