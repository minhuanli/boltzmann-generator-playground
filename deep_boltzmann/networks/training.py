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
        else:
            self.optimizer = optimizer

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


# TODO: RC function
class FlexibleTrainer:

    def __init__(self, bg, optimizer=None, lr=0.001, clipnorm=None,
                 high_energy=20000, max_energy=1e10, std_z=1.0, temperature=1.0,
                 w_ML=1.0, w_KL=1.0, w_L2_angle=0.0, w_xstal=0.0,
                 xstalloss=None, validation=True):
        """
        Parameters:
        -----------
        bg: EnergInvNet
            A boltzmann generator obejct

        optimizer: tf.keras.optimizers, default None
            The optimizer used in the training, if not given, will use Adam with the lr parameters

        lr: float
            learning rate for the optimizer

        high_energy: float
            The number of high energy used to scale the protein openmm energy

        max_energy: float
            The number used as a upper bound of the protein openmm energy

        std_z: float, default 1.0
            The standard deviation of the gaussiaon prior in latent space

        temperature: float, default 1.0
            The temperature factor used to scale the energy in real space

        w_ML: float, default 1.0
            The weight of ML loss 

        w_KL: float, default 1.0
            the weight of KL loss

        w_L2_angle: float, default 0.0
            the weight of L2 angle loss

        w_xstal: float, default 0.0
            the weight of crystallography loss

        xstalloss: losses.Xstalloss, defualt None
            The Xsatlloss class instance containing all crystalllography infos to compute the strucutrual factor loss.
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

        self.validation = validation

        self.mode = 0

        if optimizer is None:
            if clipnorm is None:
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            else:
                self.optimizer = tf.keras.optimizers.Adam(
                    learning_rate=lr, clipnorm=clipnorm)
        else:
            self.optimizer = optimizer

        self.loss_train = []

        inputs = []
        outputs = []
        self.loss_names = ["loss"]

        if self.w_ML > 0.0:
            inputs.append(self.bg.input_x)

            outputs.append(self.bg.output_z)
            outputs.append(self.bg.log_det_Jxz)
            self.mlloss = losses.MLlossNormal(std_z=self.std_z)
            self.loss_names.append("ml_loss")
            self.mode += 1

        if self.w_KL > 0.0:
            inputs.append(self.bg.input_z)
            outputs.append(self.bg.output_x)
            outputs.append(self.bg.log_det_Jzx)
            l2_loss = losses.loss_L2_angle_penalization(self.bg)
            outputs.append(l2_loss)
            self.klloss = losses.KLloss(
                self.bg.energy_model.energy_tf, self.high_energy, self.max_energy, temperature=self.temperature)
            self.loss_names.append("kl_loss")
            self.mode += 2

        if self.w_xstal > 0.0:
            if self.w_KL <= 0.0:
                inputs.append(self.bg.input_z)
                outputs.append(self.bg.output_x)
                outputs.append(self.bg.log_det_Jzx)
            self.xstalloss = xstalloss
            self.loss_names.append("xstal_loss")
            if self.validation:
                self.loss_names.append("xstal_loss_free")
                self.loss_names.append("r_work")
                self.loss_names.append("r_free")
            self.mode += 4

        if self.w_L2_angle > 0.0:
            if self.w_KL <= 0.0:
                raise ValueError(
                    "Please set an nonzero w_kl when using L2 penalization!")
            self.loss_names.append("l2_loss")

        # Construct model
        self.dual_model = tf.keras.models.Model(
            inputs=inputs, outputs=outputs, name="Dual_Model")

        self.model_parameters = []
        self.model_parameters.extend(self.dual_model.trainable_weights)

        # Seperate the scale training and the flow training
        # if self.w_xstal > 0.0:
        #     self.model_parameters.extend(self.xstalloss.trainable_weights)

    @tf.function
    def steptrain_ml(self, input_for_training):
        with tf.GradientTape() as tape:
            output_z, log_det_Jxz = self.dual_model(input_for_training)
            ml_loss = self.mlloss([output_z, log_det_Jxz])
            loss = self.w_ML*ml_loss
        grads = tape.gradient(loss, self.model_parameters)
        self.optimizer.apply_gradients(zip(grads, self.model_parameters))
        return [loss, ml_loss]

    @tf.function
    def steptrain_kl(self, input_for_training):
        with tf.GradientTape() as tape:
            output_x, log_det_Jzx, l2_loss = self.dual_model(
                input_for_training)
            kl_loss = self.klloss([output_x, log_det_Jzx])
            loss = self.w_KL*kl_loss + self.w_L2_angle*l2_loss
        grads = tape.gradient(loss, self.model_parameters)
        self.optimizer.apply_gradients(zip(grads, self.model_parameters))
        return [loss, kl_loss, l2_loss]

    @tf.function
    def steptrain_mlkl(self, input_for_training):
        with tf.GradientTape() as tape:
            output_z, log_det_Jxz, output_x, log_det_Jzx, l2_loss = self.dual_model(
                input_for_training)
            ml_loss = self.mlloss([output_z, log_det_Jxz])
            kl_loss = self.klloss([output_x, log_det_Jzx])
            loss = self.w_ML*ml_loss + self.w_KL*kl_loss + self.w_L2_angle*l2_loss
        grads = tape.gradient(loss, self.model_parameters)
        self.optimizer.apply_gradients(zip(grads, self.model_parameters))
        return [loss, ml_loss, kl_loss, l2_loss]

    @tf.function
    def steptrain_klxstal(self, input_for_training):
        with tf.GradientTape() as tape:
            output_x, log_det_Jzx, l2_loss = self.dual_model(
                input_for_training)
            kl_loss = self.klloss([output_x, log_det_Jzx])
            xstal_loss, xstal_loss_free, r_work, r_free = self.xstalloss(
                output_x[0:self.batchsize_xstal], self.batchsize_xstal)
            loss = self.w_KL*kl_loss + self.w_xstal*xstal_loss + self.w_L2_angle*l2_loss
        grads = tape.gradient(loss, self.model_parameters)
        self.optimizer.apply_gradients(zip(grads, self.model_parameters))
        if self.validation:
            return [loss, kl_loss, xstal_loss, xstal_loss_free, r_work, r_free, l2_loss]
        else:
            return [loss, kl_loss, xstal_loss, l2_loss]

    @tf.function
    def steptrain_mlklxstal(self, input_for_training):
        with tf.GradientTape() as tape:
            output_z, log_det_Jxz, output_x, log_det_Jzx, l2_loss = self.dual_model(
                input_for_training)
            ml_loss = self.mlloss([output_z, log_det_Jxz])
            kl_loss = self.klloss([output_x, log_det_Jzx])
            xstal_loss, xstal_loss_free, r_work, r_free = self.xstalloss(
                output_x[0:self.batchsize_xstal], self.batchsize_xstal)
            loss = self.w_ML*ml_loss + self.w_KL*kl_loss + \
                self.w_xstal*xstal_loss + self.w_L2_angle*l2_loss
        grads = tape.gradient(loss, self.model_parameters)
        self.optimizer.apply_gradients(zip(grads, self.model_parameters))
        if self.validation:
            return [loss, ml_loss, kl_loss, xstal_loss, xstal_loss_free, r_work, r_free, l2_loss]
        else:
            return [loss, ml_loss, kl_loss, xstal_loss, l2_loss]

    def train(self, x_train=None, epochs=2000,
              batchsize_ML=1024, batchsize_KL=None, batchsize_xstal=1,
              verbose=1, samplez_std=None, record_time=False, start_count=0):

        if self.w_ML > 0.0:
            I = np.arange(x_train.shape[0])

        if samplez_std is None:
            samplez_std = self.std_z

        if batchsize_KL is None:
            batchsize_KL = batchsize_ML

        self.batchsize_xstal = batchsize_xstal

        for e in range(epochs):

            if record_time:
                start_time = time.time()

            input_for_training = []

            if self.w_ML > 0.0:
                # sample batch
                Isel = np.random.choice(I, size=batchsize_ML, replace=True)
                x_batch = x_train[Isel]
                input_for_training.append(x_batch)

            if self.w_KL > 0.0:
                z_batch = samplez_std * \
                    np.random.randn(batchsize_KL, self.bg.dim)
                input_for_training.append(z_batch)

            if self.mode == 1:
                # ML
                losses_for_this_iteration = self.steptrain_ml(
                    input_for_training)

            elif self.mode == 2:
                # KL
                losses_for_this_iteration = self.steptrain_kl(
                    input_for_training)

            elif self.mode == 3:
                # ML + KL
                losses_for_this_iteration = self.steptrain_mlkl(
                    input_for_training)

            elif self.mode == 6:
                # KL + Xstal
                losses_for_this_iteration = self.steptrain_klxstal(
                    input_for_training)

            elif self.mode == 7:
                # ML + KL + XSTALL loss
                losses_for_this_iteration = self.steptrain_mlklxstal(
                    input_for_training)

            else:
                raise NotImplementedError("Currently only support ")

            str_ = "Epoch " + str(start_count+e) + "/" + str(start_count+epochs) + " "
            loss_this_round = [start_count+e]
            for i, loss_name in enumerate(self.loss_names):
                str_ += loss_name + ": "
                str_ += "{:.4f}".format(
                    float(losses_for_this_iteration[i])) + "  "
                loss_this_round.append(float(losses_for_this_iteration[i]))

            if record_time:
                time_this_round = round(time.time() - start_time, 3)
                str_ += "Time: " + str(time_this_round)

            self.loss_train.append(loss_this_round)

            if verbose:
                print(str_)
                sys.stdout.flush()


class ScaleTrainer:

    def __init__(self, bg, xstalloss, optimizer=None, lr=0.0005, samplez_std=1.0, batchsize=3):
        z_batch = samplez_std * np.random.randn(batchsize, bg.dim)
        self.output_x = bg.Tzx.predict(z_batch)
        self.batchsize = batchsize
        self.xstalloss = xstalloss
        self.loss_names = ["xstal_loss", "xstal_loss_free", "r_work", "r_free"]
        self.loss_train = []
        
        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            self.optimizer = optimizer
        
    @tf.function
    def steptrain(self):
        with tf.GradientTape() as tape:
            xstal_loss, xstal_loss_free, r_work, r_free = self.xstalloss(self.output_x, self.batchsize)
        grads = tape.gradient(xstal_loss, self.xstalloss.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.xstalloss.trainable_weights))
        return [xstal_loss, xstal_loss_free, r_work, r_free] 
        
    def train(self, epochs=2000, verbose=1, record_time=False, start_count=0):

        for e in range(epochs):
            
            if record_time:
                start_time = time.time() 
            
            losses_for_this_iteration = self.steptrain()
            
            str_ = "Epoch " + str(start_count+e) + "/" + str(start_count+epochs) + " "
            loss_this_round = [start_count+e]
            for i, loss_name in enumerate(self.loss_names):
                str_ += loss_name + ": "
                str_ += "{:.4f}".format(
                    float(losses_for_this_iteration[i])) + "  "
                loss_this_round.append(float(losses_for_this_iteration[i]))

            if record_time:
                time_this_round = round(time.time() - start_time, 3)
                str_ += "Time: " + str(time_this_round)

            self.loss_train.append(loss_this_round)

            if verbose:
                print(str_)
                sys.stdout.flush()



        


        



        
