# flake8:noqa

import numpy as np
np.random.seed(0)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
tf.random.set_seed(0)
from tensorflow import keras, data
import tensorflow_probability as tfp
from tensorflow.keras import layers, regularizers, activations
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import backend as K
import seaborn as sns
import matplotlib.pyplot as plt

row_mark = 225
batch_size = 1
time_step = 1
lstm_h_dim = 8
z_dim = 4
epoch_num = 32
threshold = 50
alpha = 0.8
beta = 0.2


#extension parameters
N_constraintNet = 10
w_thres = 0.01
learning_rate = 1e-6
####################

mode = 'train'
model_dir = "./lstm_vae_model/"
image_dir = "./lstm_vae_images/"

#################
# Reshape Function
#################

#################
# Reparametrization
#################

class Sampling(layers.Layer):
    def __init__(self, name='sampling_z'):
        super(Sampling, self).__init__(name=name)

    def call(self, inputs,**kwargs):
        mu, logvar = inputs
        print('mu: ', mu)
        sigma = K.exp(logvar * 0.5)
        epsilon = K.random_normal(shape=(mu.shape[0], z_dim), mean=0.0, stddev=1.0)
        return mu + epsilon * sigma

    def get_config(self):
        config = super(Sampling, self).get_config()
        config.update({'name': self.name})
        return config

#################
# Encoder
#################

class LSTMEncoder(layers.Layer):
    def __init__(self, time_step, x_dim, lstm_h_dim, z_dim, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.encoder_inputs = keras.Input(shape=(time_step, x_dim))
        self.encoder_lstm = Bidirectional(layers.LSTM(lstm_h_dim, activation='softplus', name='encoder_lstm', stateful=True))
        self.z_mean = layers.Dense(z_dim, name='z_mean')
        self.z_logvar = layers.Dense(z_dim, name='z_log_var')
        self.z_sample = Sampling()

    def call(self, inputs,**kwargs):
        self.encoder_inputs = inputs
        hidden = self.encoder_lstm(self.encoder_inputs)
        mu_z = self.z_mean(hidden)
        logvar_z = self.z_logvar(hidden)
        z = self.z_sample((mu_z, logvar_z))
        return mu_z, logvar_z, z

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'name': self.name,
            'z_sample': self.z_sample.get_config()
        })
        return config


#################
# Decoder
#################

class LSTMDecoder(layers.Layer):
    def __init__(self, time_step, x_dim, lstm_h_dim, z_dim, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.z_inputs = layers.RepeatVector(time_step, name='repeat_vector')
        self.decoder_lstm_hidden = Bidirectional(layers.LSTM(lstm_h_dim, activation='softplus', return_sequences=True,
                                               name='decoder_lstm'))
        self.x_mean = layers.Dense(x_dim, name='x_mean')
        self.x_logvar = layers.Dense(x_dim, name='x_log_var')
        self.x_sample = Sampling()
        self.x_sigma = layers.Dense(x_dim, name='x_sigma', activation='tanh')

    def call(self, inputs,**kwargs):
        z = self.z_inputs(inputs)
        hidden = self.decoder_lstm_hidden(z)
        mu_x = self.x_mean(hidden)
        logvar_x = self.x_logvar(hidden)
        sigma_x = self.x_sigma(hidden)
        x_dash = self.x_sample((mu_x, logvar_x))
        return mu_x, sigma_x,x_dash

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            'name': self.name
        })
        return config


###################
#Constraint Network
###################
class ConstraintNet(keras.Model):
    def __init__(self, time_step, x_dim, z_dim, N, name='constraint_net', **kwargs):
        super(ConstraintNet, self).__init__(name=name, **kwargs)

        self.layer1 = layers.Dense(8, input_dim=x_dim, activation='relu')
        self.layer2 = layers.Dense(16, activation='relu')
        self.out = layers.Dense(z_dim * N, activation='relu')
        self.reshape = layers.Reshape((N, z_dim))

        self.N = N
        self.z_dim = z_dim

    def call(self, inputs, z):
        h = self.layer1(inputs)
        h = self.layer2(h)
        out = self.out(h)
        c_mat = self.reshape(out)
        w = tf.matmul(z, c_mat, transpose_b=True)
        thres = tf.constant(w_thres, shape=w.shape)
        mask = tf.cast(w > thres, dtype=tf.float32)
        w_dash = tf.multiply(w, mask)
        z_dash = tf.linalg.matmul(w_dash, c_mat)
        z_dash = tf.squeeze(z_dash, axis=1)
        return z_dash

    def cosine_similarity(self, X, Y):
        X_norm = tf.norm(X, axis=1)
        Y_norm = tf.norm(Y, axis=1)
        dot_prod = tf.reduce_sum(X * Y, 1)
        w = dot_prod / (X_norm * Y_norm)
        return w
###############################################

#################
# Simple VELC
#################

loss_metric = keras.metrics.Mean(name='loss')
likelihood_metric = keras.metrics.Mean(name='log likelihood')

class VELC(keras.Model):
    def __init__(self, time_step, x_dim, lstm_h_dim, z_dim,N_constraintNet, name='velc', **kwargs):
        super(VELC, self).__init__(name=name, **kwargs)

        self.encoder = LSTMEncoder(time_step, x_dim, lstm_h_dim, z_dim, **kwargs)
        self.decoder = LSTMDecoder(time_step, x_dim, lstm_h_dim, z_dim, **kwargs)


        self.constraint_net_1 = ConstraintNet(time_step,x_dim,z_dim,N_constraintNet,name="constNet1",**kwargs)
        self.re_encoder = LSTMEncoder(time_step, x_dim, lstm_h_dim, z_dim,name="re_encoder",**kwargs)
        self.constraint_net_2 = ConstraintNet(time_step,x_dim,z_dim,N_constraintNet,name="constNet2",**kwargs)


    def call(self, inputs,**kwargs):
        mu_z, logvar_z, z = self.encoder(inputs)
        z_dash = self.constraint_net_1(inputs, z)
        mu_x, sigma_x, x_dash = self.decoder(z_dash)
        mu_re_z, logvar_re_z, re_z = self.re_encoder(x_dash)
        re_z_dash = self.constraint_net_2(x_dash, re_z)

        recons_loss = self.l2_NORM(inputs, x_dash)
        kl_loss_1 = self.kl_loss(logvar_z,mu_z)
        kl_loss_2 = self.kl_loss(logvar_re_z,mu_re_z)
        latent_loss = self.l2_NORM(z_dash, re_z_dash)
        total_loss = recons_loss + kl_loss_1 + kl_loss_2 + latent_loss


        anomaly_score = self.anomaly_score(inputs, x_dash, z_dash, re_z_dash)
        dist = tfp.distributions.Normal(loc=mu_x, scale=tf.abs(sigma_x))
        log_px = -dist.log_prob(inputs)

        return anomaly_score,total_loss, log_px


    def get_config(self):
        config = {
            'encoder': self.encoder.get_config(),
            'decoder': self.decoder.get_config(),
            'name': self.name
        }
        return config

    def l2_NORM(self, x, x_dash):
        recons_loss = tf.norm(x - x_dash,ord='euclidean', axis=1)
        return K.mean(recons_loss)

    def kl_loss(self, logvar_z, mu_z):
        kl_loss = -0.5 * (1 + logvar_z - tf.square(mu_z) - tf.exp(logvar_z))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        return kl_loss

    def mean_log_likelihood(self, log_px):
        log_px = K.reshape(log_px, shape=(log_px.shape[0], log_px.shape[2]))
        mean_log_px = K.mean(log_px, axis=1)
        return K.mean(mean_log_px, axis=0)

    def train_step(self, data):
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        with tf.GradientTape() as tape:
            anomaly_score,loss, log_px = self(x, training=True)
            mean_log_px = self.mean_log_likelihood(log_px)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        loss_metric.update_state(loss)
        likelihood_metric.update_state(mean_log_px)
        return {'loss': loss_metric.result(), 'log_likelihood': likelihood_metric.result()}


    def anomaly_score(self, x, x_dash, z_dash, re_z_dash):
        a1 = tf.squeeze(tf.norm(x - x_dash, ord=1, axis=1))
        a2 = tf.norm(z_dash - re_z_dash, ord=1, axis=0)
        ax = alpha * a1 + beta * a2
        ax_mean = K.mean(ax)
        return ax_mean
