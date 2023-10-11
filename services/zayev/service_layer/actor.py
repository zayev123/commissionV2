from datetime import datetime
from os import environ
from gymnasium import Space
from keras import Model
from keras.layers import Dense,Flatten, Input, Concatenate
from keras import backend as K
from matplotlib.dates import relativedelta
import tensorflow as tf
import numpy as np
from keras.optimizers import Adam

from services.zayev.environment.market_simulator import MarketSimulator


class Actor_Model:
    def __init__(self, env, lr, optimizer):

        self.env = env
        # self.env = env
        (stock_space, commodity_space, wallet_space) = self.env.observation_space
        stock_input = Input(shape=stock_space.shape, name='stock_observation_input')
        commodity_input = Input(shape=commodity_space.shape, name='commodity_observation_input')
        wallet_input = Input(shape=wallet_space.shape, name='wallet_observation_input')
        flattened_stock_input = Flatten()(stock_input)
        flattened_commodity_input = Flatten()(commodity_input)
        flattened_wallet_input = Flatten()(wallet_input)
        obs_input = Concatenate(name='ppo_input')([flattened_stock_input, flattened_commodity_input, flattened_wallet_input])
        X_input = Flatten()(obs_input)
        self.action_shape = self.env.action_space.shape[0]
        
        X = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        output = Dense(self.action_shape, activation="tanh")(X)

        self.Actor = Model(inputs=[stock_input, commodity_input, wallet_input], outputs = output)
        self.Actor.compile(loss=self.ppo_loss_continuous, optimizer=optimizer(lr=lr))
        #print(self.Actor.summary())

    def ppo_loss_continuous(self, y_true, y_pred):
        advantages, actions, logp_old_ph, = y_true[:, :1], y_true[:, 1:1+self.action_shape], y_true[:, 1+self.action_shape]
        LOSS_CLIPPING = 0.2
        logp = self.gaussian_likelihood(actions, y_pred)

        ratio = K.exp(logp - logp_old_ph)

        p1 = ratio * advantages
        p2 = tf.where(advantages > 0, (1.0 + LOSS_CLIPPING)*advantages, (1.0 - LOSS_CLIPPING)*advantages) # minimum advantage

        actor_loss = -K.mean(K.minimum(p1, p2))

        return actor_loss

    def gaussian_likelihood(self, actions, pred): # for keras custom loss
        log_std = -0.5 * np.ones(self.action_shape, dtype=np.float32)
        pre_sum = -0.5 * (((actions-pred)/(K.exp(log_std)+1e-8))**2 + 2*log_std + K.log(2*np.pi))
        return K.sum(pre_sum, axis=1)

    def predict(self, state):
        return self.Actor.predict(state)