from keras import Model
from keras.layers import Dense,Flatten, Input, Concatenate
from keras import backend as K
import tensorflow as tf
import numpy as np

class Critic_Model:
    def __init__(self, env, lr, optimizer):
        self.env = env
        (stock_space, commodity_space, wallet_space) = self.env.observation_space
        stock_input = Input(shape=stock_space.shape, name='stock_observation_input')
        commodity_input = Input(shape=commodity_space.shape, name='commodity_observation_input')
        wallet_input = Input(shape=wallet_space.shape, name='wallet_observation_input')
        flattened_stock_input = Flatten()(stock_input)
        flattened_commodity_input = Flatten()(commodity_input)
        flattened_wallet_input = Flatten()(wallet_input)
        obs_input = Concatenate(name='ppo_input')([flattened_stock_input, flattened_commodity_input, flattened_wallet_input])
        X_input = Flatten()(obs_input)
        old_values = Input(shape=(1,))

        V = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        V = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(V)
        V = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(V)
        V = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(V)
        value = Dense(1, activation=None)(V)

        self.Critic = Model(inputs=[stock_input, commodity_input, wallet_input, old_values], outputs = value)
        self.Critic.compile(loss=[self.critic_PPO2_loss(old_values)], optimizer=optimizer(lr=lr))

    def critic_PPO2_loss(self, values):
        def loss(y_true, y_pred):
            LOSS_CLIPPING = 0.2
            clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2
            
            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            #value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
            return value_loss
        return loss

    def predict(self, state):
        [stock_state, commodity_state, wallet_state] = state
        return self.Critic.predict([stock_state, commodity_state, wallet_state, np.zeros((stock_state.shape[0], 1))])