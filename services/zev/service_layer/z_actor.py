from keras import Model
from keras.optimizers import Adam
from keras.layers import Dense,Flatten, Input, Concatenate, Lambda
from services.zayev.environment.market_simulator import MarketSimulator
import tensorflow as tf

class ZActor:
    def __init__(self, env: MarketSimulator):
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
        self.action_shape = self.env.action_space.shape[0]
        
        X = Dense(1024, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        X = Dense(1024, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        # X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        output = Dense(self.action_shape, activation="linear")(X)
        self.Actor = Model(inputs=[stock_input, commodity_input, wallet_input], outputs = output)
        self.Actor.compile(loss=self.calc_loss, optimizer=Adam())
    
    def calc_loss(self):
        pass