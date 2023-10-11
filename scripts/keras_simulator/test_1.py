
import sys
import django
import os
file_dir = "/Users/mirbilal/Desktop/MobCommission/commissionV2/"
if file_dir not in sys.path:
    sys.path.insert(0, file_dir)

os.environ["DJANGO_SETTINGS_MODULE"] = "commissionerv2.settings"
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true" 
django.setup()
from datetime import datetime
from os import environ

from matplotlib.dates import relativedelta

from services.zayev.environment.market_simulator import MarketSimulator
from keras.layers import Input, concatenate, Flatten, Dense, Concatenate, Lambda
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K

class Test1:

    def __init__(self):
        self.config = {
            "env": "CartPole-v1",
            "obs_shape": (4,), 
            "num_workers": 1,
            "eval_num_workers": 1,
            "n_step": 3,
            "n_actions": 5,
            "max_eps": 0.5,
            "train_batch_size": 50,
            "gamma": 0.99,
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "tanh",
            "lr": 0.0001,
            "buffer_size": 100,
            "learning_starts": 100,
            "timesteps_per_iteration": 200,
            "grad_clip": 10,
            "max_samples": 500,
            # "starting_time_step": starting_time_step
        }
        db_params = {
            'database': environ.get("POSTGRES_DB"),
            'user': environ.get("POSTGRES_USER"),
            'password': environ.get("POSTGRES_PASSWORD"),
            'host': environ.get("DB_HOST"),
            'port': environ.get("DB_PORT"),
        }
        my_current_time_step  = datetime(year=1995, month=1, day=1, hour=10) + relativedelta(hours=0)

        env_config = {
            "db_params": db_params, 
            "max_episode_steps": 150, 
            "print_output": False,
            "the_current_time_step": my_current_time_step,
        }

        self.myEnv = MarketSimulator(env_config)

    def masked_loss(self, args):
        y_true, y_pred, mask = args
        masked_pred = K.sum(mask * y_pred, axis=1, keepdims=True)
        loss = K.square(y_true - masked_pred)
        return K.mean(loss, axis=-1)

    def get_Q_test(self):

        db_params = {
            'database': environ.get("POSTGRES_DB"),
            'user': environ.get("POSTGRES_USER"),
            'password': environ.get("POSTGRES_PASSWORD"),
            'host': environ.get("DB_HOST"),
            'port': environ.get("DB_PORT"),
        }
        my_current_time_step  = datetime(year=1995, month=1, day=1, hour=10) + relativedelta(hours=0)

        env_config = {
            "db_params": db_params, 
            "max_episode_steps": 150, 
            "print_output": False,
            "the_current_time_step": my_current_time_step,
        }

        myEnv = MarketSimulator(env_config)
        obs = myEnv.observation_space
        self.action_shape = myEnv.action_space.shape
        print("yp", self.action_shape[0])
        shp = obs.shape
        (stock_shape, commodity_shape, wallet_shape) = obs
        stck_shape = stock_shape.shape
        if isinstance(stck_shape, int):
            # If stock_shape is an integer, create a tuple of length 1
            stck_shape = (stck_shape,)
            print("here")
        print(wallet_shape.shape)
        # print(stock_shape.shape)
        print(stock_shape.shape)
        def get_tuple_shapes(listoshps):
            listolist = []
            for alisto in listoshps:
                alisto = list(alisto)
                listolist.append(alisto)
            shapes = listolist
            max_len = 0
            for shp in shapes:
                leni = len(shp)
                if leni>max_len:
                    max_len = leni

            for shp in shapes:
                leni = len(shp)
                adds = max_len - leni
                for sh in range(adds):
                    shp.append(0)

            max_num = 0
            for shp in shapes:
                a_num = shp[-1]
                if a_num > max_num:
                    max_num = a_num

            laListoShps = []
            for shp in shapes:
                shp[-1] = max_num


                shp = tuple(shp)
                laListoShps.append(shp)
            return(laListoShps)

        stock_input = Input(shape=stock_shape.shape, name='stock_observation_input')
        stock_input = Flatten()(stock_input)
        commodity_input = Input(shape=commodity_shape.shape, name='commodity_observation_input')
        commodity_input = Flatten()(commodity_input)
        wallet_input = Input(shape=wallet_shape.shape, name='wallet_observation_input')
        wallet_input = Flatten()(wallet_input)
        obs_input = Concatenate(name='merged_input')([stock_input, commodity_input, wallet_input])

        x = Flatten()(obs_input)
        for i, n_units in enumerate(self.config["fcnet_hiddens"]):
            layer_name = 'Q_' + str(i + 1)
            x = Dense(
                n_units,
                activation=self.config["fcnet_activation"],
                name=layer_name
            )(x)

        q_estimate_output = Dense(
            self.config["n_actions"],
            activation='linear', 
            name='Q_output'
        )(x)

        # Q Model
        Q_model = Model(
            inputs=[stock_input, commodity_input, wallet_input],
            outputs=q_estimate_output
        )
        Q_model.summary()
        Q_model.compile(optimizer=Adam(), loss='mse')
        return Q_model
    
    def tes2(self):
        (stock_space, commodity_space, wallet_space) = self.myEnv.observation_space
        stock_input = Input(shape=stock_space.shape, name='stock_observation_input')
        commodity_input = Input(shape=commodity_space.shape, name='commodity_observation_input')
        wallet_input = Input(shape=wallet_space.shape, name='wallet_observation_input')
        flattened_stock_input = Flatten()(stock_input)
        flattened_commodity_input = Flatten()(commodity_input)
        flattened_wallet_input = Flatten()(wallet_input)
        obs_input = Concatenate(name='ppo_input')([flattened_stock_input, flattened_commodity_input, flattened_wallet_input])
        X_input = Flatten()(obs_input)
        self.action_shape = self.myEnv.action_space.shape[0]
        
        X = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        output = Dense(self.action_shape, activation="tanh")(X)

        self.Actor = Model(inputs=[stock_input, commodity_input, wallet_input], outputs = output)
        self.Actor.compile(loss=self.ppo_loss_continuous, optimizer=Adam())

    def ppo_loss_continuous(self, y_true, y_pred):
        advantages, actions, logp_old_ph, = y_true[:, :1], y_true[:, 1:1+self.action_shape], y_true[:, 1+self.action_shape]
        LOSS_CLIPPING = 0.2
        logp = self.gaussian_likelihood(actions, y_pred)

        ratio = K.exp(logp - logp_old_ph)

        p1 = ratio * advantages
        p2 = tf.where(advantages > 0, (1.0 + LOSS_CLIPPING)*advantages, (1.0 - LOSS_CLIPPING)*advantages) # minimum advantage

        actor_loss = -K.mean(K.minimum(p1, p2))

        return actor_loss


    def get_trainable_model(self):
        Q_model = self.get_Q_test()
        my_layer = Q_model.get_layer(name="meged_input")
        Q_layer = Q_model.get_layer(index=1)
        obs_input = my_layer.output
        q_estimate_output = Q_model.get_layer("Q_output").output
        # define 2 new inputs
        mask_input = Input(
            shape=self.config["n_actions"],
            name='Q_mask'
        )
        sampled_bellman_input = Input(
            shape=(1,),
            name='Q_sampled'
        )
        # Trainable model
        loss_output = Lambda(self.masked_loss,output_shape=(1,), name='Q_masked_out')(
            [
                sampled_bellman_input,
                q_estimate_output,
                mask_input
            ]
        )
        print("obs_input", obs_input)
        trainable_model = Model(
            inputs=[
                obs_input,
                mask_input,
                sampled_bellman_input
            ],
            outputs=loss_output
        )
