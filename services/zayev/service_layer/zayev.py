import ray
import numpy as np
from keras import backend as K
from keras.models import Model, clone_model
from keras.layers import Input, Flatten, Dense, Lambda, concatenate, Concatenate
from keras.optimizers import Adam
from os import environ
from services.zayev.environment.market_simulator import MarketSimulator
from services.zayev.service_layer.actor import Actor
from services.zayev.service_layer.learner import Learner
from services.zayev.service_layer.parameter_server import ParameterServer
from services.zayev.service_layer.replay_buffer import ReplayBuffer
import tensorflow as tf

class Zayev:
    def __init__(self, config: dict):
        db_params = {
            'database': environ.get("POSTGRES_DB"),
            'user': environ.get("POSTGRES_USER"),
            'password': environ.get("POSTGRES_PASSWORD"),
            'host': environ.get("DB_HOST"),
            'port': environ.get("DB_PORT"),
        }
        self.starting_time_step = config.get("starting_time_step")
        self.max_episode_steps = config.get("max_episode_steps")
        env_config = {
            "db_params": db_params, 
            "max_episode_steps": self.max_episode_steps, 
            "the_current_time_step": self.starting_time_step,
            "print_output": False
        }
        self.fcnet_hiddens = config.get("fcnet_hiddens")
        self.fcnet_activation  = config.get("fcnet_activation")
        self.learning_rate = config.get("learning_rate")
        self.grad_clip = config.get("grad_clip")
        self.n_step = config.get("n_step", 1)
        self.q_update_freq = config.get("q_update_freq", 100)
        self.send_experience_freq = config.get("send_experience_freq")
        self.buffer_size = config.get("buffer_size")
        self.train_batch_size = config.get("train_batch_size", 32)
        self.eval_num_workers = config.get("eval_num_workers")
        self.num_workers = config.get("num_workers")
        self.max_eps = config.get("max_eps")
        self.max_samples = config.get("max_samples")
        self.timesteps_per_iteration = config.get("timesteps_per_iteration")
        self.gamma = config.get("gamma")
        self.learning_starts = config.get("learning_starts")
        self.eval_actor_ids: list[Actor] = []
        self.training_actor_ids: list[Actor] = []
        self.env = MarketSimulator(env_config)
        self.obs_space = self.env.observation_space
        self.action_shape = self.env.action_space.shape
        self.n_actions = self.action_shape[0]
        self.replay_buffer = ReplayBuffer(self)
        self.parameter_server = ParameterServer(self)
        self.learner: Learner = Learner(self, self.replay_buffer, self.parameter_server)


    def start_learning(self):
        self.learner.start_learning()

    def get_Q_network(self): 
        (stock_space, commodity_space, wallet_space) = self.obs_space
        stock_input = Input(shape=stock_space.shape, name='stock_observation_input')
        stock_input = Flatten()(stock_input)
        commodity_input = Input(shape=commodity_space.shape, name='commodity_observation_input')
        commodity_input = Flatten()(commodity_input)
        wallet_input = Input(shape=wallet_space.shape, name='wallet_observation_input')
        wallet_input = Flatten()(wallet_input)
        obs_input = Concatenate(name='Q_input')([stock_input, commodity_input, wallet_input])
        
        x = Flatten()(obs_input)
        for i, n_units in enumerate(self.fcnet_hiddens):
            layer_name = 'Q_' + str(i + 1)
            x = Dense(
                n_units,
                activation=self.fcnet_activation,
                name=layer_name
            )(x)

        q_estimate_output = Dense(
            self.n_actions,
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
    
    def masked_loss(self, args):
        y_true, y_pred, mask = args
        masked_pred = K.sum(mask * y_pred, axis=1, keepdims=True)
        loss = K.square(y_true - masked_pred)
        return K.mean(loss, axis=-1)
    
    def get_trainable_model(self):
        Q_model = self.get_Q_network()
        obs_input = Q_model.get_layer(name="Q_input").output
        q_estimate_output = Q_model.get_layer("Q_output").output
        # define 2 new inputs
        mask_input = Input(
            shape=self.n_actions,
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
        trainable_model = Model(
            inputs=[
                obs_input,
                mask_input,
                sampled_bellman_input
            ],
            outputs=loss_output
        )
        trainable_model.summary()
        trainable_model.compile(
            optimizer= Adam(
                learning_rate=self.learning_rate,
                clipvalue=self.grad_clip
            ),
            loss=[
                lambda y_true,
                y_pred: y_pred
            ]
        )
        return Q_model, trainable_model
    
    def create_eval_actors(self):
        for i in range(self.eval_num_workers):
            eps = 0
            actor: Actor = Actor(
                self,
                "eval-" + str(i),
                self.replay_buffer,
                self.parameter_server,
                eps,
                True
            )
            self.eval_actor_ids.append(actor)

    def create_training_actors(self):
        for i in range(self.num_workers):
            eps = self.max_eps * i / self.num_workers
            actor: Actor = Actor(
                self,
                "eval-" + str(i),
                self.replay_buffer,
                self.parameter_server,
                eps,
            )
            self.training_actor_ids.append(actor)

    def start(self):
        total_samples = 0
        best_eval_mean_reward = np.NINF
        eval_mean_rewards = []
        while total_samples < self.max_samples:
            tsid = self.replay_buffer.get_total_env_samples()
            new_total_samples = tsid
            if (new_total_samples - total_samples >= self.timesteps_per_iteration):
                total_samples = new_total_samples
                self.parameter_server.set_eval_weights()
                eval_sampling_ids = []
                for eval_actor in self.eval_actor_ids:
                    sid = eval_actor.sample()
                    eval_sampling_ids.append(sid)
                eval_rewards = eval_sampling_ids
                eval_mean_reward = np.mean(eval_rewards)
                eval_mean_rewards.append(eval_mean_reward)
                tf.summary.scalar('Mean evaluation reward', data=eval_mean_reward, step=total_samples)
                if eval_mean_reward > best_eval_mean_reward:
                    # print("best_eval_mean_reward", best_eval_mean_reward)
                    best_eval_mean_reward = eval_mean_reward
                    self.parameter_server.save_eval_weights()

        print("Finishing the training.")
        for actor in self.training_actor_ids:
            actor.stop()
        self.learner.stop()

    