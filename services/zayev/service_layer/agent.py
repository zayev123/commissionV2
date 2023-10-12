from datetime import datetime
import os

from matplotlib.dates import relativedelta
import psycopg2
import pytz
from apps.environment_simulator.models import (
    SimulatedCommodity,
    SimulatedStock,
    SimulatedCommodityBuffer,
    SimulatedStockBuffer
)

from services.zayev.environment.market_simulator import MarketSimulator
from services.zayev.service_layer.env_process import EnvProcess

from .actor import Actor_Model
from services.zayev.service_layer.critic import Critic_Model
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # -1:cpu, 0:first gpu
import random
import gymnasium as gym
import pylab
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
#tf.config.experimental_run_functions_eagerly(True) # used for debuging and development
tf.compat.v1.disable_eager_execution() # usually using this for fastest performance
from keras.layers import Dense,Flatten, Input
from keras.models import Model, load_model
from keras.optimizers import RMSprop, Adagrad, Adadelta
from keras.optimizers.legacy import Adam
from keras import backend as K
import copy
import pandas as pd

from threading import Thread, Lock
from multiprocessing import Process, Pipe
from os import environ
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass

    

class PPOAgent:
    # PPO Main Optimization Algorithm
    def __init__(self, env_config):
        # Initialization
        # Environment and PPO parameters
        self.env_name = "market"   
        self.env_config = env_config
        self.env = MarketSimulator(env_config=env_config)
        
        self.action_size = self.env.action_space.shape[0]
        (stock_space, commodity_space, wallet_space) = self.env.observation_space
        stk_size = 1
        for stck_size in stock_space.shape:
            stk_size = stk_size * stck_size
        cmdt_siz = 1
        for cmmdty_size in commodity_space.shape:
            cmdt_siz = cmdt_siz * cmmdty_size
        wlt_sz = 1
        for wllt_size in wallet_space.shape:
            wlt_sz = wlt_sz * wllt_size
        


        self.EPISODES = 20 # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 0 # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.epochs = 20 # training epochs
        self.shuffle = True
        self.Training_batch = 100
        #self.optimizer = RMSprop
        self.optimizer = Adam
        self.test_steps = min(self.Training_batch, env_config.get("test_steps", 1))

        self.replay_count = 0
        self.writer = SummaryWriter(comment="_"+self.env_name+"_"+self.optimizer.__name__+"_"+str(self.lr))
        
        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], [] # used in matplotlib plots

        # Create Actor-Critic network models
        self.Actor = Actor_Model(self.env, lr=self.lr, optimizer = self.optimizer)
        self.Critic = Critic_Model(self.env, lr=self.lr, optimizer = self.optimizer)
        
        service_path = "/Users/mirbilal/Desktop/MobCommission/commissionV2/services/zayev"
        # service_path = "."
        self.Actor_name = f"{service_path}/weightings/actor.h5"
        self.Critic_name = f"{service_path}/weightings/critic.h5"
        # self.load() # uncomment to continue training from old weights

        # do not change bellow
        self.log_std = -0.1 * np.ones(self.action_size, dtype=np.float32)
        self.std = np.exp(self.log_std)

        self.latest_state = None
        self.previous_state = None
        self.previous_actn = None


    def act(self, state):
        # Use the network to predict the next action to take, using the model
        pred = self.Actor.predict(state)

        low, high = -1.0, 1.0 # -1 and 1 are boundaries of tanh
        action = pred + np.random.uniform(low, high, size=pred.shape) * self.std
        action = np.clip(action, low, high)
        
        logp_t = self.gaussian_likelihood(action, pred, self.log_std)

        return action, logp_t

    def gaussian_likelihood(self, action, pred, log_std):
        # https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/sac/policies.py
        pre_sum = -0.5 * (((action-pred)/(np.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi)) 
        return np.sum(pre_sum, axis=1)

    def discount_rewards(self, reward):#gaes is better
        # Compute the gamma-discounted rewards over an episode
        # We apply the discount and normalize it to avoid big variability of rewards
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= (np.std(discounted_r) + 1e-8) # divide by standard deviation
        return discounted_r

    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.90, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def flatten_states(self, states):
        flattened_states = []

        for state in states:
            # Flatten each state and concatenate them into a single array
            flattened_state = [item for sublist in state for item in sublist]
            flattened_states.append(flattened_state)

        # Convert the list of flattened states into a NumPy array
        final_states = np.array(flattened_states)
        return final_states
    
    def replay(
            self, stock_states, commodity_states, wallet_states, 
            actions, rewards, dones, 
            next_stock_states, next_commodity_states, next_wallet_states, 
            logp_ts
    ):
        stock_states = np.vstack(stock_states)
        commodity_states = np.vstack(commodity_states)
        wallet_states = np.vstack(wallet_states)
        next_stock_states = np.vstack(next_stock_states)
        next_commodity_states = np.vstack(next_commodity_states)
        next_wallet_states = np.vstack(next_wallet_states)
        actions = np.vstack(actions)
        logp_ts = np.vstack(logp_ts)

        # Get Critic network predictions 
        values = self.Critic.predict([stock_states, commodity_states, wallet_states])
        next_values = self.Critic.predict([next_stock_states, next_commodity_states, next_wallet_states])

        # Compute discounted rewards and advantages
        #discounted_r = self.discount_rewards(rewards)
        #advantages = np.vstack(discounted_r - values)
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
        '''
        pylab.plot(adv,'.')
        pylab.plot(target,'-')
        ax=pylab.gca()
        ax.grid(True)
        pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
        pylab.show()
        if str(episode)[-2:] == "00": pylab.savefig(self.env_name+"_"+self.episode+".png")
        '''
        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        # in custom loss function we unpack it
        y_true = np.hstack([advantages, actions, logp_ts])
        
        # training Actor and Critic networks
        a_loss = self.Actor.Actor.fit(
            [stock_states, commodity_states, wallet_states], 
            y_true, epochs=self.epochs, 
            verbose=0, shuffle=self.shuffle
        )
        c_loss = self.Critic.Critic.fit(
            [stock_states, commodity_states, wallet_states, values], 
            target, epochs=self.epochs, verbose=0, shuffle=self.shuffle
        )

        # calculate loss parameters (should be done in loss, but couldn't find working way how to do that with disabled eager execution)
        pred = self.Actor.predict([stock_states, commodity_states, wallet_states])
        log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        logp = self.gaussian_likelihood(actions, pred, log_std)
        approx_kl = np.mean(logp_ts - logp)
        approx_ent = np.mean(-logp)

        self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/approx_kl_per_replay', approx_kl, self.replay_count)
        self.writer.add_scalar('Data/approx_ent_per_replay', approx_ent, self.replay_count)
        self.replay_count += 1
 
    def load(self):
        self.Actor.Actor.load_weights(self.Actor_name)
        self.Critic.Critic.load_weights(self.Critic_name)

    def save(self):
        self.Actor.Actor.save_weights(self.Actor_name)
        self.Critic.Critic.save_weights(self.Critic_name)

    pylab.figure(figsize=(18, 9))
    pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
    def PlotModel(self, score, episode, save=True):
        self.scores_.append(score)
        self.episodes_.append(episode)
        self.average_.append(sum(self.scores_[-50:]) / len(self.scores_[-50:]))
        if str(episode)[-2:] == "00":# much faster than episode % 100
            pylab.plot(self.episodes_, self.scores_, 'b')
            pylab.plot(self.episodes_, self.average_, 'r')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.grid(True)
                pylab.savefig(self.env_name+".png")
            except OSError:
                pass
        # saving best models
        if self.average_[-1] >= self.max_average and save:
            self.max_average = self.average_[-1]
            self.save()
            SAVING = "SAVING"
            # decreaate learning rate every saved model
            #self.lr *= 0.99
            #K.set_value(self.Actor.Actor.optimizer.lr, self.lr)
            #K.set_value(self.Critic.Critic.optimizer.lr, self.lr)
        else:
            SAVING = ""

        return self.average_[-1], SAVING
    
    def run_batch(self):
        state = self.env.reset()
        state = self.reshape_state(state)
        done, score, SAVING = False, 0, ''
        is_break = False
        while True:
            # Instantiate or reset games memory
            stock_states, commodity_states, wallet_states, \
            next_stock_states, next_commodity_states, next_wallet_states, \
            actions, rewards, dones, logp_ts = [], [], [], [], [], [], [], [], [], []
            for t in range(self.Training_batch):
                # self.env.render()
                # Actor picks an action
                action, logp_t = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.env.step(action[0])
                next_state = self.reshape_state(next_state)
                # Memorize (state, next_states, action, reward, done, logp_ts) for training
                stock_states.append(state[0])
                commodity_states.append(state[1])
                wallet_states.append(state[2])

                next_stock_states.append(next_state[0])
                next_commodity_states.append(next_state[1])
                next_wallet_states.append(next_state[2])

                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                logp_ts.append(logp_t[0])
                # Update current state shape
                state = next_state
                score += reward
                if done:
                    self.episode += 1
                    average, SAVING = self.PlotModel(score, self.episode)
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score, average, SAVING))
                    self.writer.add_scalar(f'Workers:{1}/score_per_episode', score, self.episode)
                    self.writer.add_scalar(f'Workers:{1}/lr', self.lr, self.episode)
                    self.writer.add_scalar(f'Workers:{1}/average_score',  average, self.episode)
                    
                    state, done, score, SAVING = self.env.reset(), False, 0, ''
                    state = self.reshape_state(state)
                    if  average < -10000:
                        is_break = True

            self.replay(
                stock_states, commodity_states, wallet_states, 
                actions, rewards, dones, 
                next_stock_states, next_commodity_states, next_wallet_states, 
                logp_ts
            )
            if self.episode >= self.EPISODES:
                self.episode = 0
                break

            if is_break and self.episode >= 10:
                break

        self.env.close()

    def reshape_states_beta(self, state):
        stk_shp = 1
        for stk in stock_observation.shape:
            stk_shp = stk_shp*stk

        cmdt_shp = 1
        for cmdt in commodity_observation.shape:
            cmdt_shp = cmdt_shp*cmdt

        wllt_shp = 1
        for wllt in wallet_observation.shape:
            wllt_shp = wllt_shp*wllt
            
        stock_observation = np.reshape(stock_observation, (stk_shp,))
        commodity_observation = np.reshape(commodity_observation, (cmdt_shp,))
        # commodity_observation = np.reshape(commodity_observation, (1,) + commodity_observation.shape)
        wallet_observation = np.reshape(wallet_observation, (wllt_shp,))


    def reshape_state(self, state):
        stock_observation, commodity_observation, wallet_observation = state

        # Reshape and preprocess each component
        stock_observation = np.reshape(stock_observation, (1,) + stock_observation.shape)
        commodity_observation = np.reshape(commodity_observation, (1,) + commodity_observation.shape)
        wallet_observation = np.reshape(wallet_observation, (1,) + wallet_observation.shape)

        return [stock_observation, commodity_observation, wallet_observation]

    
    def test(self, test_episodes = 100):#evaluate
        self.load()
        latest_state = None
        for e in range(1):
            state = self.env.reset()
            state = self.reshape_state(state)

            # Predict an action from the actor model            done = False
            score = 0
            i = 0
            for i in range(self.test_steps):
            # while not done:
                # self.env.render()
                self.previous_state = copy.deepcopy(state)
                action = self.Actor.predict(state)[0]
                self.previous_actn = copy.deepcopy(action)
                state, reward, done, _ = self.env.step(action)
                state = self.reshape_state(state)
                latest_state = state
                score += reward
                if done:
                    average, SAVING = self.PlotModel(score, e, save=False)
                    print("episode: {}/{}, score: {}, average{}".format(e, test_episodes, score, average))
                    break
        self.env.close()
        self.latest_state = latest_state
        # print(self.env.t)

    def run_multiprocesses(self, num_worker = 4):
        self.episode = 0
        self.EPISODES = 80
        works, parent_conns, child_conns = [], [], []
        dfs = self.get_db_data_frames()
        for idx in range(num_worker):
            parent_conn, child_conn = Pipe()
            config = copy.deepcopy(self.env_config)
            work = EnvProcess(idx, child_conn, config, dfs)
            work.start()
            works.append(work)
            parent_conns.append(parent_conn)
            child_conns.append(child_conn)

        stock_states =   [[] for _ in range(num_worker)]
        commodity_states =   [[] for _ in range(num_worker)]
        wallet_states =   [[] for _ in range(num_worker)]
        next_stock_states =   [[] for _ in range(num_worker)]
        next_commodity_states =   [[] for _ in range(num_worker)]
        next_wallet_states =   [[] for _ in range(num_worker)]
        actions =       [[] for _ in range(num_worker)]
        rewards =       [[] for _ in range(num_worker)]
        dones =         [[] for _ in range(num_worker)]
        logp_ts =       [[] for _ in range(num_worker)]
        score =         [0 for _ in range(num_worker)]

        state = [0 for _ in range(num_worker)]
        worker_states = {
            "stock_tt": [],
            "commodity_tt": [],
            "wallet_tt": []
        }
        for worker_id, parent_conn in enumerate(parent_conns):
            temp_state = self.reshape_state(parent_conn.recv())
            worker_states["stock_tt"].append(temp_state[0])
            worker_states["commodity_tt"].append(temp_state[1])
            worker_states["wallet_tt"].append(temp_state[2])

        while self.episode < self.EPISODES:
            # get batch of action's and log_pi's
            action, logp_pi = self.act([
                np.vstack(worker_states["stock_tt"]), 
                np.vstack(worker_states["commodity_tt"]), 
                np.vstack(worker_states["wallet_tt"])
            ])
            
            for worker_id, parent_conn in enumerate(parent_conns):
                parent_conn.send(action[worker_id])
                actions[worker_id].append(action[worker_id])
                logp_ts[worker_id].append(logp_pi[worker_id])

            for worker_id, parent_conn in enumerate(parent_conns):
                next_state, reward, done, _ = parent_conn.recv()
                next_state = self.reshape_state(next_state)
                
                stock_states[worker_id].append(worker_states["stock_tt"][worker_id])
                commodity_states[worker_id].append(worker_states["commodity_tt"][worker_id])
                wallet_states[worker_id].append(worker_states["wallet_tt"][worker_id])
                next_stock_states[worker_id].append(next_state[0])
                next_commodity_states[worker_id].append(next_state[1])
                next_wallet_states[worker_id].append(next_state[2])
                rewards[worker_id].append(reward)
                dones[worker_id].append(done)
                state[worker_id] = next_state
                score[worker_id] += reward

                if done:
                    average, SAVING = self.PlotModel(score[worker_id], self.episode)
                    print("episode: {}/{}, worker: {}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, worker_id, score[worker_id], average, SAVING))
                    self.writer.add_scalar(f'Workers:{num_worker}/score_per_episode', score[worker_id], self.episode)
                    self.writer.add_scalar(f'Workers:{num_worker}/learning_rate', self.lr, self.episode)
                    self.writer.add_scalar(f'Workers:{num_worker}/average_score',  average, self.episode)
                    score[worker_id] = 0
                    if(self.episode < self.EPISODES):
                        self.episode += 1
                        
                        
            for worker_id in range(num_worker):
                if len(stock_states[worker_id]) >= self.Training_batch:
                    self.replay(
                        stock_states[worker_id], commodity_states[worker_id], wallet_states[worker_id],
                        actions[worker_id], rewards[worker_id], dones[worker_id], 
                        next_stock_states[worker_id], next_commodity_states[worker_id], next_wallet_states[worker_id], 
                        logp_ts[worker_id]
                    )

                    stock_states[worker_id] = []
                    commodity_states[worker_id] = []
                    wallet_states[worker_id] = []
                    next_stock_states[worker_id] = []
                    next_commodity_states[worker_id] = []
                    next_wallet_states[worker_id] = []
                    actions[worker_id] = []
                    rewards[worker_id] = []
                    dones[worker_id] = []
                    logp_ts[worker_id] = []

        # terminating processes after a while loop
        works.append(work)
        for work in works:
            work.terminate()
            print('TERMINATED:', work)
            work.join()

    def get_db_data_frames(self):
        cmmdties_df = pd.DataFrame(list(SimulatedCommodity.objects.all().values()))
        stck_df = pd.DataFrame(list(SimulatedStock.objects.all().values()))
        cmmdties_buffer_df = pd.DataFrame(list(SimulatedCommodityBuffer.objects.all().values()))
        stcks_buffer_df = pd.DataFrame(list(SimulatedStockBuffer.objects.all().values()))
        return [stck_df, cmmdties_df, stcks_buffer_df, cmmdties_buffer_df]

    def get_data_frames(self):
        the_current_time_step = self.env_config.get("the_current_time_step")
        __last_time_step = the_current_time_step + relativedelta(hours=505)
        the_current_time_step = pytz.utc.localize(datetime.strptime(str(the_current_time_step), '%Y-%m-%d %H:%M:%S'))
        str_time_step = str(the_current_time_step)

        __last_time_step = pytz.utc.localize(datetime.strptime(str(__last_time_step), '%Y-%m-%d %H:%M:%S'))
        str_last_time_step = str(__last_time_step)

        db_params = self.env_config.get("db_params")
        db_conn = psycopg2.connect(**db_params)
        cursor = db_conn.cursor()

        stcks_query = """
            SELECT *
            FROM simulated_stocks
        """
        cursor.execute(stcks_query)
        stck_data = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        stck_df = pd.DataFrame(stck_data, columns=column_names) 
        # self.stck_df = self.stck_df.to_dict(orient='records')

        stcks_buffer_query = f"""
            SELECT simulated_stocks_buffers.*, simulated_stocks.index
            FROM simulated_stocks_buffers 
            JOIN simulated_stocks on simulated_stocks.id = simulated_stocks_buffers.stock_id
            WHERE 
            captured_at >= '{str_time_step}' AND captured_at <= '{str_last_time_step}'
        """
        cursor.execute(stcks_buffer_query)
        stcks_buffer_data = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        stcks_buffer_df = pd.DataFrame(stcks_buffer_data, columns=column_names)
        # stcks_buffer_df = stcks_buffer_df.to_dict(orient='records')

        cmmdties_buffer_query = f"""
            SELECT simulated_commodities_buffers.*, simulated_commodities.index
            FROM simulated_commodities_buffers 
            JOIN simulated_commodities on simulated_commodities.id = simulated_commodities_buffers.commodity_id
            WHERE 
            captured_at >= '{str_time_step}' AND captured_at <= '{str_last_time_step}'
        """
        cursor.execute(cmmdties_buffer_query)
        cmmdties_buffer_data = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        cmmdties_buffer_df = pd.DataFrame(cmmdties_buffer_data, columns=column_names)
        # cmmdties_buffer_df = cmmdties_buffer_df.to_dict(orient='records')

        cmmdties_query = """
            SELECT *
            FROM simulated_commodities
        """
        cursor.execute(cmmdties_query)
        cmmdties_data = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        cmmdties_df = pd.DataFrame(cmmdties_data, columns=column_names)

        return [stck_df, cmmdties_df, stcks_buffer_df, cmmdties_buffer_df] 
            

if __name__ == "__main__":
    env_name = 'BipedalWalker-v3'
    agent = PPOAgent(env_name)
    agent.run_batch() # train as PPO
    #agent.run_multiprocesses(num_worker = 16)  # train PPO multiprocessed (fastest)
    # agent.test()