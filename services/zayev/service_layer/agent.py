import os

from services.zayev.environment.market_simulator import MarketSimulator

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
from keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
from keras import backend as K
import copy

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
    def __init__(self, current_time_step):
        # Initialization
        # Environment and PPO parameters
        self.env_name = "market"  
        db_params = {
            'database': environ.get("POSTGRES_DB"),
            'user': environ.get("POSTGRES_USER"),
            'password': environ.get("POSTGRES_PASSWORD"),
            'host': environ.get("DB_HOST"),
            'port': environ.get("DB_PORT"),
        }
        env_config = {
            "db_params": db_params, 
            "max_episode_steps": 150, 
            "the_current_time_step": current_time_step,
            "print_output": False
        }   
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
        
        state_size = stk_size +cmdt_siz + wllt_size


        self.state_size = state_size
        self.EPISODES = 200 # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 0 # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.epochs = 10 # training epochs
        self.shuffle = True
        self.Training_batch = 512
        #self.optimizer = RMSprop
        self.optimizer = Adam

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
        self.log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        self.std = np.exp(self.log_std)


    def act(self, state):
        # Use the network to predict the next action to take, using the model
        pred = self.Actor.predict(state)

        low, high = -10.0, 10.0 # -1 and 1 are boundaries of tanh
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

    def replay(self, states, actions, rewards, dones, next_states, logp_ts):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        logp_ts = np.vstack(logp_ts)

        # Get Critic network predictions 
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)

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
        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        c_loss = self.Critic.Critic.fit([states, values], target, epochs=self.epochs, verbose=0, shuffle=self.shuffle)

        # calculate loss parameters (should be done in loss, but couldn't find working way how to do that with disabled eager execution)
        pred = self.Actor.predict(states)
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
            # self.save()
            SAVING = "SAVING"
            # decreaate learning rate every saved model
            #self.lr *= 0.99
            #K.set_value(self.Actor.Actor.optimizer.learning_rate, self.lr)
            #K.set_value(self.Critic.Critic.optimizer.learning_rate, self.lr)
        else:
            SAVING = ""

        return self.average_[-1], SAVING
    
    def run_batch(self):
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        done, score, SAVING = False, 0, ''
        while True:
            # Instantiate or reset games memory
            states, next_states, actions, rewards, dones, logp_ts = [], [], [], [], [], []
            for t in range(self.Training_batch):
                self.env.render()
                # Actor picks an action
                action, logp_t = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.env.step(action[0])
                # Memorize (state, next_states, action, reward, done, logp_ts) for training
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.state_size]))
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                logp_ts.append(logp_t[0])
                # Update current state shape
                state = np.reshape(next_state, [1, self.state_size])
                score += reward
                if done:
                    self.episode += 1
                    average, SAVING = self.PlotModel(score, self.episode)
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score, average, SAVING))
                    self.writer.add_scalar(f'Workers:{1}/score_per_episode', score, self.episode)
                    self.writer.add_scalar(f'Workers:{1}/learning_rate', self.lr, self.episode)
                    self.writer.add_scalar(f'Workers:{1}/average_score',  average, self.episode)
                    
                    state, done, score, SAVING = self.env.reset(), False, 0, ''
                    state = np.reshape(state, [1, self.state_size])

            self.replay(states, actions, rewards, dones, next_states, logp_ts)
            if self.episode >= self.EPISODES:
                break

        self.env.close()


    def reshape_state(self, state):
        stock_observation, commodity_observation, wallet_observation = state

        # Reshape and preprocess each component
        stock_observation = np.reshape(stock_observation, (1,) + stock_observation.shape)
        commodity_observation = np.reshape(commodity_observation, (1,) + commodity_observation.shape)
        wallet_observation = np.reshape(wallet_observation, (1,) + wallet_observation.shape)

        return [stock_observation, commodity_observation, wallet_observation]

    
    def test(self, test_episodes = 100):#evaluate
        self.load()
        for e in range(1):
            state = self.env.reset()
            state = self.reshape_state(state)

            # Predict an action from the actor model            done = False
            score = 0
            i = 0
            for i in range(100):
            # while not done:
                # self.env.render()
                action = self.Actor.predict(state)[0]
                print(action)
                state, reward, done, _ = self.env.step(action)
                state = self.reshape_state(state)
                score += reward
                if done:
                    average, SAVING = self.PlotModel(score, e, save=False)
                    print("episode: {}/{}, score: {}, average{}".format(e, test_episodes, score, average))
                    break
        self.env.close()
            

if __name__ == "__main__":
    # newest gym fixed bugs in 'BipedalWalker-v2' and now it's called 'BipedalWalker-v3'
    env_name = 'BipedalWalker-v3'
    agent = PPOAgent(env_name)
    #agent.run_batch() # train as PPO
    #agent.run_multiprocesses(num_worker = 16)  # train PPO multiprocessed (fastest)
    agent.test()