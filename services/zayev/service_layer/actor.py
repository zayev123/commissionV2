from collections import deque
import ray
import numpy as np
from services.zayev.service_layer.parameter_server import ParameterServer
from services.zayev.service_layer.replay_buffer import ReplayBuffer
# from services.zayev.service_layer.zayev import Zayev


# @ray.remote
class Actor:
    def __init__(
        self,
        zayev, #: Zayev,
        actor_id,
        replay_buffer,
        parameter_server,
        eps,
        eval=False,
    ):
        self.actor_id = actor_id
        self.replay_buffer: ReplayBuffer = replay_buffer
        self.parameter_server: ParameterServer = parameter_server
        self.eps = eps
        self.eval = eval
        self.Q = zayev.get_Q_network()
        self.env = zayev.env
        self.local_buffer = []
        self.obs_shape = zayev.obsevation_shape
        self.n_actions = zayev.n_actions
        self.multi_step_n = zayev.n_step
        self.q_update_freq = zayev.q_update_freq
        self.send_experience_freq = zayev.send_experience_freq
        self.gamma = zayev.gamma
        self.continue_sampling = True
        self.cur_episodes = 0
        self.cur_steps = 0

    def update_q_network(self):
        if self.eval:
            pid = self.parameter_server.get_eval_weights()
        else:
            pid = self.parameter_server.get_weights()
        new_weights = pid
        if new_weights:
            self.Q.set_weights(new_weights)

    def stop(self):
        self.continue_sampling = False

    def sample(self):
        self.update_q_network()
        observation = self.env.reset()
        # print("pleasy1", observation)
        # print("pleasy2", observation)
        episode_reward = 0
        episode_length = 0
        n_step_buffer = deque(maxlen=self.multi_step_n + 1)
        while self.continue_sampling:
            # here 2
            if type(observation) is tuple:
                observation = np.array(observation[0])
            action = self.get_action (observation)
            abcd = self.env.step(action)
            next_observation, reward, \
            done, truncated, info = abcd
            n_step_buffer.append((observation, action, reward, done))
            if len(n_step_buffer) == self.multi_step_n + 1:
                self.local_buffer.append(
                self.get_n_step_trans(n_step_buffer))
            self.cur_steps += 1
            episode_reward += reward
            episode_length += 1
            if done:
                if self.eval:
                    break
                next_observation = self.env.reset()
                if len(n_step_buffer) > 1:
                    self.local_buffer.append(self.get_n_step_trans(n_step_buffer))
                self.cur_episodes += 1
                episode_reward = 0
                episode_length = 0
            observation = next_observation
            if self.cur_steps % \
                    self.send_experience_freq == 0 \
                    and not self.eval:
                self.send_experience_to_replay()
            if self.cur_steps % \
                    self.q_update_freq == 0 and not self.eval:
                self.update_q_network()
        return episode_reward
    
    def get_action(self, observation):
        ## here 1
        observation = observation.reshape((1, -1))
        q_estimates = self.Q.predict(observation)[0]
        if np.random.uniform() <= self.eps:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(q_estimates)
        return action
    
    def get_n_step_trans(self, n_step_buffer):
        gamma = self.zayev
        discounted_return = 0
        cum_gamma = 1
        for trans in list(n_step_buffer)[:-1]:
            _, _, reward, _ = trans
            discounted_return += cum_gamma * reward
            cum_gamma *= gamma
        observation, action, _, _ = n_step_buffer[0]
        last_observation, _, _, done = n_step_buffer[-1]
        experience = (observation, action, discounted_return,
        last_observation, done, cum_gamma)
        return experience
    
    def send_experience_to_replay(self):
        rf = self.replay_buffer.add(self.local_buffer)
        # ray.wait([rf])
        self.local_buffer = []