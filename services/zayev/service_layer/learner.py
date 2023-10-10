from services.zayev.service_layer.parameter_server import ParameterServer
from services.zayev.service_layer.replay_buffer import ReplayBuffer
# from services.zayev.service_layer.zayev import Zayev
from keras.models import clone_model
import ray
import numpy as np
from keras.models import Model


# @ray.remote
class Learner:
    def __init__(
            self, zayev, #: Zayev, 
            replay_buffer: ReplayBuffer, 
            parameter_server: ParameterServer
    ):
        self.replay_buffer: ReplayBuffer = replay_buffer
        self.parameter_server = parameter_server
        self.Q, self.trainable = zayev.get_trainable_model()
        self.Q: Model = self.Q
        self.trainable: Model = self.trainable
        self.target_network = clone_model(self.Q)
        self.train_batch_size = zayev.train_batch_size  # You can set the default value accordingly
        self.total_collected_samples = 0
        self.samples_since_last_update = 0
        self.learning_starts = zayev.learning_starts
        self.obs_shape = zayev.obs_space.shape
        self.n_actions = zayev.n_actions
        self.stopped = False

    def send_weights(self):
        id = self.parameter_server.update_weights(self.trainable.get_weights())
        # ray.get(id)
    
    def start_learning(self):
        print("Learning starting...")
        self.send_weights()
        while not self.stopped:
            total_samples = self.replay_buffer.get_total_env_samples()
            # total_samples = ray.get(sid)
            if total_samples >= self.learning_starts:
                self.optimize()

    def optimize(self):
        samples = self.replay_buffer.sample(self.train_batch_size)
        if samples:
            N = len(samples)
            self.total_collected_samples += N
            self.samples_since_last_update += N
            ndim_obs = 1
            for s in self.obs_shape:
                if s:
                    ndim_obs *= s
            n_actions = self.n_actions
            obs = np.array([sample[0] for sample in samples]).reshape((N, ndim_obs))
            actions = np.array([sample[1] for sample in samples]).reshape((N, n_actions))
            rewards = np.array([sample[2] for sample in samples]).reshape((N,))
            last_obs = np.array([sample[3] for sample in samples]).reshape((N, ndim_obs))
            done_flags = np.array([sample[4] for sample in samples]).reshape((N,))
            gammas = np.array([sample[5] for sample in samples]).reshape((N,))
            masks = np.zeros((N, n_actions))
            masks[np.arange(N), actions] = 1
            dummy_labels = np.zeros((N,))
            # double DQN
            maximizer_a = np.argmax(self.Q.predict(last_obs),axis=1)
            target_network_estimates = self.target_network.predict(last_obs)
            q_value_estimates = np.array(
                [
                    target_network_estimates[
                        i,
                        maximizer_a[i]
                    ]
                    for i in range(N)
                ]).reshape((N,)
            )
            sampled_bellman = rewards + gammas * q_value_estimates * (1 - done_flags)
            
            # the obs can be something like the demand for the patty that day
            # masks is the action taken and sampled bellman is the expected rewards for that actions
            trainable_inputs = [obs, masks, sampled_bellman]
            self.trainable.fit(trainable_inputs, dummy_labels,verbose=0)
            self.send_weights()

            if self.samples_since_last_update > 500:
                self.target_network.set_weights(self.Q.get_weights())
                self.samples_since_last_update = 0
                return True
            
    def stop(self):
        self.stopped = True