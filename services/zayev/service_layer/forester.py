from services.zayev.environment.market_simulator import MarketSimulator
import numpy as np
import tensorflow as tf
from copy import deepcopy

class Forester:
    def __init__(self, market: MarketSimulator, start_learning_after_steps = 6):
        self.market = market
        self.sim_steps = self.market.max_episode_steps
        self.start_learning_after_steps = start_learning_after_steps

        self.eps = 0.2

        self.train_input = []
        self.train_output = []

        self.test_input = []
        self.test_output = []

    def act_random(self):
        random_actions = np.random.uniform(-1, 1, 6)
        scaled_actions = self.scale_actions(random_actions)
        return scaled_actions

    def scale_actions(self, x):
        neg_mask = x < 0
        neg_sum = tf.reduce_sum(tf.boolean_mask(x, neg_mask))

        ratio_x = tf.abs(tf.cast(1.0, dtype=tf.float64)) / tf.abs(neg_sum)
        x = tf.where(neg_mask & (neg_sum>-1), x * ratio_x, x)
        neg_sum = tf.reduce_sum(tf.boolean_mask(x, neg_mask))

        pos_mask = x > 0
        pos_values_sum = tf.reduce_sum(tf.boolean_mask(x, pos_mask))

        ratio_a = tf.abs(-1 - neg_sum) / tf.abs(neg_sum)
        ratio_b = 1- ratio_a

        x = tf.where(neg_mask & (neg_sum<-1), x * ratio_b, x)

        # Reduce the positive sum if it's greater than the adjusted neg_sum
        ratio_c = tf.abs(tf.maximum(-1.0, neg_sum)) / pos_values_sum

        x = tf.where(pos_mask, x * ratio_c, x)
        return x.numpy()
    
    def classify_output(self, raw_output):
        gen_output = deepcopy(raw_output)
        gen_output = gen_output*100
        data_len = len(gen_output)
        classified_output = np.zeros(data_len)
        for indx in range(data_len):
            if gen_output[indx] <-10:
                classified_output[indx] = 0
            elif gen_output[indx] <-9:
                classified_output[indx] = 1
            elif gen_output[indx] <-8:
                classified_output[indx] = 2
            elif gen_output[indx] <-7:
                classified_output[indx] = 3
            elif gen_output[indx] <-6:
                classified_output[indx] = 4
            elif gen_output[indx] <-5:
                classified_output[indx] = 5
            elif gen_output[indx] <-4:
                classified_output[indx] = 6
            elif gen_output[indx] <-3:
                classified_output[indx] = 7
            elif gen_output[indx] <-2:
                classified_output[indx] = 8
            elif gen_output[indx] <-1:
                classified_output[indx] = 9
            elif gen_output[indx] <-0:
                classified_output[indx] = 10
            elif gen_output[indx] <1:
                classified_output[indx] = 11
            elif gen_output[indx] <2:
                classified_output[indx] = 12
            elif gen_output[indx] <3:
                classified_output[indx] = 13
            elif gen_output[indx] <4:
                classified_output[indx] = 14
            elif gen_output[indx] <5:
                classified_output[indx] = 15
            elif gen_output[indx] <6:
                classified_output[indx] = 16
            elif gen_output[indx] <7:
                classified_output[indx] = 17
            elif gen_output[indx] <8:
                classified_output[indx] = 18
            elif gen_output[indx] <9:
                classified_output[indx] = 19
            elif gen_output[indx] <10:
                classified_output[indx] = 20
            else:
                classified_output[indx] = 21
        # print("----------")
        # print("gen_output", gen_output)
        # print(data_len)
        # print(classified_output)
        # print("----------")
        return classified_output
    
    @staticmethod
    def get_flattened_states(state):
        (curr_stock_state, curr_commodity_state, curr_volume_state) = state
        v1 = curr_stock_state.flatten()
        v2 = curr_commodity_state.flatten()
        v3 = curr_volume_state.flatten()
        input_data = np.concatenate((v1, v2, v3))
        return input_data

    
    def make_forest_data(self):
        self.train_input = []
        self.train_output = []
        self.test_input = []
        self.test_output = []

        for a_stp in range(self.sim_steps):
            if a_stp > self.start_learning_after_steps:
                (curr_stock_state, curr_commodity_state, curr_volume_state) = self.market.state
                v1 = curr_stock_state.flatten()
                v2 = curr_commodity_state.flatten()
                v3 = curr_volume_state.flatten()
                input_gen_data = deepcopy(np.concatenate((v1, v2, v3)))
                # print("input_gen_data", input_gen_data, "input_gen_data")

            rand_acts = self.act_random()
            self.market.step(rand_acts)

            if a_stp > self.start_learning_after_steps:
                (new_stock_state, new_commodity_state, new_volume_state) = self.market.state
                new_stock_prices = new_stock_state[:, 0]
                result_gen_data = np.array(new_stock_prices)
                classified_output = self.classify_output(result_gen_data)

                thresh = np.random.random()
                if thresh >= self.eps: 
                #     print(f"eps {a_stp}")
                #     self.test_input.append(input_gen_data)
                #     self.test_output.append(classified_output)
                # else:
                    self.train_input.append(input_gen_data)
                    self.train_output.append(classified_output)
                    #     print("input_ken_data")
                else:
                    self.test_input.append(deepcopy(input_gen_data))
                    self.test_output.append(deepcopy(classified_output))
                # print(self.train_input)
                # print(self.train_output)