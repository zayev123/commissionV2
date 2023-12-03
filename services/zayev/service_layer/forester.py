from services.zayev.environment.market_simulator import MarketSimulator
import numpy as np
import tensorflow as tf

class Forester:
    def __init__(self, market: MarketSimulator):
        self.market = market

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