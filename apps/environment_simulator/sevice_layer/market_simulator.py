from datetime import datetime
import gymnasium as gym
from matplotlib.dates import relativedelta
from gymnasium import spaces
import numpy as np
from asgiref.sync import sync_to_async
import asyncio
import psycopg2
import pandas as pd
import pytz

class MarketSimulator(gym.Env):
    def __init__(self, env_config: dict=None):
        

        # There are two actions, first will get reward of 1, second reward of -1. 
        __time_step  = datetime(year=1995, month=1, day=1, hour=10)
        __last_time_step = __time_step + relativedelta(hours=505)
        self.__time_step = pytz.utc.localize(datetime.strptime(str(__time_step), '%Y-%m-%d %H:%M:%S'))
        str_time_step = str(self.__time_step)
        self.__step_no = 0
        self.max_episode_steps = 200

        self.__last_time_step = pytz.utc.localize(datetime.strptime(str(__last_time_step), '%Y-%m-%d %H:%M:%S'))
        str_last_time_step = str(self.__last_time_step)

        db_params = env_config.get("db_params")
        self.db_conn = psycopg2.connect(**db_params)
        self.cursor = self.db_conn.cursor()

        stcks_query = """
            SELECT *
            FROM simulated_stocks
        """
        self.cursor.execute(stcks_query)
        stck_data = self.cursor.fetchall()
        column_names = [desc[0] for desc in self.cursor.description]
        self.stck_df = pd.DataFrame(stck_data, columns=column_names) 
        # self.stck_df = self.stck_df.to_dict(orient='records')

        stcks_buffer_query = f"""
            SELECT simulated_stocks_buffers.*, simulated_stocks.index
            FROM simulated_stocks_buffers 
            JOIN simulated_stocks on simulated_stocks.id = simulated_stocks_buffers.stock_id
            WHERE 
            captured_at >= '{str_time_step}' AND captured_at <= '{str_last_time_step}'
        """
        self.cursor.execute(stcks_buffer_query)
        stcks_buffer_data = self.cursor.fetchall()
        column_names = [desc[0] for desc in self.cursor.description]
        self.stcks_buffer_df = pd.DataFrame(stcks_buffer_data, columns=column_names)
        # self.stcks_buffer_df = self.stcks_buffer_df.to_dict(orient='records')

        cmmdties_buffer_query = f"""
            SELECT simulated_commodities_buffers.*, simulated_commodities.index
            FROM simulated_commodities_buffers 
            JOIN simulated_commodities on simulated_commodities.id = simulated_commodities_buffers.commodity_id
            WHERE 
            captured_at >= '{str_time_step}' AND captured_at <= '{str_last_time_step}'
        """
        self.cursor.execute(cmmdties_buffer_query)
        cmmdties_buffer_data = self.cursor.fetchall()
        column_names = [desc[0] for desc in self.cursor.description]
        self.cmmdties_buffer_df = pd.DataFrame(cmmdties_buffer_data, columns=column_names)
        # self.cmmdties_buffer_df = self.cmmdties_buffer_df.to_dict(orient='records')

        cmmdties_query = """
            SELECT *
            FROM simulated_commodities
        """
        self.cursor.execute(cmmdties_query)
        cmmdties_data = self.cursor.fetchall()
        column_names = [desc[0] for desc in self.cursor.description]
        self.cmmdties_df = pd.DataFrame(cmmdties_data, columns=column_names) 
        # self.cmmdties_df = self.cmmdties_df.to_dict(orient='records')

        self.cursor.close()
        self.db_conn.close()

        
        self.__initial_balance = 100000
        self.action_space = self.__get_actn_shape()
        (stock_observation_space, commodity_observation_space, wallet_observation_space) = self.__get_obs_shape() 
        self.observation_space = spaces.Tuple((stock_observation_space, commodity_observation_space, wallet_observation_space))
        self.state = self.__get_state(init=True)
        
    
    def __get_actn_shape(self):
        no_of_stocks = 5



        lower_bounds = [0] * no_of_stocks
        upper_bounds = [float('inf')] * no_of_stocks

        actn_shape = spaces.Box(low=np.array(lower_bounds), high=np.array(upper_bounds))
                
        return actn_shape
    
    def __get_obs_shape(self):
        no_of_stocks = 5
        stock_observation_shape = (no_of_stocks, 8)
        stock_lower_bounds = np.zeros(stock_observation_shape)
        stock_upper_bounds = np.full(stock_observation_shape, float('inf'))
        
        no_of_cmmdts = 6
        commodity_observation_shape = (no_of_cmmdts, 1) 

        commodity_lower_bounds = np.zeros(commodity_observation_shape)
        commodity_upper_bounds = np.full(commodity_observation_shape, float('inf'))

        stock_observation_space = spaces.Box(low=stock_lower_bounds, high=stock_upper_bounds)

        commodity_observation_space = spaces.Box(low=commodity_lower_bounds, high=commodity_upper_bounds)

        wallet_observation_space = spaces.Box(low=0, high=float('inf'), shape=(1,))

                    
        return (stock_observation_space, commodity_observation_space, wallet_observation_space)
    
    def __get_state(self, init = False):
        num_stocks = self.observation_space[0].shape[0]
        num_commodities = self.observation_space[1].shape[0]
        num_stock_attributes = self.observation_space[0].shape[1]
        if init:
            stock_state = np.zeros((num_stocks, num_stock_attributes))
            commodity_state = np.zeros((num_commodities, 1))
            wallet_state = np.array([self.__initial_balance])
        else:
            (stock_state, commodity_state, wallet_state) = self.state

        target_date = self.__time_step
        stck_condition = self.stcks_buffer_df['captured_at'] == target_date
        filtered_stck_df = self.stcks_buffer_df.loc[stck_condition]
        stcks_buffer_df = filtered_stck_df.to_dict(orient='records')

        cmmdty_condition = self.cmmdties_buffer_df['captured_at'] == target_date
        filtered_cmmdty_df = self.cmmdties_buffer_df.loc[cmmdty_condition]
        cmmdties_buffer_df = filtered_cmmdty_df.to_dict(orient='records')

        for a_stck in stcks_buffer_df:
            indx = a_stck["index"] -1
            stock_state[indx][0] = a_stck["price_snapshot"]
            stock_state[indx][1] = a_stck["volume"]
            stock_state[indx][2] = a_stck["bid_vol"]
            stock_state[indx][3] = a_stck["bid_price"]
            stock_state[indx][4] = a_stck["offer_vol"]
            stock_state[indx][5] = a_stck["offer_price"]
            if init:
                stock_state[indx][6] = 0
                stock_state[indx][7] = 0
        

        for a_cmmdty in cmmdties_buffer_df:
            indx = a_cmmdty["index"] - 1
            commodity_state[indx] = a_cmmdty["price_snapshot"]
                    
        return (stock_state, commodity_state, wallet_state)

        
    
    def reset(self):
        self.state = self.__get_state(init=True)
        #return <obs>
        return self.state
                           
    def step(self, action):
        # if we took an action, we were in state 1
        self.__time_step = self.__time_step + relativedelta(hours=2, minutes=30)
        self.__step_no = self.__step_no + 1
        (stock_state, commodity_state, wallet_state) = self.__get_state()

        actions = action
        no_of_actions = len(actions)
        wallet_balance = wallet_state[0]
        old_portfolio_value = wallet_balance
        new_portfolio_value = 0
        total_value_bought = 0
        total_value_sold = 0
        penalty = 0
        new_shares = {}
        flagged = False
        done = False
        for index in range(no_of_actions):
            stck_price = stock_state[index][0]
            stck_vol = stock_state[index][1]
            bid_vol = stock_state[index][2]
            bid_price = stock_state[index][3]
            offer_vol = stock_state[index][4]
            offer_price = stock_state[index][5]
            old_no_of_shares = stock_state[index][6]
            old_portfolio_value = old_portfolio_value + old_no_of_shares*stck_price

            change_in_shares = actions[index]
            if change_in_shares > 0:
                if change_in_shares - offer_vol < 0:
                    penalty = -10
                    flagged = True
                    break
                total_value_bought = total_value_bought + change_in_shares*offer_price
            
            if change_in_shares < 0:
                if abs(change_in_shares) - bid_vol < 0:
                    penalty = -10
                    flagged = True
                    break
                elif abs(change_in_shares) > old_no_of_shares:
                    penalty = -10
                    flagged = True
                    break
                total_value_sold = total_value_sold + abs(change_in_shares)*bid_price
            new_shares[index] = change_in_shares + old_no_of_shares

        if old_portfolio_value == 0:
            done = True
            flagged = True
            penalty = -10
        total_freed_capital = total_value_sold + wallet_balance
        if total_value_bought > total_freed_capital and not flagged:
            flagged = True
            penalty = -10
        
        elif not flagged:
            freed_balance = total_freed_capital - total_value_bought
            wallet_state[0] = freed_balance
            for index in range(no_of_actions):
                stock_state[index][6] = stock_state[index][7]
                stock_state[index][7] = new_shares[index]
                new_portfolio_value = new_portfolio_value + new_shares[index]*stock_state[index][0]

        reward = 0
        if not flagged:
            reward = new_portfolio_value - old_portfolio_value
        else:
            reward = penalty

        self.state = (stock_state, commodity_state, wallet_state)
        info = {}

        if self.__step_no == 200:
            done = True

        return self.state, reward, done, info


# class MarketSimulator(gym.Env):
#     def __init__(self, env_config=None):
#        # There are two actions, first will get reward of 1, second reward of -1. 
#         self.action_space = spaces.Box(low=np.array([0,0,-2,0,1]),
#                                high=np.array([1,1,2,1,20]),
#                                dtype=np.float32)     #<gym.Space>
#         self.observation_space = spaces.Discrete(2) #<gym.Space>
    
#     def reset(self):
#         state = 0
#         #return <obs>
#         return state
                           
#     def step(self, action):

#         # if we took an action, we were in state 1
#         state = 1
    
#         sum = action[0] + action[1] + action[2] + action[3] + action[4]
#         if sum == 5:
#             reward = 1
#         else:
#             reward = -1
            
#         # regardless of the action, game is done after a single step
#         done = True
#         info = {}
#         # return <obs>, <reward: float>, <done: bool>, <info: dict>

#         return state, reward, done, info  