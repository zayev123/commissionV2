import copy
from datetime import datetime
from math import ceil, floor
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
    def __init__(self, env_config: dict=None, data_frames = None):
        self.env_config = env_config
        self.n_step_stocks =  self.env_config.get("n_step_stocks")
        self.n_step_cmmdties =  self.env_config.get("n_step_cmmdties")
        self.is_live = self.env_config.get("is_live", False)
        self.no_of_stocks = self.env_config.get("no_of_stocks", None)
        self.no_of_cmmdts = self.env_config.get("no_of_cmmdts", None)
        self.preparing = self.env_config.get("preparing", False)
        self.max_episode_steps = self.env_config.get("max_episode_steps")
        self.__print_output = self.env_config.get("print_output")
        self.is_test = self.env_config.get("is_test", False)
        self.__initial_balance = 100000
        self.action_space = self.__get_actn_shape()
        (stock_observation_space, commodity_observation_space, volumes_observation_space) = self.__get_obs_shape() 
        self.wallet_state = self.__initial_balance
        self.observation_space = spaces.Tuple((stock_observation_space, commodity_observation_space, volumes_observation_space))
        self.data_frames = data_frames
        self.shares_data = {}
        self.change_in_shares = {}
        self.abs_change_in_shares = {}
        self.reset()
        
    
    def __get_actn_shape(self):
        wallet = 1
        no_of_actions = self.no_of_stocks + wallet

        # actns = [list(range(-1000,1001,1)) for i in range(self.no_of_stocks)]
        # num_possible_values = 2001  # 2001 values from -1000 to 1000 (inclusive)
        # actn_shape = gym.spaces.MultiDiscrete(np.array([[1001, 2],[1001, 2],[1001, 2],[1001, 2],[1001, 2]]))
        # amount_shape = gym.spaces.MultiDiscrete([1001,1001,1001,1001,1001])



        lower_bounds = [-1] * no_of_actions
        upper_bounds = [1] * no_of_actions

        actn_shape = spaces.Box(low=np.array(lower_bounds), high=np.array(upper_bounds))
                
        return actn_shape
    
    def __get_obs_shape(self):
        stock_observation_shape = (self.no_of_stocks, self.n_step_stocks)
        stock_lower_bounds = np.zeros(stock_observation_shape)
        stock_upper_bounds = np.full(stock_observation_shape, float('inf'))
        
        commodity_observation_shape = (self.no_of_cmmdts, self.n_step_cmmdties) 

        commodity_lower_bounds = np.zeros(commodity_observation_shape)
        commodity_upper_bounds = np.full(commodity_observation_shape, float('inf'))

        stock_observation_space = spaces.Box(low=stock_lower_bounds, high=stock_upper_bounds)

        commodity_observation_space = spaces.Box(low=commodity_lower_bounds, high=commodity_upper_bounds)

        volumes_observation_space = spaces.Box(low=stock_lower_bounds, high=stock_upper_bounds)


                    
        return (stock_observation_space, commodity_observation_space, volumes_observation_space)
    
    
    def get_prev_stock_prices_data(self, target_date):
        tme = target_date
        target_dates = [0 for nstk in range(self.n_step_stocks+1)]
        num_stocks = self.observation_space[0].shape[0]
        for nstk in range(self.n_step_stocks+1):
            target_dates[nstk] = tme # [date6, date5, ...]
            if self.is_live:
                for dyIndx in range(1,4):
                    tme: datetime = tme - relativedelta(days=dyIndx)
                    if tme.weekday() not in [5, 6]:
                        break
            else:
                tme = tme - relativedelta(hours=2, minutes=30)

        stock_prev_prices = np.zeros((num_stocks, self.n_step_stocks+1, 2))
        
        for dt in range(self.n_step_stocks+1): # [date6, date5, ...]
            a_date = target_dates[dt]
            a_stck_condition = self.stcks_buffer_df['captured_at'] == a_date # ==date6
            a_filtered_stck_df = self.stcks_buffer_df.loc[a_stck_condition]
            a_stcks_buffer_df = a_filtered_stck_df.to_dict(orient='records')
            for a_stck in a_stcks_buffer_df:
                indx = a_stck["index"] -1
                stock_prev_prices[indx][dt][0] = a_stck["price_snapshot"]
                stock_prev_prices[indx][dt][1] = a_stck["volume"]

        price_perc_changes = np.zeros((num_stocks, self.n_step_stocks, 2))
        copy_prev_prices = copy.deepcopy(stock_prev_prices)
        cpy_prev_prices = np.zeros((num_stocks, self.n_step_stocks, 2))
        for a_indx in range(num_stocks):
            for price_change_index in range(self.n_step_stocks): # [date6, date5, ...]
                last_price_index = price_change_index+1 # date5
                cpy_prev_prices[a_indx][price_change_index] = copy_prev_prices[a_indx][last_price_index]
                if copy_prev_prices[a_indx][last_price_index][0] != 0.0: # @date5 != 0
                    price_perc_changes[a_indx][price_change_index][0] = (
                        copy_prev_prices[a_indx][price_change_index][0] - copy_prev_prices[a_indx][last_price_index][0]
                    )/copy_prev_prices[a_indx][last_price_index][0]
                else:
                    price_perc_changes[a_indx][price_change_index][0] = 0.0
                
                if copy_prev_prices[a_indx][last_price_index][1] != 0.0: # @date5 != 0
                    price_perc_changes[a_indx][price_change_index][1] = (
                        copy_prev_prices[a_indx][price_change_index][1] - copy_prev_prices[a_indx][last_price_index][1]
                    )/copy_prev_prices[a_indx][last_price_index][1]
                else:
                    price_perc_changes[a_indx][price_change_index][1] = 0.0
            # print("stock_prev_prices", stock_prev_prices)
        return price_perc_changes
            
    def get_prev_cmmdty_prices_data(self, target_date):
        tme = target_date
        target_dates = [0 for nstk in range(self.n_step_cmmdties+1)]
        num_cmmdties = self.observation_space[1].shape[0]
        for ncmdt in range(self.n_step_cmmdties+1):
            target_dates[ncmdt] = tme
            if self.is_live:
                for dyIndx in range(1,4):
                    tme: datetime = tme - relativedelta(days=dyIndx)
                    if tme.weekday() not in [5, 6]:
                        break
            else:
                tme = tme - relativedelta(hours=2, minutes=30)

        cmmdty_prev_prices = np.zeros((num_cmmdties, self.n_step_cmmdties+1))
        
        for dt in range(self.n_step_cmmdties+1):
            a_date = target_dates[dt]
            a_cmmdty_condition = self.cmmdties_buffer_df['captured_at'] == a_date
            a_filtered_cmmdty_df = self.cmmdties_buffer_df.loc[a_cmmdty_condition]
            a_cmmdties_buffer_df = a_filtered_cmmdty_df.to_dict(orient='records')
            for a_cmmdty in a_cmmdties_buffer_df:
                indx = a_cmmdty["index"] -1
                cmmdty_prev_prices[indx][dt] = a_cmmdty["price_snapshot"]

        price_perc_changes = np.zeros((num_cmmdties, self.n_step_cmmdties))
        copy_prev_prices = copy.deepcopy(cmmdty_prev_prices)
        cpy_prev_prices = np.zeros((num_cmmdties, self.n_step_cmmdties))
        for a_indx in range(num_cmmdties):
            for price_change_index in range(self.n_step_cmmdties):
                last_price_index = price_change_index+1
                cpy_prev_prices[a_indx][price_change_index] = copy_prev_prices[a_indx][last_price_index]
                if copy_prev_prices[a_indx][last_price_index] != 0.0:
                    price_perc_changes[a_indx][price_change_index] = (copy_prev_prices[a_indx][price_change_index] - copy_prev_prices[a_indx][last_price_index])/copy_prev_prices[a_indx][last_price_index]
                else:
                    price_perc_changes[a_indx][price_change_index] = 0.0

        return price_perc_changes


    
    def get_the_state(self, init = False):
        num_stocks = self.observation_space[0].shape[0]
        num_commodities = self.observation_space[1].shape[0]
        num_stock_attributes = self.observation_space[0].shape[1]
        num_cmmdty_attributes = self.observation_space[1].shape[1]
        if init:
            stock_state = np.zeros((num_stocks, num_stock_attributes))
            commodity_state = np.zeros((num_commodities, num_cmmdty_attributes))
            volume_state = np.zeros((num_stocks, num_stock_attributes))
        else:
            (stock_state, commodity_state, volume_state) = self.state

        target_date = self.the_current_time_step
        stck_condition = self.stcks_buffer_df['captured_at'] == target_date
        filtered_stck_df = self.stcks_buffer_df.loc[stck_condition]
        # filtered_stck_df = filtered_stck_df[filtered_stck_df['id'] != 0]
        stcks_buffer_df = filtered_stck_df.to_dict(orient='records')
        prev_stck_prices_data = self.get_prev_stock_prices_data(target_date=target_date)

        cmmdty_condition = self.cmmdties_buffer_df['captured_at'] == target_date
        filtered_cmmdty_df = self.cmmdties_buffer_df.loc[cmmdty_condition]
        # filtered_cmmdty_df = filtered_cmmdty_df[filtered_cmmdty_df['id'] != 0]
        cmmdties_buffer_df = filtered_cmmdty_df.to_dict(orient='records')
        prev_cmmdty_prices_data = self.get_prev_cmmdty_prices_data(target_date=target_date)

        
        self.stock_data = {}
        # print(stcks_buffer_df)
        for d_stck in stcks_buffer_df:
            if d_stck["index"] not in self.stock_data:
                self.stock_data[d_stck["index"]] = d_stck

            if init:
                available_per_stock = self.__initial_balance/5
                if d_stck["price_snapshot"] == 0:
                    self.shares_data[d_stck["index"]] = 0
                else:
                    self.shares_data[d_stck["index"]] = 3

        for a_stck in stcks_buffer_df:
            indx = a_stck["index"] -1
            prev_prices = prev_stck_prices_data[indx]
            ij = 0
            # print(f"prev_prices{indx}", prev_prices)
            for prev_price in prev_prices:
                stock_state[indx][ij] = prev_price[0]
                volume_state[indx][ij] = prev_price[1]
                ij = ij + 1

            # print(f"stock_state[{indx}]", stock_state[indx])
        

        self.commodty_data = {}
        for d_cmmdty in cmmdties_buffer_df:
            if d_cmmdty["index"] not in self.commodty_data:
                self.commodty_data[d_cmmdty["index"]] = d_cmmdty

        for a_cmmdty in cmmdties_buffer_df:
            indx = a_cmmdty["index"] - 1
            prev_prices = prev_cmmdty_prices_data[indx]
            ij = 0
            for prev_price in prev_prices:
                commodity_state[indx][ij] = prev_price
                ij = ij + 1
        
        return (stock_state, commodity_state, volume_state)

    
    def reset(self):
        the_current_time_step = self.env_config.get("the_current_time_step")
        max_epi_len = 2*self.max_episode_steps
        if max_epi_len%2:
            minutes = 30
        else: 
            minutes = 0
        d_hrs = floor(2.5*(max_epi_len+5))

        
        __last_time_step = the_current_time_step + relativedelta(hours=d_hrs, minutes=minutes)
        self.the_current_time_step = pytz.utc.localize(datetime.strptime(str(the_current_time_step), '%Y-%m-%d %H:%M:%S'))
        if self.is_live:
            for dyIndx in range(0,4):
                self.the_current_time_step: datetime = self.the_current_time_step + relativedelta(days=dyIndx)
                if self.the_current_time_step.weekday() not in [5, 6]:
                    break
        self.__last_time_step = pytz.utc.localize(datetime.strptime(str(__last_time_step), '%Y-%m-%d %H:%M:%S'))
        str_time_step = str(self.the_current_time_step)
        self.__step_no = 0
        if self.data_frames!= None:
            self.stck_df = self.data_frames[0]
            self.cmmdties_df = self.data_frames[1]
            self.stcks_buffer_df = self.data_frames[2]
            self.cmmdties_buffer_df = self.data_frames[3]
        else:
            str_last_time_step = str(self.__last_time_step)
            db_params = self.env_config.get("db_params")
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
        #return <obs>
        self.state = self.get_the_state(init=True)
        return self.state
                           
    def __get_action_change(self, action, index):
        actions = action
        change_in_shares = actions[index]
        # divisor = 2

        # factor = change_in_shares // divisor
        # remainder = change_in_shares % divisor

        # if remainder == 0:
        #     change_in_shares = -1*factor
        # else:
        #     change_in_shares = factor
        return change_in_shares

    
    def step(self, action):
        # if the sign changes and a profit was made, then give it a reward, else dont do anything
        copied_action = copy.deepcopy(action)
        old_shares_data = copy.deepcopy(self.shares_data)
        # if self.is_live:
        #     for dyIndx in range(1,4):
        #         self.the_current_time_step: datetime = self.the_current_time_step + relativedelta(days=dyIndx)
        #         if self.the_current_time_step.weekday() not in [5, 6]:
        #             break
        # else:
        #     self.the_current_time_step = self.the_current_time_step + relativedelta(hours=2, minutes=30)
        self.__step_no = self.__step_no + 1
        (stck_state, cmmdty_state, volume_state) = copy.deepcopy(self.get_the_state())
        if self.__print_output:
            print("")
            print(f"stepping_a {action}")
            print(stck_state)
            print(cmmdty_state)
            print(volume_state)


        no_of_actions = len(action)
        no_of_actions = no_of_actions - 1
        if self.preparing:
            wallet_balance = self.__initial_balance
        else:
            wallet_balance = self.wallet_state
        old_portfolio_value = wallet_balance
        total_freed_capital = wallet_balance
        current_portfolio_value = wallet_balance
        new_portfolio_value = 0
        # total_value_bought = 0
        # total_value_sold = 0
        penalty = 0
        new_shares = {}
        flagged = False
        done = False

        reward = 0
        if self.change_in_shares:
            for indx in self.change_in_shares:
                reward = reward+ stck_state[indx-1][0]*self.change_in_shares[indx]
        
        for indx in range(no_of_actions):
            data_index = indx +1
            self.change_in_shares[data_index] = action[indx]

        # print(self.the_current_time_step, self.stock_data)
        for index in range(no_of_actions):
            data_index = index +1
            stck_price = self.stock_data[data_index]["price_snapshot"]
            chng = self.stock_data[data_index]["change"]
            old_stock_price = stck_price/(1+chng)
            current_no_of_shares = self.shares_data[data_index]

            old_portfolio_value = old_portfolio_value + current_no_of_shares*old_stock_price
            current_portfolio_value = current_portfolio_value + current_no_of_shares*stck_price

        for index in range(no_of_actions):
            data_index = index +1
            bid_vol = self.stock_data[data_index]["bid_vol"]
            action_amount = copied_action[index]*current_portfolio_value

            bid_price = self.stock_data[data_index]["bid_price"]
            offer_price = self.stock_data[data_index]["offer_price"]
            if action_amount > 0:
                change_in_shares = action_amount/offer_price       
            else:
                change_in_shares = action_amount/bid_price

            action[index] = change_in_shares
            if self.__print_output:
                print(f"change_in_shares for index: {index}", change_in_shares, current_no_of_shares, bid_vol)

        if self.__print_output:
            print("nw_ports", current_portfolio_value)
            print("old_actions", copied_action)
            print("new_actions", action)

        for index in range(no_of_actions):
            data_index = index +1
            stck_price = self.stock_data[data_index]["price_snapshot"]
            stck_vol = self.stock_data[data_index]["volume"]
            bid_vol = self.stock_data[data_index]["bid_vol"]
            bid_price = self.stock_data[data_index]["bid_price"]
            offer_vol = self.stock_data[data_index]["offer_vol"]
            offer_price = self.stock_data[data_index]["offer_price"]
            chng = self.stock_data[data_index]["change"]
            old_stock_price = stck_price/(1+chng)
            current_no_of_shares = self.shares_data[data_index]

            change_in_shares = action[index]
            
            if change_in_shares < 0:

                if bid_vol + change_in_shares < 0:
                    penalty = penalty + bid_vol + change_in_shares
                    flagged = True
                    # if self.__print_output:
                    #     print("break_2")
                    change_in_shares = -1*bid_vol

                if current_no_of_shares + change_in_shares < 0:
                    penalty = penalty + current_no_of_shares + change_in_shares
                    flagged = True
                    # if self.__print_output:
                    #     print("break_3")
                    change_in_shares = -1*current_no_of_shares + 1
                total_freed_capital = total_freed_capital + abs(change_in_shares)*bid_price
                new_shares[index] = change_in_shares + current_no_of_shares
            
            if change_in_shares == 0:
                new_shares[index] = current_no_of_shares

        for index in range(no_of_actions):
            data_index = index +1
            stck_price = self.stock_data[data_index]["price_snapshot"]
            stck_vol = self.stock_data[data_index]["volume"]
            bid_vol = self.stock_data[data_index]["bid_vol"]
            bid_price = self.stock_data[data_index]["bid_price"]
            offer_vol = self.stock_data[data_index]["offer_vol"]
            offer_price = self.stock_data[data_index]["offer_price"]
            chng = self.stock_data[data_index]["change"]
            old_stock_price = stck_price/(1+chng)
            current_no_of_shares = self.shares_data[data_index]

            change_in_shares = self.__get_action_change(action, index)
                
            if change_in_shares > 0:
                if offer_vol - change_in_shares < 0:
                    penalty = penalty + offer_vol - change_in_shares
                    flagged = True
                    # if self.__print_output:
                    #     print("break_1")
                    change_in_shares = offer_vol
                    
                capital_locked = change_in_shares*offer_price
                if capital_locked > total_freed_capital:
                    penalty = penalty + total_freed_capital - capital_locked
                    capital_locked = total_freed_capital
                    # if self.__print_output:
                    #     print("crack_2")
                change_in_shares = capital_locked/offer_price
                total_freed_capital = total_freed_capital - capital_locked

                # total_value_bought = total_value_bought + capital_locked*offer_price
            
                new_shares[index] = change_in_shares + current_no_of_shares

        if current_portfolio_value == 0:
            done = True
            flagged = True
            penalty = penalty -10
            # if self.__print_output:
            #     print("break_4")
        
        # freed_balance = total_freed_capital - total_value_bought
        self.wallet_state = total_freed_capital
        new_portfolio_value = total_freed_capital
        for index in range(no_of_actions):
            data_index = index +1
            if new_shares[index] <= 0:
                new_shares[index] = 1
            self.shares_data[data_index] = new_shares[index]
            new_portfolio_value = new_portfolio_value + self.shares_data[data_index]*self.stock_data[data_index]["price_snapshot"]

        

        if self.is_live:
            for dyIndx in range(1,4):
                self.the_current_time_step: datetime = self.the_current_time_step + relativedelta(days=dyIndx)
                if self.the_current_time_step.weekday() not in [5, 6]:
                    break
        else:
            self.the_current_time_step = self.the_current_time_step + relativedelta(hours=2, minutes=30)
        
        (stck_state, cmmdty_state, volume_state) = copy.deepcopy(self.get_the_state())
        self.state = (stck_state, cmmdty_state, volume_state)
        info = {}

        if old_shares_data:
            for shr_indx, shr_nmbrs in self.shares_data.items():
                self.abs_change_in_shares[shr_indx] = self.shares_data[shr_indx] - old_shares_data[shr_indx]

        if self.is_test:
            (is_valid_output, output) = self.test_output_validity(
                self.the_current_time_step, 
                self.shares_data, 
                old_shares_data,
                stck_state,
                self.stcks_buffer_df 
            )

            if not is_valid_output:
                print("test output:", output)

        if self.__step_no > self.max_episode_steps:
            done = True

        if self.__print_output:
            print(f"stepping_b: {flagged} {old_portfolio_value}, {current_portfolio_value}, {new_portfolio_value} {penalty} {reward}")
            print(self.stock_data)
            print(self.shares_data)
            print("")

        return self.state, reward, done, info
    
    def test_output_validity(self, new_time, new_shares, old_shares, new_stock_state, dfs):
        new_stck_condition = dfs['captured_at'] == new_time
        new_filtered_stck_df = dfs.loc[new_stck_condition]
        new_stcks_buffer_df = new_filtered_stck_df.to_dict(orient='records')
        new_stock_data = {}
        for new_data in new_stcks_buffer_df:
            if new_data["index"] not in new_stock_data:
                new_stock_data[new_data["index"]] = new_data
        if self.is_live:
            for dyIndx in range(1,4):
                previous_time_step: datetime = new_time - relativedelta(days=dyIndx)
                if previous_time_step.weekday() not in [5, 6]:
                    break
        else:
            previous_time_step = new_time - relativedelta(hours=2, minutes=30)
        old_stck_condition = dfs['captured_at'] == previous_time_step
        old_filtered_stck_df = dfs.loc[old_stck_condition]
        old_stcks_buffer_df = old_filtered_stck_df.to_dict(orient='records')
        old_stock_data = {}
        for old_data in old_stcks_buffer_df:
            if old_data["index"] not in old_stock_data:
                old_stock_data[old_data["index"]] = old_data
        change_in_shares = {}
        for indx in new_shares:
            change_in_shares[indx] = new_shares[indx] - old_shares[indx]
        
        for indx, chng in change_in_shares.items():
            if chng>0:
                if chng > new_stock_data[indx]["offer_vol"]:
                    return (False, f"indx, {indx} chng > offer_vol {chng}> {new_stock_data[indx]['offer_vol']}")
            if chng<0:
                if abs(chng) > new_stock_data[indx]["bid_vol"]:
                    return (False, f"indx, {indx} chng > bid_vol {chng}> {new_stock_data[indx]['bid_vol']}")
                
        
        for indx in new_stock_data:
            new_price = new_stock_data[indx]["price_snapshot"]
            old_price = old_stock_data[indx]["price_snapshot"]
            change = (new_price - old_price)/old_price
            if change!=new_stock_state[indx-1][0]:
                return (False, f"indx, {indx} change != stock_state {change} != {new_stock_state[indx-1][0]}")

        return (True, "success")
    
    def get_raw_acts(self, target_date):
        stck_condition = self.stcks_buffer_df['captured_at'] == target_date
        filtered_stck_df = self.stcks_buffer_df.loc[stck_condition]
        # filtered_stck_df = filtered_stck_df[filtered_stck_df['id'] != 0]
        stcks_buffer_df = filtered_stck_df.to_dict(orient='records')
        acts = {}
        for buff in stcks_buffer_df:
            indx = buff["index"]
            if indx not in acts:
                o4_day_ma = buff["14_day_ma"]
                curr_prc = buff["price_snapshot"]
                if o4_day_ma != 0 and curr_prc != 0:
                    chng = (curr_prc - o4_day_ma)/o4_day_ma
                else:
                    chng = 0
                acts[indx] = chng
        return acts

    
    def get_mov_avg_actions(self):
        target_date = self.the_current_time_step
        acts = self.get_raw_acts(target_date)
        old_date = None
        for dyIndx in range(1,4):
            old_date: datetime = target_date - relativedelta(days=dyIndx)
            if old_date.weekday() not in [5, 6]:
                break

        old_acts =   self.get_raw_acts(old_date)              
        
        avg_acts = np.zeros(self.no_of_stocks)
        for kindx, chng in acts.items():
            wght = 1
            # if old_acts[kindx]*acts[kindx] < 0:
            #     wght = 5
            # else:
            #     wght = 0
            avg_acts[kindx-1] = chng*wght

        # print("old_acts", old_acts)
        # print("avg_acts", acts)

        min_val = -10
        max_val = 10
        # scaled_array = avg_acts * ((max_val - min_val) / (avg_acts.max() - avg_acts.min()))
        scaled_array = (avg_acts - avg_acts.min()) / (avg_acts.max() - avg_acts.min()) * (max_val - min_val) + min_val
        scaled_array_plus_10 = scaled_array + 10
        return scaled_array_plus_10



