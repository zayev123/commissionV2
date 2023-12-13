from datetime import datetime, date

from matplotlib.dates import relativedelta
import pytz
from apps.environment.models.commodity import Commodity, CommodityBuffer
from apps.environment.models.stock import Stock, StockBuffer
from psx import stocks
import yfinance as yf
import psycopg2
import pandas as pd
import os
from joblib import dump

class DataTransformer:
    def __init__(self, env_config: dict):
        self.env_config = env_config
        the_current_time_step = env_config.get("the_current_time_step")
        self.max_epi_len = env_config.get("max_episode_steps")
        last_time_step = the_current_time_step + relativedelta(days=self.max_epi_len)
        self.the_current_time_step = pytz.utc.localize(datetime.strptime(str(the_current_time_step), '%Y-%m-%d %H:%M:%S'))
        self.last_time_step = pytz.utc.localize(datetime.strptime(str(last_time_step), '%Y-%m-%d %H:%M:%S'))
        today = pytz.utc.localize(datetime.now())
        self.added_days = 500
        rem_days = today - self.last_time_step
        if rem_days.days - 500 <0:
            self.added_days = rem_days.days

    @staticmethod
    def retreive_stocks_buffers():
        all_stocks = Stock.objects.all()

        stock_syms = {}

        for a_stock in all_stocks:
            if a_stock.symbol not in stock_syms:
                stock_syms[a_stock.symbol] = {
                    "object": a_stock,
                    "snapshots": {},
                    "last_date": None
                }

        snapshots = StockBuffer.objects.select_related("stock").all()
        for snapshot in snapshots:
            snp_stck = snapshot.stock
            snp_stck_symbl = snp_stck.symbol
            if snp_stck_symbl in stock_syms:
                snp_data = stock_syms[snp_stck_symbl]
                stck_snps = snp_data["snapshots"]
                if snapshot.captured_at not in stck_snps:
                    stck_snps[snapshot.captured_at.strftime('%Y-%m-%d %H:%M:%S')] = snapshot
                    if not snp_data["last_date"]:
                        snp_data["last_date"] = snapshot.captured_at.date()

        # tst = 1
        for stck_sym, stck_data in stock_syms.items():
            new_snapshots = []
            last_date: datetime = stck_data["last_date"]
            if last_date is None:
                start_from = date(2020, 1, 1)
            else:
                start_from = last_date
            print(last_date, stck_sym, start_from)
            end_date = pytz.utc.localize(datetime.now() + relativedelta(days=2)).date()
            data = stocks(stck_sym, start=start_from, end=end_date)
            data_points_len = len(data)
            stck_snpshts = stck_data["snapshots"]
            the_stock: Stock = stck_data["object"]
            last_data_point = None
            for digi in range(data_points_len):
                data_point = data.iloc[digi]
                time_point = data_point.name.to_pydatetime()
                time_point_str = data_point.name.strftime('%Y-%m-%d %H:%M:%S')
                if time_point_str not in stck_snpshts:
                    zone_point_time = pytz.utc.localize(time_point)
                    nw_price = data_point.Close
                    if last_data_point is None:
                        change = 0
                    else:
                        old_price = last_data_point.Close
                        if old_price and old_price > 0:
                            change = (nw_price-old_price)/old_price
                        else:
                            change = 0
                    new_snapshots.append(StockBuffer(
                        stock = the_stock,
                        captured_at = zone_point_time,
                        price_snapshot = nw_price,
                        change = change,
                        volume = data_point.Volume,
                        bid_vol = 100000000,
                        bid_price = nw_price,
                        offer_vol = 100000000,
                        offer_price = nw_price,
                        open = data_point.Open,
                        close = nw_price,
                        high = data_point.High,
                        low = data_point.Low
                    ))
                last_data_point = data_point
            StockBuffer.objects.bulk_create(new_snapshots)

            # tst = tst + 1
            # if tst == 4:
            #     break

    @staticmethod
    def retreive_commodities_buffers():
        all_commodities = Commodity.objects.all()

        commodity_syms = {}

        for a_commodity in all_commodities:
            if a_commodity.symbol not in commodity_syms:
                commodity_syms[a_commodity.symbol] = {
                    "object": a_commodity,
                    "snapshots": {},
                    "last_date": None
                }

        snapshots = CommodityBuffer.objects.select_related("commodity").all()
        for snapshot in snapshots:
            snp_cmmdty = snapshot.commodity
            snp_cmmdty_symbl = snp_cmmdty.symbol
            if snp_cmmdty_symbl in commodity_syms:
                cmmdty_data = commodity_syms[snp_cmmdty_symbl]
                cmmdty_snps = cmmdty_data["snapshots"]
                if snapshot.captured_at not in cmmdty_snps:
                    cmmdty_snps[snapshot.captured_at.strftime('%Y-%m-%d %H:%M:%S')] = snapshot
                    if not cmmdty_data["last_date"]:
                        cmmdty_data["last_date"] = snapshot.captured_at.date()

        # tst = 1
        for cmmdty_sym, cmmdty_data in commodity_syms.items():
            new_snapshots = []
            last_date: datetime = cmmdty_data["last_date"]
            if last_date is None:
                start_from = date(2019, 12, 30)
            else:
                start_from = date(2019, 12, 30)
                # start_from = last_date
            print(last_date, cmmdty_sym, start_from)
            end_date = pytz.utc.localize(datetime.now() + relativedelta(days=2)).date()
            data = yf.download(cmmdty_sym, start=start_from, end=end_date)
            data_points_len = len(data)
            cmmdty_snpshts = cmmdty_data["snapshots"]
            the_commodity: Commodity = cmmdty_data["object"]
            last_data_point = None
            for digi in range(data_points_len):
                data_point = data.iloc[digi]
                time_point = data_point.name.to_pydatetime()
                time_point_str = data_point.name.strftime('%Y-%m-%d %H:%M:%S')
                if time_point_str not in cmmdty_snpshts:
                    zone_point_time = pytz.utc.localize(time_point)
                    nw_price = data_point.Close
                    if last_data_point is None:
                        change = 0
                    else:
                        old_price = last_data_point.Close
                        if old_price and old_price > 0:
                            change = (nw_price-old_price)/old_price
                        else:
                            change = 0
                    new_snapshots.append(CommodityBuffer(
                        commodity = the_commodity,
                        captured_at = zone_point_time,
                        price_snapshot = nw_price,
                    ))
                last_data_point = data_point
            CommodityBuffer.objects.bulk_create(new_snapshots)

            # tst = tst + 1
            # if tst == 4:
            #     break

    def start_raw_input_dataframes(self):
        str_time_step = str(self.the_current_time_step)
        last_time_step = self.last_time_step + relativedelta(days=self.added_days)
        str_last_time_step = str(last_time_step)
        db_params = self.env_config.get("db_params")
        db_conn = psycopg2.connect(**db_params)
        cursor = db_conn.cursor()

        stcks_query = """
            SELECT *
            FROM stocks
        """
        cursor.execute(stcks_query)
        stck_data = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        self.stck_df = pd.DataFrame(stck_data, columns=column_names) 
        # stck_df = stck_df.to_dict(orient='records')

        stcks_buffer_query = f"""
            SELECT stocks_buffers.*, stocks.index
            FROM stocks_buffers 
            JOIN stocks on stocks.id = stocks_buffers.stock_id
            WHERE 
            captured_at >= '{str_time_step}' AND captured_at <= '{str_last_time_step}'
        """
        cursor.execute(stcks_buffer_query)
        stcks_buffer_data = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        self.stcks_buffer_df = pd.DataFrame(stcks_buffer_data, columns=column_names)

        # stcks_buffer_df = stcks_buffer_df.to_dict(orient='records')

        cmmdties_buffer_query = f"""
            SELECT commodities_buffers.*, commodities.index
            FROM commodities_buffers 
            JOIN commodities on commodities.id = commodities_buffers.commodity_id
            WHERE 
            captured_at >= '{str_time_step}' AND captured_at <= '{str_last_time_step}'
        """
        cursor.execute(cmmdties_buffer_query)
        cmmdties_buffer_data = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        self.cmmdties_buffer_df = pd.DataFrame(cmmdties_buffer_data, columns=column_names)
        # cmmdties_buffer_df = cmmdties_buffer_df.to_dict(orient='records')

        cmmdties_query = """
            SELECT *
            FROM commodities
        """
        cursor.execute(cmmdties_query)
        cmmdties_data = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        self.cmmdties_df = pd.DataFrame(cmmdties_data, columns=column_names) 

    def create_working_days(self):
        # Create a list of working days within the date range
        added_end_date = self.last_time_step + relativedelta(days=self.added_days)
        working_days = pd.date_range(start=self.the_current_time_step, end=added_end_date, freq='B')

        # Repeat rows for each index from 1 to 100
        stk_index_range = range(1, 101)

        # Create a DataFrame with the specified structure
        stcks_data = {'captured_at': [], 'change': [], 'index': []}

        for date in working_days:
            for idx in stk_index_range:
                stcks_data['captured_at'].append(date)
                stcks_data['change'].append(0)
                stcks_data['index'].append(idx)

        self.stcks_working_day_df = pd.DataFrame(stcks_data)

        cmmdty_index_range = range(1, 15)
        cmmdts_data = {'captured_at': [], 'index': []}

        for date in working_days:
            for idx in cmmdty_index_range:
                cmmdts_data['captured_at'].append(date)
                cmmdts_data['index'].append(idx)

        self.cmmdts_working_day_df = pd.DataFrame(cmmdts_data)

    def populate_missing_stock_days(self):

        stocks_fields_list = ["id", "price_snapshot", "volume", "bid_vol",	"bid_price", "offer_vol",	"offer_price",	"stock_id",	"close",	"high",	"low",	"open",]

        stcks_df1 = self.stcks_buffer_df
        stcks_df2 = self.stcks_working_day_df

        # Merge dataframes on 'captured_at' and 'index' and keep only the rows present in stcks_df2
        stcks_merged_df = pd.merge(stcks_df2, stcks_df1, on=['captured_at', 'index'], how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)


        stcks_concatenated_df = pd.concat([stcks_df1, stcks_merged_df], ignore_index=True)

        # Print the concatenated dataframe

        # Sort the DataFrame by 'index' and 'captured_at'
        stcks_concatenated_df = stcks_concatenated_df.sort_values(by=['index', 'captured_at'])

        # Forward fill NaN values based on 'index'

        for field in stocks_fields_list:
            stcks_concatenated_df[field] = stcks_concatenated_df.groupby('index')[field].fillna(method='ffill')
            stcks_concatenated_df[field] = stcks_concatenated_df.groupby('index')[field].fillna(method='bfill')


        # Display the upcaptured_atd DataFrame

        stcks_concatenated_df['change'].fillna(stcks_concatenated_df['change_x'], inplace=True)

        # Drop the 'change_x' column if you no longer need it
        stcks_concatenated_df.drop(columns=['change_x', 'change_y'], inplace=True)

        self.stcks_buffer_df = stcks_concatenated_df

    def populate_missing_commodity_days(self):
        commodities_fields_list = ["id", "price_snapshot", "commodity_id"]

        cmmdts_df1 = self.cmmdties_buffer_df
        cmmdts_df2 = self.cmmdts_working_day_df

        # Merge dataframes on 'captured_at' and 'index' and keep only the rows present in cmmdts_df2
        cmmdts_merged_df = pd.merge(cmmdts_df2, cmmdts_df1, on=['captured_at', 'index'], how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
        cmmdts_concatenated_df = pd.concat([cmmdts_df1, cmmdts_merged_df], ignore_index=True)

        # Print the concatenated dataframe
        # Sort the DataFrame by 'index' and 'captured_at'
        cmmdts_concatenated_df = cmmdts_concatenated_df.sort_values(by=['index', 'captured_at'])

        # Forward fill NaN values based on 'index'

        for field in commodities_fields_list:
            cmmdts_concatenated_df[field] = cmmdts_concatenated_df.groupby('index')[field].fillna(method='ffill')
            cmmdts_concatenated_df[field] = cmmdts_concatenated_df.groupby('index')[field].fillna(method='bfill')


        # Display the upcaptured_atd DataFrame

        self.cmmdties_buffer_df = cmmdts_concatenated_df

    def create_input_dataframes(self): 
        self.start_raw_input_dataframes()
        self.create_working_days()
        self.populate_missing_stock_days()
        self.populate_missing_commodity_days()
        self.stck_df.fillna(0, inplace=True)
        self.cmmdties_df.fillna(0, inplace=True)
        self.stcks_buffer_df.fillna(0, inplace=True)
        self.cmmdties_buffer_df.fillna(0, inplace=True)
        return [
            self.stck_df,
            self.cmmdties_df,
            self.stcks_buffer_df,
            self.cmmdties_buffer_df
        ]
    
    @staticmethod
    def get_new_file_path():
        base_path = '/Users/mirbilal/Desktop/MobCommission/commissionV2/apps/environment/joblibs/'
        file_name = 'trading_job_v'

        # Step 1: Determine the latest version number
        existing_versions = [f for f in os.listdir(base_path) if f.startswith(file_name)]
        latest_version = max(map(lambda x: int(x[len(file_name):]), existing_versions), default=0)

        # Step 2: Increment the version number
        new_version = latest_version + 1

        # Step 3: Create the new file with the incremented version number
        current_file_path = f"{base_path}{file_name}{latest_version}"
        new_file_path = f"{base_path}{file_name}{new_version}"
        return (current_file_path, new_file_path)
