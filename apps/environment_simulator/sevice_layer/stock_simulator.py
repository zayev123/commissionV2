import json
from apps.environment_simulator.models import (
    SimulatedStock,
    SimulatedStockCovariance,
    SimulatedStockBuffer,
    SimulatedStockXCommodity,
)
from apps.environment_simulator.models.simulated_commodity import SimulatedCommodityBuffer
from apps.environment_simulator.sevice_layer.commodity_simulator import CommoditySimulator, CommodityVaryData
import numpy as np
from datetime import datetime
from django.db.models import Q
import random
from dateutil.relativedelta import relativedelta



class StockSimulator:
    def __init__(self, latest_time_step: datetime, next_time_step: datetime, is_coupled = False):
        self.eps = 0.3
        self.stocks = SimulatedStock.objects.all()
        self.stocks = list(self.stocks)
        self.no_of_stocks = len(list(self.stocks))
        self.stocks_covariances: list[SimulatedStockCovariance] = SimulatedStockCovariance.objects.all()
        self.stock_x_commodities: list[SimulatedStockXCommodity] = SimulatedStockXCommodity.objects.all()
        self.stocks = random.sample(self.stocks, self.no_of_stocks)
        self.is_coupled = is_coupled

        self.affected_stocks = {}
        self.covariances_data = {}
        self.latest_time_step = latest_time_step
        self.next_time_step = next_time_step

        for a_covar in self.stocks_covariances:
            stock_a = a_covar.stock_a
            stock_b = a_covar.stock_b
            cvr = a_covar.factor

            if stock_a.id not in self.covariances_data:
                self.covariances_data[stock_a.id] = {}
            
            stock_a_covars = self.covariances_data[stock_a.id]
            if stock_b.id not in stock_a_covars:
                stock_a_covars[stock_b.id] = cvr
            
            if stock_b.id not in self.covariances_data:
                self.covariances_data[stock_b.id] = {}

            stock_b_covars = self.covariances_data[stock_b.id]
            if stock_a.id not in stock_b_covars:
                stock_b_covars[stock_a.id] = cvr

        stock_ids = []
        for z_stocks in self.stocks:
            stock_ids.append(z_stocks.id)

        previous_time_snapshot = SimulatedStockBuffer.objects.order_by("-captured_at").filter(Q (captured_at__lt = latest_time_step)).first()
        if previous_time_snapshot:
            previous_time_step = previous_time_snapshot.captured_at
        else:
            previous_time_step = latest_time_step - relativedelta(hours=2, minutes=30)
        
        previous_captures = SimulatedStockBuffer.objects.filter(
            Q(stock_id__in = stock_ids)
            & Q (captured_at = previous_time_step)
        ).all()
        # print(previous_captures, previous_time_step)
        
        latest_captures = SimulatedStockBuffer.objects.filter(
            Q(stock_id__in = stock_ids)
            & Q (captured_at = latest_time_step)
        ).all()

        existing_next_captures = SimulatedStockBuffer.objects.filter(
            Q(stock_id__in = stock_ids)
            & Q (captured_at = next_time_step)
        ).all()

        self.last_buffer = {}
        for pev_capt in previous_captures:
            self.last_buffer[pev_capt.stock_id] = pev_capt

        # print("self.last_buffer ", self.last_buffer )
        
        self.next_buffer = {}
        for next_capt in existing_next_captures:
            self.next_buffer[next_capt.stock_id] = next_capt

        self.mem_buffer = {}
        for capt in latest_captures:
            self.mem_buffer[capt.stock_id] = capt

        self.varied_stck_x_cmmdts_dict = {}
        self.cmmdts_x_stcks_covars = {}

        for xcmst_var in self.stock_x_commodities:
            if xcmst_var.commodity.id not in self.cmmdts_x_stcks_covars:
                self.cmmdts_x_stcks_covars[xcmst_var.commodity.id] = {}
            cmmdt_x_stcks_covars = self.cmmdts_x_stcks_covars[xcmst_var.commodity.id]
            if xcmst_var.stock.id not in cmmdt_x_stcks_covars:
                cmmdt_x_stcks_covars[xcmst_var.stock.id] = xcmst_var.factor
        
        self.market_sentiment = self.observe_volumes()

    def vary_stock_prices(self):
        if self.is_coupled:
            cmmdty_sims = CommoditySimulator(self.latest_time_step, self.next_time_step)
            varied_cmmdts: dict[int, CommodityVaryData] = cmmdty_sims.vary_commodity_prices()
            for cmmdty_id, vary_data in varied_cmmdts.items():
                if cmmdty_id in self.cmmdts_x_stcks_covars:
                    vrd_stcks = self.cmmdts_x_stcks_covars[cmmdty_id]
                    # for each of the stock in the stocks of its x_commodity
                    for v_stck_id in vrd_stcks:
                        if v_stck_id not in self.varied_stck_x_cmmdts_dict:
                            self.varied_stck_x_cmmdts_dict[v_stck_id] = {}
                        stcks_vary_data = self.varied_stck_x_cmmdts_dict[v_stck_id]
                        if cmmdty_id not in stcks_vary_data:
                            stcks_vary_data[cmmdty_id] = vary_data.perc_change * vrd_stcks[v_stck_id]
                            # vary the set price with each of these commodities variations
            # try:
            #     print("comms vary", json.dumps([{k: {
            #         "original_pice": varied_cmmdts[k].original_pice,
            #         "next_price": varied_cmmdts[k].next_price,
            #         "change": varied_cmmdts[k].change,
            #         "perc_change": varied_cmmdts[k].perc_change,
            #     }} for k in varied_cmmdts], indent=3))
            #     print("stocks_vary", json.dumps(self.varied_stck_x_cmmdts_dict, indent=3))
            # except Exception as error:
            #     print(error)



        updated_stcks = []
        grad_stocks: list[SimulatedStock] = []
        for stck in self.stocks:
            gradient = stck.price_gradient
            sd = stck.price_sd
            min_price = stck.min_price
            max_price = stck.max_price
            avg_forward_steps = stck.avg_forward_steps
            avg_backward_steps = stck.avg_backward_steps
            steps_left = stck.price_steps_left
            
            curr_price = self.mem_buffer[stck.id].price_snapshot

            next_grad_price = curr_price + gradient
            next_grad_price = np.random.normal(next_grad_price, sd)

            if next_grad_price <= min_price:
                next_grad_price = min_price
            elif next_grad_price >= max_price:
                next_grad_price = max_price

            price_diff = next_grad_price - curr_price
            perc_diff = price_diff/curr_price
            # print(f"{stck.id}, {stck.name}", curr_price, gradient, next_grad_price, perc_diff)
            if stck.id not in self.affected_stocks:
                self.affected_stocks[stck.id] = {}
            
            covars_data = self.covariances_data[stck.id]
            for stck_id, xvar in covars_data.items():
                if stck_id != stck.id:
                    if stck_id not in self.affected_stocks:
                        self.affected_stocks[stck_id] = {}
                    self.affected_stocks[stck_id][stck.id] = xvar*perc_diff
            
            # print(json.dumps(self.affected_stocks, indent=3))
            # print("")

            steps_left = steps_left - 1
            new_grad = gradient
            if steps_left <= 0:
                neg_gradient = gradient * -1
                new_grad = np.random.choice([gradient, neg_gradient])
                if new_grad <=0:
                    steps_left = avg_backward_steps
                else:
                    steps_left = avg_forward_steps

            stck.price_steps_left = steps_left
            stck.price_gradient = new_grad
            updated_stcks.append(stck)
            grad_stocks.append({
                "obj": stck,
                "next_grad_price": next_grad_price,
                "original_price": curr_price
            })

        # print(json.dumps(self.affected_stocks, indent=3))
        # print(grad_stocks)

        next_snapshots: list[SimulatedStockBuffer] = []
        for a_stock_data in grad_stocks:
            a_stck: SimulatedStock = a_stock_data["obj"]
            next_price = a_stock_data["next_grad_price"]
            set_price = a_stock_data["original_price"]
            # print(set_price, "orgnl1", a_stck, next_price)
            extra_effects = self.affected_stocks[a_stck.id]
            for an_id, price_effect in extra_effects.items():
                next_price = next_price + price_effect*set_price

            # print(set_price, "orgnl2", a_stck, next_price)
            if self.is_coupled:
                if a_stck.id in self.varied_stck_x_cmmdts_dict:
                    cmmdties_vars = self.varied_stck_x_cmmdts_dict[a_stck.id]
                    for cmmdty_x_id in cmmdties_vars:
                        cmm_price_effect = cmmdties_vars[cmmdty_x_id]
                        next_price = next_price + cmm_price_effect*set_price

            new_volume = 1000
            if a_stck.id in self.market_sentiment:
                senti_data = self.market_sentiment[a_stck.id]
                new_volume = senti_data["new_volume"]
                vol_price_effect = senti_data["vol_price_effect"]
                next_price = next_price + vol_price_effect*set_price

            # print(set_price, "orgnl3", a_stck, next_price)

            # print("fnl", a_stck, next_price, new_volume)
            # print("")

            new_change = next_price - set_price
            mem_snpsht = self.mem_buffer[a_stck.id]

            (offer_vol, offer_price, bid_vol, bid_price) = StockSimulator.manage_offer_bids(mem_snpsht, new_change, next_price)

            next_snapshots.append(
                SimulatedStockBuffer(
                    stock = a_stck,
                    captured_at = self.next_time_step,
                    price_snapshot = next_price,
                    change = new_change,
                    volume = new_volume,
                    bid_vol = bid_vol,
                    bid_price = bid_price,
                    offer_vol = offer_vol,
                    offer_price = offer_price
                )
            )
        un_accounted_for_stcks = []
        for unaccntd_stck in updated_stcks:
            if unaccntd_stck.id not in self.next_buffer:
                un_accounted_for_stcks.append(unaccntd_stck)
        
        un_accounted_for_snpshts = []
        for snpsht in next_snapshots:
            if snpsht.stock_id not in self.next_buffer:
                un_accounted_for_snpshts.append(snpsht)

        SimulatedStock.objects.bulk_update(un_accounted_for_stcks, [
            'price_steps_left', 
            'price_gradient', 
        ])
        SimulatedStockBuffer.objects.bulk_create(un_accounted_for_snpshts)

    def observe_volumes(self):
        latest_time_step = self.next_time_step
        ten_time_steps_ago = latest_time_step - relativedelta(hours= 22, minutes=30)
        # print("ten_time_steps_ago", ten_time_steps_ago)
        last_10_snapshots: list[SimulatedStockBuffer] = SimulatedStockBuffer.objects.select_related("stock").filter(
            Q(captured_at__gte=ten_time_steps_ago)
            & Q(captured_at__lte=latest_time_step)
        ).order_by('-captured_at')
        last_10_snapshots = list(reversed(last_10_snapshots))
        # print(last_10_snapshots[0].captured_at, last_10_snapshots[-1].captured_at)
        stcks_variations = {}
        # print(len(last_10_snapshots))
        # print("")
        for a_snpsht in last_10_snapshots:
            if a_snpsht.stock_id not in stcks_variations:
                if not a_snpsht.volume:
                    a_snpsht.volume = 0
                stcks_variations[a_snpsht.stock_id] = {
                    "first_8_consecutives": 0,
                    "latest_2_consecutives": 0,
                    "latest_change": 0,
                    "index": 1,
                    "price_x_volume_factor": a_snpsht.stock.price_x_volume_factor,
                    "expected_volume_perc_change": 0,
                    "volume_x_price_factor": a_snpsht.stock.volume_x_price_factor,
                    "volume_so_far": a_snpsht.volume,
                    "vol_price_effect": 0
                }
            stck_variations: dict = stcks_variations[a_snpsht.stock_id]
            stck_indx = stck_variations["index"]
            if not a_snpsht.change:
                a_snpsht.change = 0
            latest_change = a_snpsht.change/a_snpsht.price_snapshot
            if not latest_change:
                latest_change = 0
            last_change = stck_variations["latest_change"]
            if not last_change:
                last_change = 0

            # if a_snpsht.stock_id == 338:
            #     print(a_snpsht.captured_at)
            #     print("latest_change", a_snpsht.change, latest_change, a_snpsht.price_snapshot)
            #     print("last_change", last_change)

            if ((latest_change >= 0 and last_change <= 0) or (latest_change <= 0 and last_change >= 0) or (latest_change == 0 and last_change == 0)):
                if stck_indx <=8:
                    stck_variations["first_8_consecutives"] = 0
                else:
                    stck_variations["latest_2_consecutives"] = 0
            else:
                if latest_change>0:
                    update = 1
                else:
                    update = -1
                if stck_indx <=8:
                    stck_variations["first_8_consecutives"] = stck_variations["first_8_consecutives"] + update
                else:
                    stck_variations["latest_2_consecutives"] = stck_variations["latest_2_consecutives"] + update
            
            # if a_snpsht.stock_id == 338:
            #     print(a_snpsht.volume)
            #     print(stck_variations)
            #     print("")
            
            stck_variations["index"] = stck_variations["index"] + 1
            stck_variations["latest_change"] = latest_change
            stck_variations["volume_so_far"] = a_snpsht.volume

        # print(a_snpsht.captured_at)
        for stcx_id, vols_data in stcks_variations.items():
            first_8_consecutives = vols_data["first_8_consecutives"]
            latest_2_consecutives = vols_data["latest_2_consecutives"]
            price_x_volume_factor = vols_data["price_x_volume_factor"]
            volume_x_price_factor = vols_data["volume_x_price_factor"]
            perc_price_change = vols_data["latest_change"]
            sd_vol = price_x_volume_factor/10
            sd_price = volume_x_price_factor/10
            set_price_to_vol_effect_perc = np.random.normal(price_x_volume_factor, sd_vol)
            set_vol_to_price_effect_perc = np.random.normal(volume_x_price_factor, sd_price)
            expected_volume_perc_change = set_price_to_vol_effect_perc
            pos_sign = 1
            if first_8_consecutives >= 5 and latest_2_consecutives < 0:
                expected_volume_perc_change = -1*expected_volume_perc_change
                pos_sign = -1
            elif first_8_consecutives <= 5 and latest_2_consecutives > 0:
                pass
            else:
                expected_volume_perc_change = 0
                set_price_to_vol_effect_perc = 0
            vols_data["expected_volume_perc_change"] = expected_volume_perc_change
            last_volume = vols_data["volume_so_far"]
            if last_volume< 10:
                last_volume = 10
            amplifier = 5
            new_volume = last_volume + abs(perc_price_change)*set_price_to_vol_effect_perc*last_volume*amplifier
            vols_data["new_volume"] = np.random.normal(new_volume, new_volume/100)
            new_volume = vols_data["new_volume"]
            
            # if stcx_id == 343:
            #     print(new_volume < last_volume)
            #     print(new_volume, last_volume)
            
            if new_volume < last_volume:
                # if stcx_id == 343:
                #     print("why", last_volume)
                new_volume = last_volume + (last_volume - new_volume)/5
                vols_data["new_volume"] = new_volume
            
            # if stcx_id == 343:
            #     print("start")
            #     print(last_volume)
            #     print(vols_data["new_volume"])
            #     print("")

            volume_perc_change = (new_volume-last_volume)/last_volume
            vol_price_effect = volume_perc_change*set_vol_to_price_effect_perc*pos_sign
            vols_data["vol_price_effect"] = vol_price_effect

        # print(json.dumps(stcks_variations, indent = 3))
        return stcks_variations
    
    @staticmethod
    def manage_offer_bids(stck: SimulatedStockBuffer, new_change, new_price):
        offer_vol = None
        offer_price = None
        bid_vol = None
        bid_price = None
        offer_sd = stck.offer_vol/40
        bid_sd = stck.bid_vol/40

        perc_change = new_change/stck.price_snapshot
        if perc_change > 0.05:
            offer_vol_change = -1*perc_change
            bid_vol_change = 1*perc_change
        elif perc_change < -0.05:
            offer_vol_change = perc_change
            bid_vol_change = -1*perc_change
        else:
            offer_vol = abs(np.random.normal(stck.offer_vol, offer_sd))
            bid_vol = abs(np.random.normal(stck.bid_vol, bid_sd))

        if offer_vol is None and bid_vol is None:
            offer_vol = stck.offer_vol*(1+offer_vol_change)
            bid_vol = stck.bid_vol*(1+bid_vol_change)

            offer_vol = abs(np.random.normal(stck.offer_vol, offer_sd))
            bid_vol = abs(np.random.normal(stck.bid_vol, bid_sd))

        if perc_change >0.05:
            offer_price = new_price + abs(perc_change*abs((np.random.normal(6,0.5))))
            bid_price = new_price + abs(perc_change*abs((np.random.normal(3,0.5))))
        elif perc_change < 0.05:
            offer_price = new_price + abs(perc_change*abs((np.random.normal(3,0.5))))
            bid_price = new_price + abs(perc_change*abs((np.random.normal(1,0.5))))
        
        else:
            offer_price = new_price + abs(perc_change*abs((np.random.normal(4,0.5))))
            bid_price = new_price + abs(perc_change*abs((np.random.normal(2,0.5))))

        return (offer_vol, offer_price, bid_vol, bid_price)




    def reset_sentiments(self):
        vols = self.market_sentiment
        for stock_id, vol_data in vols.items():
            vol = np.random.choice(list(range(30,100,10)))*np.random.choice(list(range(5,70,10)))
            vol_data["new_volume"] = vol
            vols[stock_id] = vol_data


                
