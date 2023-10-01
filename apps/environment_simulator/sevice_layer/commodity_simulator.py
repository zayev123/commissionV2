import json
from apps.environment_simulator.models import (
    SimulatedCommodity, 
    SimulatedCommodityCovariance,
    SimulatedCommodityBuffer
)
import numpy as np
from datetime import datetime
from django.db.models import Q
import random

class CommoditySimulator:
    def __init__(self, latest_time_step: datetime, next_time_step: datetime):
        self.eps = 0.3
        self.commodities = SimulatedCommodity.objects.all()
        self.commodities = list(self.commodities)
        self.no_of_commodities = len(list(self.commodities))
        self.commodities_covariances: list[SimulatedCommodityCovariance] = SimulatedCommodityCovariance.objects.all()
        self.commodities = random.sample(self.commodities, self.no_of_commodities)

        self.affected_commodities = {}
        self.covariances_data = {}
        self.next_time_step = next_time_step

        for a_covar in self.commodities_covariances:
            cmmdty_a = a_covar.commodity_a
            cmmdty_b = a_covar.commodity_b
            cvr = a_covar.factor

            if cmmdty_a.id not in self.covariances_data:
                self.covariances_data[cmmdty_a.id] = {}
            
            cmmdty_a_covars = self.covariances_data[cmmdty_a.id]
            if cmmdty_b.id not in cmmdty_a_covars:
                cmmdty_a_covars[cmmdty_b.id] = cvr
            
            if cmmdty_b.id not in self.covariances_data:
                self.covariances_data[cmmdty_b.id] = {}

            cmmdty_b_covars = self.covariances_data[cmmdty_b.id]
            if cmmdty_a.id not in cmmdty_b_covars:
                cmmdty_b_covars[cmmdty_a.id] = cvr

        cmmdty_ids = []
        for z_cmmdty in self.commodities:
            cmmdty_ids.append(z_cmmdty.id)
        
        latest_captures = SimulatedCommodityBuffer.objects.filter(
            Q(commodity_id__in = cmmdty_ids)
            & Q (captured_at = latest_time_step)
        ).all()

        existing_next_captures = SimulatedCommodityBuffer.objects.filter(
            Q(commodity_id__in = cmmdty_ids)
            & Q (captured_at = next_time_step)
        ).all()

        self.next_buffer = {}
        for next_capt in existing_next_captures:
            self.next_buffer[next_capt.commodity_id] = next_capt

        self.mem_buffer = {}
        for capt in latest_captures:
            self.mem_buffer[capt.commodity_id] = capt


    def vary_commodity_prices(self):
        updated_cmmdties = []
        grad_commodities: list[SimulatedCommodity] = []
        for cmmdty in self.commodities:
            gradient = cmmdty.gradient
            sd = cmmdty.sd
            min_price = cmmdty.min_price
            max_price = cmmdty.max_price
            avg_forward_steps = cmmdty.avg_forward_steps
            avg_backward_steps = cmmdty.avg_backward_steps
            steps_left = cmmdty.steps_left
            
            curr_price = self.mem_buffer[cmmdty.id].price_snapshot

            next_grad_price = curr_price + gradient
            next_grad_price = np.random.normal(next_grad_price, sd)

            if next_grad_price <= min_price:
                next_grad_price = min_price
            elif next_grad_price >= max_price:
                next_grad_price = max_price

            price_diff = next_grad_price - curr_price
            perc_diff = price_diff/curr_price
            # print(f"{cmmdty.id}, {cmmdty.name}", curr_price, gradient, next_grad_price, perc_diff)
            if cmmdty.id not in self.affected_commodities:
                self.affected_commodities[cmmdty.id] = {}
            
            covars_data = self.covariances_data[cmmdty.id]
            for cmdty_id, xvar in covars_data.items():
                if cmdty_id != cmmdty.id:
                    if cmdty_id not in self.affected_commodities:
                        self.affected_commodities[cmdty_id] = {}
                    self.affected_commodities[cmdty_id][cmmdty.id] = xvar*perc_diff
            
            # print(json.dumps(self.affected_commodities, indent=3))
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

            cmmdty.steps_left = steps_left
            cmmdty.gradient = new_grad
            updated_cmmdties.append(cmmdty)
            grad_commodities.append({
                "obj": cmmdty,
                "next_grad_price": next_grad_price,
                "original_price": curr_price
            })

        # print(json.dumps(self.affected_commodities, indent=3))
        # print(grad_commodities)
        next_snapshots: list[SimulatedCommodityBuffer] = []
        for a_cmmdty_data in grad_commodities:
            a_cmmdty: SimulatedCommodity = a_cmmdty_data["obj"]
            next_price = a_cmmdty_data["next_grad_price"]
            set_price = a_cmmdty_data["original_price"]
            extra_effects = self.affected_commodities[a_cmmdty.id]
            for an_id, price_effect in extra_effects.items():
                next_price = next_price + price_effect*set_price

            next_snapshots.append(
                SimulatedCommodityBuffer(
                    commodity = a_cmmdty,
                    captured_at = self.next_time_step,
                    price_snapshot = next_price
                )
            )
        un_accounted_for_cmmdties = []
        for unaccntd_cmmdty in updated_cmmdties:
            if unaccntd_cmmdty.id not in self.next_buffer:
                un_accounted_for_cmmdties.append(unaccntd_cmmdty)
        
        un_accounted_for_snpshts = []
        for snpsht in next_snapshots:
            if snpsht.commodity_id not in self.next_buffer:
                un_accounted_for_snpshts.append(snpsht)

        SimulatedCommodity.objects.bulk_update(un_accounted_for_cmmdties, [
            'steps_left', 
            'gradient', 
        ])
        SimulatedCommodityBuffer.objects.bulk_create(un_accounted_for_snpshts)
                

