from django.db import models


class SimulatedCommodity(models.Model):
    name = models.CharField(max_length=300)
    index = models.IntegerField(blank=True, null=True)
    gradient = models.FloatField(blank=True, null=True)
    sd = models.FloatField(blank=True, null=True)
    min_price = models.FloatField(blank=True, null=True)
    max_price = models.FloatField(blank=True, null=True)
    avg_forward_steps = models.IntegerField(blank=True, null=True)
    avg_backward_steps = models.IntegerField(blank=True, null=True)
    steps_left = models.IntegerField(blank=True, null=True)
    units = models.CharField(max_length=30, blank=True, null=True)

    def __str__(self):
        return f"{str(self.id)}, " + str(self.index) + ', ' + str(self.name)
    
    class Meta:
        verbose_name_plural = "       simulated_commodities"
        ordering = ['index']
        db_table = 'simulated_commodities'
        app_label = 'environment_simulator'


class SimulatedCommodityCovariance(models.Model):
    commodity_a = models.ForeignKey(SimulatedCommodity, related_name='commodity_b_covariances', on_delete=models.CASCADE)
    commodity_b = models.ForeignKey(SimulatedCommodity, related_name='commodity_a_covariances', on_delete=models.CASCADE)
    factor = models.FloatField(blank=True, null=True)

    def __str__(self):
        return f"{str(self.id)}, " + str(self.commodity_a.name) + ', ' + str(self.commodity_b.name)

    class Meta:
        verbose_name_plural = "       simulated_commodities_covariances"
        db_table = 'simulated_commodities_covariances'
        app_label = 'environment_simulator'


class SimulatedCommodityBuffer(models.Model):
    commodity = models.ForeignKey(SimulatedCommodity, related_name='memory_snapshots', on_delete=models.CASCADE)
    captured_at = models.DateTimeField(blank=True, null=True)
    price_snapshot = models.FloatField(blank=True, null=True)


    def __str__(self):
        return f"{str(self.id)}, " + str(self.commodity.name) + ', ' + str(self.price_snapshot)
    
    class Meta:
        verbose_name_plural = "       simulated_commodities_buffers"
        ordering = ['-captured_at']
        db_table = 'simulated_commodities_buffers'
        app_label = 'environment_simulator'
