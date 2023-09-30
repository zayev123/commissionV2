from django.db import models
from .simulated_sector import SimulatedSector

class SimulatedStock(models.Model):
    name = models.CharField(max_length=300)
    index = models.IntegerField(blank=True, null=True)
    sector = models.ForeignKey(SimulatedSector, related_name='stocks', on_delete=models.CASCADE)
    price_gradient = models.FloatField(blank=True, null=True)
    price_sd = models.FloatField(blank=True, null=True)
    min_price = models.FloatField(blank=True, null=True)
    max_price = models.FloatField(blank=True, null=True)
    avg_forward_steps = models.IntegerField(blank=True, null=True)
    avg_backward_steps = models.IntegerField(blank=True, null=True)
    price_steps_left = models.IntegerField(blank=True, null=True)
    volume_sd = models.FloatField(blank=True, null=True)
    volume_x_price_factor = models.FloatField(blank=True, null=True)
    price_x_volume_factor = models.FloatField(blank=True, null=True)

    def __str__(self):
        return f"{str(self.id)}, " + str(self.index) + ', ' + str(self.name)
    
    class Meta:
        verbose_name_plural = "       simulated_stocks"
        ordering = ['index']
        db_table = 'simulated_stocks'
        app_label = 'environment_simulator'


class SimulatedStockCovariance(models.Model):
    stock_a = models.ForeignKey(SimulatedStock, related_name='stock_b_covariances', on_delete=models.CASCADE)
    stock_b = models.ForeignKey(SimulatedStock, related_name='stock_a_covariances', on_delete=models.CASCADE)
    factor = models.FloatField(blank=True, null=True)

    def __str__(self):
        return f"{str(self.id)}, " + str(self.stock_a.name) + ', ' + str(self.stock_b.name)

    class Meta:
        verbose_name_plural = "       simulated_stocks_covariances"
        db_table = 'simulated_stocks_covariances'
        app_label = 'environment_simulator'


class SimulatedStockBuffer(models.Model):
    stock = models.ForeignKey(SimulatedStock, related_name='memory_snapshots', on_delete=models.CASCADE)
    captured_at = models.DateTimeField(blank=True, null=True)
    price_snapshot = models.FloatField(blank=True, null=True)
    ldcp = models.FloatField(blank=True, null=True)
    open = models.FloatField(blank=True, null=True)
    high = models.FloatField(blank=True, null=True)
    low = models.FloatField(blank=True, null=True)
    change = models.FloatField(blank=True, null=True)
    volume = models.FloatField(blank=True, null=True)

    def __str__(self):
        return f"{str(self.id)}, " + str(self.stock.name) + ', ' + str(self.price_snapshot)


    class Meta:
        verbose_name_plural = "       simulated_stocks_buffers"
        ordering = ['-captured_at']
        db_table = 'simulated_stocks_buffers'
        app_label = 'environment_simulator'