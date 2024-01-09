from django.db import models

class Crypto(models.Model):
    name = models.CharField(max_length=300)
    index = models.IntegerField(blank=True, null=True)
    type = models.IntegerField(blank=True, null=True)
    symbol = models.CharField(max_length=300, blank=True, null=True)
    
    def __str__(self):
        return f"{str(self.id)}, " + str(self.index) + ', ' + str(self.name)
    
    class Meta:
        verbose_name_plural = "       Cryptos"
        ordering = ['index']
        db_table = 'cryptos'
        app_label = 'environment'


class CryptoBuffer(models.Model):
    crypto = models.ForeignKey(Crypto, related_name='memory_snapshots', on_delete=models.CASCADE)
    captured_at = models.DateTimeField(blank=True, null=True)
    price_snapshot = models.FloatField(blank=True, null=True)
    change = models.FloatField(blank=True, null=True)
    volume = models.FloatField(blank=True, null=True)
    bid_vol = models.FloatField(blank=True, null=True)
    bid_price = models.FloatField(blank=True, null=True)
    offer_vol = models.FloatField(blank=True, null=True)
    offer_price = models.FloatField(blank=True, null=True)
    open = models.FloatField(blank=True, null=True)
    close = models.FloatField(blank=True, null=True)
    high = models.FloatField(blank=True, null=True)
    low = models.FloatField(blank=True, null=True)



    def __str__(self):
        return f"{str(self.id)}, " + str(self.crypto.name) + ', ' + str(self.price_snapshot)


    class Meta:
        verbose_name_plural = "       cryptos_buffers"
        ordering = ['-captured_at']
        db_table = 'cryptos_buffers'
        app_label = 'environment'