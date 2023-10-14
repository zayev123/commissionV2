from django.db import models

class Sector(models.Model):
    name = models.CharField(max_length=300)
    index = models.IntegerField(blank=True, null=True)

    class Meta:
        verbose_name_plural = "       sectors"
        ordering = ['index']
        db_table = 'sectors'
        app_label = 'scraper_pipes'