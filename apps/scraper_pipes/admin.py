from django.contrib import admin

from apps.scraper_pipes.models import Sector, Stock, StockBuffer

# Register your models here.

@admin.register(Sector)
class SectorAdmin(admin.ModelAdmin):
    list_display = [
        'index', 
        'name', 
        'id', 
    ]
    search_fields = ['index', 'name']

@admin.register(Stock)
class StockAdmin(admin.ModelAdmin):
    list_display = [
        'index', 
        'id', 
        'name', 
    ]
    search_fields = ['index', 'name']


@admin.register(StockBuffer)
class StockBufferAdmin(admin.ModelAdmin):
    list_display = [
        'captured_at',
        'stock', 
        'id', 
        'price_snapshot',
        'change',
        'volume',
        'bid_vol',
        'bid_price',
        'offer_vol',
        'offer_price',
    ]
    search_fields = ['stock__name', 'captured_at']
