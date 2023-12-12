from django.contrib import admin
from apps.environment.models import (
    Commodity, 
    CommodityBuffer,
    Stock,
    StockBuffer,
)

# Register your models here.
@admin.register(Commodity)
class CommodityAdmin(admin.ModelAdmin):
    list_display = [
        'index',
        'id', 
        'name',
        'symbol',
    ]
    search_fields = ['index', 'name', 'avg_forward_steps']


@admin.register(CommodityBuffer)
class CommodityBufferAdmin(admin.ModelAdmin):
    list_display = [
        'captured_at',
        'commodity', 
        'price_snapshot', 
        'id', 
    ]
    search_fields = ['commodity__name', 'captured_at']


@admin.register(Stock)
class StockAdmin(admin.ModelAdmin):
    list_display = [
        'index', 
        'id', 
        "symbol",
        'name', 
        "type",
    ]
    search_fields = ['index', 'name', "type"]


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
