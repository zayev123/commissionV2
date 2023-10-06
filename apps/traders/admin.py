from django.contrib import admin
from django.contrib.auth.models import Group
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.forms import UserChangeForm, UserCreationForm
from .models import Trader, SimulatedOwnership
from django.contrib.gis.db.models import PointField
from mapwidgets.widgets import GooglePointFieldInlineWidget

# Register your models here.
class TraderAdmin(BaseUserAdmin):
    # The forms to add and change user instances
    form = UserChangeForm
    add_form = UserCreationForm

    # The fields to be used in displaying the User model.
    # These override the definitions on the base UserAdmin
    # that reference specific fields on auth.User.
    list_display = ('email', 'user_name', 'is_admin', "simulated_portfolio_value")
    list_filter = ('is_admin',)
    fieldsets = (
        (None, {'fields': ('phone_number', 'user_name', 'user_string', 'email', 'password')}),
        ('Personal info', {'fields': (
            'name', 
            'date_of_birth', 
            'gender', 
            'image', 
            'location',
        )}),
        ('Permissions', {'fields': ('is_admin',)}),
    )
    # add_fieldsets is not a standard ModelAdmin attribute. UserAdmin
    # overrides get_fieldsets to use this attribute when creating a user.
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('date_of_birth', 'phone_number', 'gender', 'name', 'user_name', 'email', 'password1', 'password2'),
        }),
    )
    search_fields = ('email', 'user_name')
    ordering = ('email',)
    filter_horizontal = ()
    list_display = ['id', 'email', 'user_name', "simulated_portfolio_value"]
    formfield_overrides = {
        PointField: {"widget": GooglePointFieldInlineWidget}
    }


# Now register the new UserAdmin...
admin.site.register(Trader, TraderAdmin)

# ... and, since we're not using Django's built-in permissions,
# unregister the Group model from admin.
admin.site.unregister(Group)

@admin.register(SimulatedOwnership)
class SimulatedCommodityAdmin(admin.ModelAdmin):
    list_display = [
        'stock',
        'trader', 
        'shares', 
    ]
    search_fields = ['stock__name', 'trader__name', 'shares']
