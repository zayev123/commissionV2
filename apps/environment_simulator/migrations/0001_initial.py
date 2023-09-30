# Generated by Django 4.2.5 on 2023-09-29 13:00

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='SimulatedCommodity',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=300)),
                ('index', models.IntegerField(blank=True, null=True)),
                ('price', models.FloatField(blank=True, null=True)),
                ('gradient', models.FloatField(blank=True, null=True)),
                ('sd', models.FloatField(blank=True, null=True)),
                ('min_price', models.FloatField(blank=True, null=True)),
                ('max_price', models.FloatField(blank=True, null=True)),
                ('steps_to_reversal', models.IntegerField(blank=True, null=True)),
                ('steps_left', models.IntegerField(blank=True, null=True)),
                ('tensor_weight', models.FloatField(blank=True, default=0.0, null=True)),
            ],
            options={
                'verbose_name_plural': '       simulated_commodities',
                'db_table': 'simulated_commodities',
                'ordering': ['index'],
            },
        ),
    ]
