
import pandas as pd
import os

his_data_category_cols = ['Family', 'Region', 'Salesman', 'Client', 'Category', 'Subcategory', 'SKU', 'Description']
his_data_cycle_rel_cols = result = ['Starting Year', 'Starting Period', 'Periods Per Year', 'Periods Per Cycle']


def all_forecast_dimensions():
    return pd.Index([col for col in his_data_category_cols if col != 'Description'])


class ForecastDims:
    def __init__(self, value):
        _dimensions = all_forecast_dimensions()
        if value == 0:
            self.forecast_dims_sel = _dimensions
        else:
            self.forecast_dims_sel = _dimensions[value-1]


