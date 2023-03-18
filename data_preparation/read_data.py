import pandas as pd
import os
import public_variables.refreshes
import generic_columns_name_mo.generic_columns_name


def read_historical_data(inputs_folder_name):
    _ref = public_variables.refreshes.refresh_hist_inputs()

    _excel_con = pd.ExcelFile(os.path.join(inputs_folder_name, 'Historical Data.xlsx'))
    _dtypes = {c: object for c in generic_columns_name_mo.generic_columns_name.all_forecast_dimensions().tolist()}

    _read = pd.read_excel(_excel_con, sheet_name='Historical Data 2',
                          dtype=_dtypes, converters={'Description': str.strip})

    # _run_new_indexes_name = change_dim_indexes_name

    _df = _read.rename(columns=generic_columns_name_mo.generic_columns_name.new_name_cols_cat_dict())

    return _df


print(read_historical_data("inputs_folder_name"))
