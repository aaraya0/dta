
import pandas as pd
import os

his_data_categ_cols = ['Family', 'Region', 'Salesman', 'Client', 'Category', 'Subcategory', 'SKU', 'Description']
his_data_cycle_rel_cols = result = ['Starting Year', 'Starting Period', 'Periods Per Year', 'Periods Per Cycle']


def read_allocation_generic_name(inputs_folder_name):
    _excel_con = pd.ExcelFile(os.path.join(inputs_folder_name, 'Allocation Matrix.xlsx'))
    _df = pd.read_excel(_excel_con, sheet_name='Dimensions Name')
    _df["New Name"] = _df["New Name"].fillna(_df["Generic Name"])
    return _df
# inputs_folder_name = "path/to/folder"
# result = read_allocation_generic_name(inputs_folder_name)


def his_data_category_cols_new_name():
    his_data_categ_cols_new_name = []
    length = len(his_data_categ_cols)
    for x in range(length):
        _df = read_allocation_generic_name("inputs_folder_name")
        _filter = his_data_categ_cols[x]
        _new_name = _df[_df["Generic Name"] == _filter]["New Name"].iloc[0]
        his_data_categ_cols_new_name.append(_new_name)
    return his_data_categ_cols_new_name


def new_name_cols_cat_dict():
    _keys_list = his_data_categ_cols
    _values_list = his_data_category_cols_new_name()
    _zip_iterator = zip(_keys_list, _values_list)
    _dict = dict(_zip_iterator)
    return _dict


def all_forecast_dimensions():
    return pd.Index([col for col in his_data_category_cols_new_name() if col != 'Description'])
