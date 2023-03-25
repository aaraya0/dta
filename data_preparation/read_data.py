import pandas as pd
import os
import public_variables.refreshes
import generic_columns_name_mo.generic_columns_name as gen_cols
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from pandas.api.types import is_numeric_dtype


class CycleCols:
    def __init__(self, inputs_folder_name):
        hist_data = base_historical_data(inputs_folder_name)
        self.starting_year = int(hist_data.iloc[0]['Starting Year'])
        self.starting_period = int(hist_data.iloc[0]['Starting Period'])
        self.periods_per_year = int(hist_data.iloc[0]['Periods Per Year'])
        self.periods_per_cycle = int(hist_data.iloc[0]['Periods Per Cycle'])

        _cols = [col for col in hist_data if col != 'None']
        _df = hist_data[_cols]
        self.n_historical_periods = len(_df.iloc[:, 12:].columns)

        if self.periods_per_year is None:
            self.periodicity = None
        elif self.periods_per_year <= 12:
            self.periodicity = 'Month'
        elif self.periods_per_year <= 53:
            self.periodicity = 'Week'
        else:
            self.periodicity = 'Day'

        if self.periodicity == 'Month':
            self.initial_date = datetime.date(year=self.starting_year, month=self.starting_period, day=1)
        elif self.periodicity == 'Week':
            _year = self.starting_year
            _week = self.starting_period
            self.initial_date = datetime.datetime.strptime(f'{_year} {_week} 4', '%G %V %u').date()  # Thursday
        else:
            self.initial_date = datetime.date(year=self.starting_year, month=1, day=1) + \
                                datetime.timedelta(days=self.starting_period-1)

        _n = self.n_historical_periods - 1
        if self.periodicity == 'Month':
            self.final_date = self.initial_date + relativedelta(months=_n)
        elif self.periodicity == 'Week':
            self.finale_date = self.initial_date + pd.Timedelta(days=_n * 7)
        else:
            self.final_date = self.initial_date + pd.Timedelta(days=_n)

        if self.periodicity == 'Month':
            self.hist_dates_index = pd.date_range(start=self.initial_date, end=self.final_date, freq="MS")
        else:
            self.hist_dates_index = pd.date_range(start=self.initial_date, end=self.final_date,
                                                  periods=self.n_historical_periods)


def read_historical_data(inputs_folder_name):
    _excel_con = pd.ExcelFile(os.path.join(inputs_folder_name, 'Historical Data.xlsx'))
    his_data_category_cols = gen_cols.his_data_categ_cols
    _dtypes = {c: str for c in his_data_category_cols}

    _df = pd.read_excel(_excel_con, sheet_name='Historical Data 2', dtype=_dtypes,
                        converters={'Description': str.strip})

    for col in his_data_category_cols:
        _df[col] = _df[col].str.strip()

    # _run_new_indexes_name = change_dim_indexes_name

    _df = _df.rename(columns=gen_cols.new_name_cols_cat_dict())
    return _df


# print(read_historical_data("inputs_folder_name"))


def read_launches_by_similarity_data(inputs_folder_name):
    _ref = public_variables.refreshes.refresh_hist_inputs()

    _excel_con = pd.ExcelFile(os.path.join(inputs_folder_name, 'Launches and Discontinued Data.xlsx'))
    _dtypes = {c: str for c in gen_cols.his_data_categ_cols}
    _df = pd.read_excel(_excel_con, sheet_name='Launches by Similarity', dtype=_dtypes)
    for col in gen_cols.his_data_categ_cols:
        _df[col] = _df[col].str.strip()
    # _run_new_indexes_name = change_dim_indexes_name
    _df = _df.rename(columns=gen_cols.new_name_cols_cat_dict())
    return _df


# print(read_launches_by_similarity_data("inputs_folder_name"))


def read_absolute_launches_data(inputs_folder_name):
    _ref = public_variables.refreshes.refresh_hist_inputs()

    _excel_con = pd.ExcelFile(os.path.join(inputs_folder_name, 'Launches and Discontinued Data.xlsx'))
    _dtypes = {c: str for c in gen_cols.his_data_categ_cols}
    _df = pd.read_excel(_excel_con, sheet_name='Absolute Launches', dtype=_dtypes)
    for col in gen_cols.his_data_categ_cols:
        _df[col] = _df[col].str.strip()
    # _run_new_indexes_name = change_dim_indexes_name
    _df = _df.rename(columns=gen_cols.new_name_cols_cat_dict())
    return _df


# print(read_absolute_launches_data("inputs_folder_name"))


def read_sku_allocation_matrix(inputs_folder_name):
    _ref = public_variables.refreshes.refresh_hist_inputs()

    _excel_con = pd.ExcelFile(os.path.join(inputs_folder_name, 'Allocation Matrix.xlsx'))
    _dtypes = {c: str for c in ['SKU from', 'SKU to']}
    _df = pd.read_excel(_excel_con, sheet_name='Allocation Matrix', dtype=_dtypes)
    for col in ['SKU from', 'SKU to']:
        _df[col] = _df[col].str.strip()
    return _df


# print(read_sku_allocation_matrix("inputs_folder_name"))


def sku_allocation_matrix(inputs_folder_name):
    allocation_matrix = read_sku_allocation_matrix(inputs_folder_name)
    if allocation_matrix.empty:
        return pd.DataFrame()

    sku_col = gen_cols.generic_sku
    desc_col = gen_cols.generic_description

    historical_data = read_historical_data(inputs_folder_name)
    sku_to_desc_map = historical_data[[sku_col, desc_col]].drop_duplicates()

    matrix_with_desc = (
        allocation_matrix.merge(sku_to_desc_map, how="left", left_on="SKU to", right_on=sku_col)
        .merge(sku_to_desc_map, how="left", left_on="SKU from", right_on=sku_col, suffixes=[" to", " from"])
        .rename(columns={desc_col + " to": "Description to", desc_col + " from": "Description from"})
        .loc[:, ["SKU from", "SKU to", "percentage", "Description from", "Description to"]]
    )

    return matrix_with_desc


# print(sku_allocation_matrix("inputs_folder_name"))


def base_historical_data(inputs_folder_name):
    allocation_matrix = sku_allocation_matrix(inputs_folder_name)
    if not allocation_matrix.empty:
        sku_col = gen_cols.generic_sku
        description_col = gen_cols.generic_description
        from_col, to_col, perc_col, desc_from_col, desc_to_col = allocation_matrix.columns
        hist_data = read_historical_data(inputs_folder_name).merge(allocation_matrix,
                                                                   left_on=sku_col, right_on=from_col, how='left')
        hist_data[sku_col] = np.where(hist_data[to_col].isnull(), hist_data[sku_col], hist_data[to_col])
        hist_data[description_col] = np.where(hist_data[to_col].isnull(),
                                              hist_data[description_col],
                                              np.where(hist_data[desc_to_col].isnull(),
                                                       hist_data[desc_from_col], hist_data[desc_to_col]))
        hist_data = hist_data.drop([desc_from_col, desc_to_col], axis=1)
        hist_data[perc_col] = hist_data[perc_col].fillna(1)
        hist_data = hist_data.drop([from_col, to_col], axis=1)
        dimensions = [col for col in hist_data.columns if col not in
                      gen_cols.his_data_category_cols_new_name() +
                      gen_cols.his_data_cycle_rel_cols]
        hist_data[dimensions] = hist_data[dimensions].multiply(hist_data[perc_col], axis="index")
        hist_data = hist_data.drop([perc_col], axis=1)
    else:
        hist_data = read_historical_data(inputs_folder_name)

    """hist_data = hist_data[~hist_data.drop(gen_cols.his_data_category_cols_new_name()
                                          + gen_cols.his_data_cycle_rel_cols, axis=1).isin([0., 0, 'nan']).all(1)]"""

    return hist_data


# print(base_historical_data("inputs_folder_name"))


def historical_launches_by_similarity_data(inputs_folder_name):
    hist_data = base_historical_data(inputs_folder_name)
    rel_cols = gen_cols.his_data_cycle_rel_cols
    description_col = gen_cols.generic_description
    sim_data = read_launches_by_similarity_data(inputs_folder_name)
    if len(sim_data) > 0:
        similarity_launches = sim_data.iloc[:, :8].copy()
        historical_data = read_historical_data(inputs_folder_name)
        df = similarity_launches.merge(historical_data[rel_cols], on=rel_cols, how='left')
        df[description_col] = similarity_launches[description_col]
        df.fillna(0, inplace=True)
        cycle_cols = CycleCols(inputs_folder_name)
        df['Starting Year'] = cycle_cols.starting_year
        df['Starting Period'] = cycle_cols.starting_period
        df['Periods Per Year'] = cycle_cols.periods_per_year if len(hist_data) > 0 else None
        df['Periods Per Cycle'] = cycle_cols.periods_per_cycle
    else:
        df = pd.DataFrame()
    return df


# print(historical_launches_by_similarity_data("inputs_folder_name"))


def historical_absolute_launches_data(inputs_folder_name):
    _abs_launches = read_absolute_launches_data(inputs_folder_name)
    hist_data = base_historical_data(inputs_folder_name)
    rel_cols = gen_cols.his_data_cycle_rel_cols
    description_col = gen_cols.generic_description
    if len(_abs_launches) > 0:
        _historical_data = read_historical_data(inputs_folder_name)
        _absolute_launches = _abs_launches
        _cols_to_keep = _absolute_launches.columns.tolist()[0:8]
        _absolute_launches = _absolute_launches[_cols_to_keep]
        df = pd.DataFrame(columns=_historical_data.columns)
        for col in _cols_to_keep:
            df[col] = _absolute_launches[col]

        for col in rel_cols:
            df[col] = _historical_data[col]

        _date_cols = df.columns[~ df.columns.isin((_cols_to_keep + [description_col] + rel_cols))]

        for col in _date_cols:
            df[col] = df[col].fillna(0.)

        if len(hist_data) > 0:
            cycle_cols = CycleCols(inputs_folder_name)
            df['Starting Year'] = cycle_cols.starting_year
            df['Starting Period'] = cycle_cols.starting_period
            df['Periods Per Year'] = cycle_cols.periods_per_year
            df['Periods Per Cycle'] = cycle_cols.periods_per_cycle
        else:
            df['Starting Year'] = None
            df['Starting Period'] = None
            df['Periods Per Year'] = None
            df['Periods Per Cycle'] = None

    else:
        df = pd.DataFrame()

    return df


# print(historical_absolute_launches_data("inputs_folder_name"))


def historic_data(inputs_folder_name):
    _df = base_historical_data(inputs_folder_name)

    _cat_cols_new_name = gen_cols.his_data_category_cols_new_name()
    # Drops null rows
    _non_date_cols = _cat_cols_new_name + gen_cols.his_data_cycle_rel_cols
    _date_cols = [col for col in _df if col not in _non_date_cols]
    _df_dates = _df[_date_cols]
    indexes_to_keep = _df_dates.loc[~(_df_dates == 0).all(axis=1)].index
    _df = _df.loc[indexes_to_keep]

    # Sort dataframe
    _df = _df.sort_values(by=_cat_cols_new_name, ascending=True)

    _absolute_launches = historical_absolute_launches_data(inputs_folder_name)
    _launches_by_similarity = historical_launches_by_similarity_data(inputs_folder_name)

    _df = pd.concat([_df, _absolute_launches, _launches_by_similarity])

    # Remove None columns
    _cols = [col for col in _df if col != 'None']
    _df = _df[_cols]

    # Replace date columns
    _old_columns = _df.columns.tolist()
    _hist_dates_index = CycleCols(inputs_folder_name)
    _hist_dates_index = _hist_dates_index.hist_dates_index
    _new_columns = _old_columns[:len(_old_columns)-len(_hist_dates_index)] + _hist_dates_index.tolist()
    _df.columns = _new_columns

    # Remove empty rows
    _df = _df.dropna(how='all', subset=_cat_cols_new_name)

    # Prevent error if empty description
    generic_description = gen_cols.generic_description
    if pd.isnull(_df[generic_description]).all():
        _df[generic_description] = _df[generic_description].fillna('None')

    for col in _cat_cols_new_name:
        if col != generic_description:
            _df[col] = _df[col].fillna('None')

        # Convert category columns to string
        if is_numeric_dtype(_df[col]):
            _df[col] = _df[col].astype(int).astype(str)
        else:
            _df[col] = _df[col].astype(str)
        _df[col] = _df[col].str.strip()

    # Convert integer columns
    for col in gen_cols.his_data_cycle_rel_cols:
        if is_numeric_dtype(_df[col]):
            _df[col] = _df[col].astype(int)

    # Melt date columns
    _non_date_cols = _cat_cols_new_name + gen_cols.his_data_cycle_rel_cols
    _date_cols = [col for col in _df if col not in _non_date_cols]
    _df = _df.melt(id_vars=_non_date_cols, value_vars=_date_cols, var_name='Date', value_name='Value')

    # Transform date columns
    _df['Date'] = pd.to_datetime(_df['Date'], format='%Y-%m-%d', errors='raise')

    # Remove dimensions with only None values
    forecast_dims = gen_cols.ForecastDims(0)
    for col in forecast_dims.forecast_dims_sel:
        if col != gen_cols.generic_family:
            _dim_values = _df[col].unique().tolist()
            if len(_dim_values) == 1 and _dim_values[0] == 'None':
                _df = _df.drop(columns=col)

    # Unify generic SKU with Description
    generic_sku = gen_cols.generic_sku
    if generic_sku in _df:
        _df[generic_sku] = _df[generic_sku] + np.where((_df[generic_description] == 'None') |
                                                       (_df[generic_description] == ''), '', '__' +
                                                       _df[generic_description])
    _df = _df.drop(columns=generic_description).reset_index(drop=True)

    return _df


# print(historic_data("inputs_folder_name"))

