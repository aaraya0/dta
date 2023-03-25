import generic_columns_name_mo.generic_columns_name as gen_cols#
import read_data
import selectors as sel
import pandas as pd
import os

generic_family = gen_cols.generic_family
generic_region = gen_cols.generic_region
generic_category = gen_cols.generic_category
generic_client = gen_cols.generic_client
generic_subcategory = gen_cols.generic_subcategory
generic_salesman = gen_cols.generic_salesman
comb_f_topdown_forecasts = [
    [generic_family, generic_region, generic_salesman, generic_client, generic_category, generic_subcategory],
    [generic_family, generic_region, generic_client, generic_category, generic_subcategory],
    [generic_family, generic_region, generic_salesman, generic_category, generic_subcategory],
    [generic_family, generic_region, generic_category, generic_subcategory],
    [generic_family, generic_category, generic_subcategory],
    [generic_family, generic_region, generic_salesman, generic_client, generic_category],
    [generic_family, generic_region, generic_client, generic_category],
    [generic_family, generic_region, generic_salesman, generic_category],
    [generic_family, generic_region, generic_category],
    [generic_family, generic_category],
    [generic_family, generic_region, generic_client],
    [generic_family, generic_region],
    [generic_family],
    [generic_region, generic_salesman, generic_client, generic_category, generic_subcategory],
    [generic_region, generic_client, generic_category, generic_subcategory],
    [generic_region, generic_salesman, generic_category, generic_subcategory],
    [generic_region, generic_category, generic_subcategory],
    [generic_region, generic_salesman, generic_client, generic_category],
    [generic_region, generic_client, generic_category],
    [generic_region, generic_salesman, generic_category],
    [generic_region, generic_category],
    [generic_region],
    [generic_salesman, generic_client, generic_category, generic_subcategory],
    [generic_salesman, generic_client, generic_category],
    [generic_salesman, generic_category, generic_subcategory],
    [generic_salesman, generic_subcategory],
    [generic_salesman, generic_client],
    [generic_salesman],
    [generic_client, generic_category, generic_subcategory],
    [generic_client, generic_category],
    [generic_client, generic_subcategory],
    [generic_client],
    [generic_category, generic_subcategory],
    [generic_category],
    [generic_subcategory],
]


def sel_comb_f_topdown_forecasts(inputs_folder_name, run_mode, dim_top_down, fc_dims):
    historical_data = read_data.historic_data(inputs_folder_name, fc_dims)
    run_mode_sel = sel.RunMode(run_mode).value
    dimensions_for_top_down_sel = sel.TopDownDims(dim_top_down).value
    forecasted_dimensions = sel.ForecastDims(fc_dims).value
    sel_forecasted_dimensions = pd.Index([generic_family] + forecasted_dimensions
                                         if generic_family not in forecasted_dimensions else forecasted_dimensions)
    rem_sel_forecasted_dimensions = pd.Index([col for col in sel_forecasted_dimensions if col in historical_data])
    if run_mode_sel == 'Top-Down':
        result = [[dimensions_for_top_down_sel]]
    else:
        # Remove combinations with dimensions that have only one value (they do not add information)
        _unneeded_dims = [d for d in rem_sel_forecasted_dimensions if
                          historical_data[d].nunique() == 1 and d != generic_family]
        _dims = [d for d in rem_sel_forecasted_dimensions if d not in _unneeded_dims]

        _combs = []
        for c in comb_f_topdown_forecasts:
            _flag = True
            for d in c:
                if d not in _dims:
                    _flag = False
                    break
            if _flag:
                _combs.append(c)

        result = [d for d in _combs if d != rem_sel_forecasted_dimensions.tolist()]

    return result


print(sel_comb_f_topdown_forecasts("inputs_folder_name", 0, 2, 0))

