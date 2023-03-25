import generic_columns_name_mo.generic_columns_name as gen_cols



class RunMode:
    def __init__(self, selection):
        if selection == 0:
            self.value = 'Bottom-Up'
        elif selection == 1:
            self.value = 'Top-Down'
        elif selection == 2:
            self.value = 'Both'


class TopDownDims:
    def __init__(self, selection):
        _options = gen_cols.all_forecast_dimensions()
        self.value = _options[selection]


class ForecastDims:
    def __init__(self, selection):
        _dimensions = gen_cols.all_forecast_dimensions()
        if selection == 0:
            self.value = _dimensions
        else:
            self.value = [_dimensions[selection-1]]


class ReplaceOutliers:
    def __init__(self, selection):
        if selection == 0:
            self.value = 'Yes'
        elif selection == 1:
            self.value = 'No'


class InterpolateNegatives:
    def __init__(self, selection):
        if selection == 0:
            self.value = 'Yes'
        elif selection == 1:
            self.value = 'No'


class InterpolateZeros:
    def __init__(self, selection):
        if selection == 0:
            self.value = 'Yes'
        elif selection == 1:
            self.value = 'No'


class InterpolationMethods:
    def __init__(self, selection):
        interpolation_methods = ['linear', 'nearest', 'replace with zero', 'slinear', 'quadratic', 'cubic', 'spline',
                                 'piecewise_polynomial', 'from_derivatives', 'pchip', 'akima']
        self.value = interpolation_methods[selection]


class OutliersDetection:
    def __init__(self, selection):
        if selection == 0:
            self.value = 'Mean - Std'
        elif selection == 1:
            self.value = 'Interquartile Range'


class MissingValuesT:
    def __init__(self, selection):
        if selection == 0:
            self.value = 'Interpolate with Selected Method'
        elif selection == 1:
            self.value = 'Replace with Zero'


class PExVariables:
    def __init__(self, selection):
        if selection == 0:
            self.value = 'Form'
        elif selection == 1:
            self.value = 'Excel Spreadsheet'


class Mapes100:
    def __init__(self, selection):
        if selection == 0:
            self.value = 'Yes'
        elif selection == 1:
            self.value = 'No'

