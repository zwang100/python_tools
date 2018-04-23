import pandas as pd
import numpy as np
import xlsxwriter
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from IPython.display import display, HTML

pd.options.display.float_format = '{:,.2f}'.format


def eda_report(df, exclude_vars=[], quantiles=[5, 95], save_path='', show_most_value=False):
    '''

    :param df:
    :param exclude_vars:
    :param quantiles:
    :param save_path:
    :param show_most_value:
    :return:
    '''

    def get_cat_features(df):
        return list(df.select_dtypes(include=['object']).columns)

    def get_num_features(df, exlude_list=exclude_vars):
        return [cont for cont in list(df.select_dtypes(include=['float64', 'int64']).columns) if
                cont not in exlude_list]

    # create categorical dataframe
    cat_attr = get_cat_features(df)
    cat_df = df[cat_attr]

    # create numerical dataframe
    num_attr = get_num_features(df)
    num_df = df[num_attr]

    # initialize the flag values
    num_flag = 0
    cat_flag = 0

    if num_attr != []:
        # set flag
        num_flag = 1
        var_names_num = num_df.columns.tolist()
        var_types = num_df.dtypes.rename('type')
        nunique_value = num_df.nunique().rename('#uniques')
        min_values = num_df.min().rename('minimum')
        max_values = num_df.max().rename('maximum')
        mean_values = num_df.mean(axis=0).rename('mean')
        median_values = num_df.quantile(.5).rename('median')
        sum_values = num_df.sum().rename('sum')
        count_values = num_df.count().rename('my_count')

        # calculate the quantiles
        cal_quantiles = [num_df.quantile(quantile / 100).rename('Q_' + str(quantile)) for quantile in quantiles]

        # concat cols
        num_cols = [var_types, nunique_value, min_values, max_values, mean_values, median_values, sum_values,
                    count_values] + cal_quantiles
        numerical_df = pd.concat(num_cols, axis=1).reset_index().rename(columns={'index': 'vars'})

        # add col of number of observations
        numerical_df['#obs'] = df.shape[0]
        numerical_df['#missing'] = numerical_df['#obs'] - numerical_df['my_count']
        numerical_df['%missing'] = numerical_df['#missing'] / numerical_df['#obs']
        numerical_df['most_value'] = pd.Series([num_df[x].value_counts().iloc[:5].to_dict() for x in num_df.columns])
        del numerical_df['my_count']

    # for categorical vars
    if cat_attr != []:
        # set flag
        cat_flag = 1
        var_names_cat = cat_df.columns
        n_obs = cat_df.count().rename('#obs')
        var_types = cat_df.dtypes.rename('type')
        nunique_value = cat_df.nunique().rename('#uniques')
        count_values = cat_df.count().rename('my_count')
        cat_cols = [n_obs, var_types, nunique_value, count_values]
        categorical_df = pd.concat(cat_cols, axis=1).reset_index().rename(columns={'index': 'vars'})
        categorical_df['#obs'] = df.shape[0]
        categorical_df['#missing'] = categorical_df['#obs'] - categorical_df['my_count']
        categorical_df['%missing'] = categorical_df['#missing'] / categorical_df['#obs']
        categorical_df['most_value'] = pd.Series([cat_df[x].value_counts().iloc[:5].to_dict() for x in cat_df.columns])

        # usr 'str' instead of 'object', for display purpose
        categorical_df['type'] = 'str'
        del categorical_df['my_count']

    if num_flag == 1 and cat_flag == 1:
        my_df = numerical_df.append(categorical_df)

    elif num_flag == 1:
        my_df = numerical_df
        var_names = var_names_num
    elif cat_flag == 1:
        my_df = categorical_df
        var_names = var_names_cat
    else:
        print('NO cols in dataset!')

    # get a list containing quantiles
    import re

    r = re.compile("Q_.")
    Q_list = filter(r.match, my_df.columns)

    column_order = ['vars', 'type', '#uniques', '#obs', '#missing', '%missing', 'sum', 'mean', 'median', 'minimum',
                    'maximum'] + Q_list + ['most_value']
    table = my_df[column_order].reset_index().drop(['index'], axis=1)

    # parameter for showing most_value
    if show_most_value == False:
        del table['most_value']

    # display the summary table
    display(HTML(table.to_html()))

    if save_path != '':

        writer = pd.ExcelWriter(save_path, engine='xlsxwriter')
        table.to_excel(writer)

        workbook = writer.book

        # Add a header format.
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D3D3D3',
            'border': 1})

        # Write the column headers with the defined format.
        for col_num, value in enumerate(table.columns.values):
            worksheet.write(0, col_num + 1, value, header_format)
