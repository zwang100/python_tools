# imports
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats, integrate
import matplotlib.pyplot as plt
from pandas import Series
import xlsxwriter

from IPython.display import display, HTML
pd.options.display.float_format = '{:,.2f}'.format

def single_univariate_eda(df, xname, loss, premium=None, left_tail=0.05, right_tail=0.95, n=50, ytype='log', path_excel=''):

    y1name=loss
    if premium=None:
        df['premium'] = pd.Series(np.repeat(1, df.shape[0]))
        y0name = 'premium'
    else:
        y0name = premium

    # Calculate the quantile of left bound and right bound

    x_lb = df[xname].quantile(left_tail)
    x_ub = df[xname].quantile(right_tail)

    # masks for left bound and right bound

    mask_lb = df[xname] <= x_lb
    mask_ub = df[xname] >= x_ub

    # cap
    df.loc[mask_lb, xname] = x_lb
    df.loc[mask_ub, xname] = x_ub

    # calculate the bins
    df['bin'] = pd.qcut(df[xname], q=n, duplicates='drop')

    # calculate the ratio: loss/premium
    df['ratio'] = np.where(df[y0name] == 0, df[y1name], df[y1name] / df[y0name])

    # calculate the statistics
    n_obs = df.groupby(df['bin'])[xname].count().rename('OBS')
    avg_x = df.groupby(df['bin'])[xname].mean().rename('AVG_x')
    avg_y1 = df.groupby(df['bin'])[y1name].mean().rename('AVG_y1')
    avg_y0 = df.groupby(df['bin'])[y0name].mean().rename('AVG_y0')
    std_y = df.groupby(df['bin'])['ratio'].std().rename('STD_y')
    pct = (n_obs / df.shape[0] * 100).round(2).rename('PCT')

    # log(y) or true value
    if ytype == 'log': # default value

        avg_ratio_bin = np.log(avg_y1 / avg_y0 + 0.0000001).rename('AVG_y')

    else: # keep the true value
        avg_ratio_bin = (avg_y1 / avg_y0).rename('AVG_y')

    # columns in the output table
    table_cols = [n_obs, pct, avg_x, avg_y0, avg_y1, avg_ratio_bin, std_y]

    # output summary table
    table = pd.concat(table_cols, axis=1)

    table['bin'] = table.index

    table['bin'] = table.bin.astype(str)

    table.reset_index(drop=True, inplace=True)

    # reorder the columns
    new_col_order = ['bin', 'OBS', 'PCT', 'AVG_x', 'AVG_y0', 'AVG_y1', 'AVG_y', 'STD_y']
    table = table[new_col_order]

    # calculate KS
    y1_cdf = (avg_y1 * n_obs).cumsum() / sum(avg_y1 * n_obs)
    y0_cdf = (avg_y0 * n_obs).cumsum() / sum(avg_y0 * n_obs)

    if len(y1_cdf) == 1:
        ks = 0
    else:
        ks = abs(y1_cdf - y0_cdf).max()

    # calculate IV
    y1_pdf = y1_cdf.reset_index().drop(['bin'], axis=1).ix[:, 0] - pd.Series([0] + y1_cdf.tolist()[:-1])
    y0_pdf = y0_cdf.reset_index().drop(['bin'], axis=1).ix[:, 0] - pd.Series([0] + y0_cdf.tolist()[:-1])

    pdf_df = pd.concat([y1_pdf, y0_pdf], axis=1).rename(columns={0: 'y1_pdf', 1: 'y0_pdf'})

    # calculate woe
    def calculate_woe(row):
        if row['y1_pdf'] == 0:
            row['woe'] = row['y0_pdf']
        else:
            if row['y0_pdf'] == 0:
                row['woe'] = row['y1_pdf']
            else:
                row['woe'] = (row['y1_pdf'] - row['y0_pdf']) * np.log(row['y1_pdf'] / row['y0_pdf'])
        return row['woe']

    woe = pdf_df.apply(calculate_woe, axis=1)

    # calculate information value by woe
    iv = round(woe.sum(), 6)

    # plot
    g = sns.regplot(x="AVG_x", y="AVG_y", data=table, fit_reg = True, n_boot=500, y_jitter=.03, ci=68,
                    scatter_kws={'s': 10 * table['PCT']})
    plt.rcParams['figure.figsize'] = (16, 16)
    xlabel1 = xname
    xlabel2 = xname + ' [' + str(left_tail) + ' , ' + str(right_tail) + ']'
    y_label = y1name + '/' + y0name
    g.set_title('ks = ' + str(ks) + '\n' + 'iv = ' + str(iv))
    g.set_xlabel(xlabel2)
    g.set_ylabel(y_label)

    # check if the parameter is set.
    if path_excel == '':
        path_image = './' + 'univariate_' + xname + '.png'
    else:
        path_image = path_excel.replace('xlsx', 'png')

    g.figure.savefig(path_image)

    if path_excel == '':
        path_xlsx = './' + 'univariate_' + xname + '.xlsx'
    else:
        path_xlsx = path_excel

    writer = pd.ExcelWriter(path_xlsx, engine='xlsxwriter')
    table.to_excel(writer, sheet_name=xname)

    workbook = writer.book
    worksheet = writer.sheets[xname]

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

    # Insert an image with scaling.
    worksheet.insert_image('K2', path_image, {'x_scale': 0.8, 'y_scale': 0.8})

    # display the summary table
    display(HTML(table.to_html()))
