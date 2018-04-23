
# imports
from IPython.display import display, HTML
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
pd.options.mode.chained_assignment = None

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import seaborn as sns
from scipy import stats, integrate
import matplotlib.pyplot as plt
from pandas import Series
import xlsxwriter

import os
import glob

# input is strings xnames = ['col1', 'col2'], loss = 'loss_col_name'

def batch_univariate(df, xnames, loss, impute = ["mean", "median", "zero", "none"], premium=None, left_tail=0.05, right_tail=0.95, n=50, ytype='log', path_excel='', display_plots=False):

    '''
    :param df: dataframe
    :param xnames:
    :param y0name:
    :param y1name:
    :param left_tail:
    :param right_tail:
    :param n:
    :param ytype:
    :param path_excel:
    :return:
    '''

    tables = []
    ivs = []
    kss = []

    y1name = loss
    


    # If premium is not given, assign all 1's  so that the y is loss

    if premium is None:
        # Need to copy df so that the data set isn't permanently altered (premium and imputing). 
        # Only keep needed columns to reduce run time.
        df = df[xnames + [loss]]
        #  Create col of all 1's
        df['premium'] = pd.Series(np.repeat(1, df.shape[0]))
        y0name = 'premium'
    else:
        # Need to copy df so that the data set isn't permanently altered (premium and imputing). 
        # Only keep needed columns to reduce run time.
        df = df[xnames + [loss, premium]]
        y0name = premium


    # loop the Xs
    for xname in xnames:

        # implement the desired impute method
        if impute == "mean":
            df[xname] = df[xname].fillna(np.nanmean(df[xname]))

        elif impute == "median":
            df[xname] = df[xname].fillna(np.nanmedian(df[xname]))
        elif impute == "zero":
            df[xname] = df[xname].fillna(0)

        # Calculate the quantile of left bound and right bound

        x_lb = df[xname].quantile(left_tail)
        x_ub = df[xname].quantile(right_tail)

        # masks for left bound and right bound

        mask_lb = df[xname] <= x_lb
        mask_ub = df[xname] >= x_ub

        # set the bounds
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


        # create new column "bin", and set the type "str"
        table['bin'] = table.index.astype(str)

        # reset index
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
                        scatter_kws={'s': 50 * table['PCT']})
        sns.set(font_scale=2)

        if display_plots == True:
            plt.rcParams['figure.figsize'] = (7,5)
            
        else:
            plt.rcParams['figure.figsize'] = (16, 12)
            

        #xlabel1 = xname
        xlabel2 = xname + ' [' + str(left_tail) + ' , ' + str(right_tail) + ']'
        y_label = 'log(' + y1name + '/' + y0name + ')'
        g.set_title('ks = ' + str(ks) + '\n' + 'iv = ' + str(iv))
        g.set_xlabel(xlabel2, fontsize = 25)
        g.set_ylabel(y_label, fontsize = 25)
       

        # save the images in the temp folder
        path_image = './' + xname + '.png'
        g.figure.savefig(path_image)


        # collect the outputs
        tables.append(table)
        ivs.append(iv)
        kss.append(ks)

        if display_plots == True:
            plt.show()

        plt.clf()

    
    if display_plots == False:    

        # check if the excel path is set.
        if path_excel == '':
            path_xlsx = './' + 'univariate_all_attributes.xlsx'
        else:
            path_xlsx = path_excel

        writer = pd.ExcelWriter(path_xlsx, engine='xlsxwriter')
        workbook = writer.book


        for i, table in enumerate(tables):

            table.to_excel(writer, sheet_name=xnames[i])

            worksheet = writer.sheets[xnames[i]]

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
            worksheet.insert_image('K2','./'+ xnames[i] + '.png', {'x_scale': 0.4, 'y_scale': 0.4})


        writer.close()

    # remove the temporary image files
    images = glob.glob('./*.png')
    for image in images:
        os.remove(image)
