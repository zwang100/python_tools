# imports

import pandas as pd, numpy as np, seaborn as sns, matplotlib as mpl, matplotlib.pyplot as plt
from scipy import stats, integrate
from IPython.display import display, HTML
from ggplot import *


# assign bins by weighted quantile
def wtdQuantile(dataframe, var, weight=None, n=10):
    if weight == None:
        return pd.qcut(dataframe[var], n, labels=False, duplicates='drop')
    else:
        dataframe.sort_values(var, ascending=True, inplace=True)
        cum_sum = dataframe[weight].cumsum()
        cutoff = max(cum_sum) / n
        quantile = cum_sum / cutoff
        quantile[-1:] -= 1
        return quantile.map(int)

# calculate KS
def cal_ks(y0_cdf, y1_cdf):
    if len(y1_cdf) == 1:
        return 0
    else:
        return abs(y1_cdf - y0_cdf).max()

# calculate gini
def cal_gini(y0_cdf, y1_cdf):
    # GINI
    y1_cdf_prime = [0.0] + y1_cdf.tolist()[:-1]
    y0_cdf_prime = [0.0] + y0_cdf.tolist()[:-1]
    y1_area = (y1_cdf + y1_cdf_prime) / 2
    y0_area = (y0_cdf + y0_cdf_prime) / 2
    return round(abs(sum((y1_area - y0_area))) / sum(y0_area), 6)

# calculate cdf
def cal_cdf(avg, obs):
    return (avg * obs).cumsum() / sum(avg * obs)

# plot the gains chart
def plot_gains(table, ks, gini):
    sns.set_style("ticks")
    fig, ax = plt.subplots()
    g1 = sns.barplot(x="bin", y="obs", data=table, color='blue', ax=ax)
    ax2 = ax.twinx()
    g2 = sns.pointplot(x="bin", y="relat_lr", data=table, color='red', ax=ax2)
    xlabel = 'bin'
    y_label1 = 'loss ratio'
    y_label2 = 'obs'
    g1.set_title('KS = ' + str(ks) + '\n' + 'GINI = ' + str(gini))
    g1.set_xlabel(xlabel)
    g1.set_ylabel(y_label2)
    g2.set_ylabel(y_label1)

def gains_table(df, y0name, y1name, xname=None, bin=None, n=10):
    '''
    :param df:
    :param xname:
    :param y0name:
    :param y1name:
    :param bin:
    :param n: if has bin, n is ignored.
    :return:
    '''

    # check the input vars. If not int or float, convert.
    if xname and df[xname].dtypes == 'O':
        df[xname] = df[xname].astype('float')
    if y0name and df[y0name].dtypes == 'O':
        df[y0name] = df[y0name].astype('float')
    if y1name and df[y1name].dtypes == 'O':
        df[y1name] = df[y1name].astype('float')

    if bin:
        df['bin'] = df['bin']
    else:
        df['bin'] = wtdQuantile(df, xname, y0name, n)

    obs = df.groupby(df['bin'])['bin'].count().rename('obs')
    avg_x = df.groupby(df['bin'])[xname].mean().rename('average_x')
    avg_y1 = df.groupby(df['bin'])[y1name].mean().rename('avg_loss')
    avg_y0 = df.groupby(df['bin'])[y0name].mean().rename('avg_weight')
    LR = (avg_y1 / avg_y0).rename('lr')
    total_lr = sum(avg_y1 * obs) / sum(avg_y0 * obs)
    relat_lr = (LR / total_lr).rename('relat_lr')
    n_pct = (obs / df.shape[0] * 100).round(2).rename('n_pct')

    my_cols = [obs, n_pct, avg_x, avg_y1, avg_y0, LR, relat_lr]
    table = pd.concat(my_cols, axis=1).reset_index()
    table['bin'] = table['bin'] + 1

    y0_cdf = cal_cdf(avg_y0, obs)
    y1_cdf = cal_cdf(avg_y1, obs)
    ks = cal_ks(y0_cdf, y1_cdf)
    gini = cal_gini(y0_cdf, y1_cdf)
    display(HTML(table.to_html()))
    plot_gains(table, ks, gini)
