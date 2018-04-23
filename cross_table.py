# imports

import seaborn as sns
import pandas as pd


def cross_table(df, column1, column2):
    '''
    :param df: dataframe
    :param column1: var1
    :param column2: var2
    :return:
    '''
    df = pd.crosstab(df[column1], df[column2])

    idx = df.columns.union(df.index)

    df = df.reindex(index=idx, columns=idx, fill_value=0)

    cm = sns.light_palette("green", as_cmap=True)

    cross_table = df.style.background_gradient(cmap=cm)

    return cross_table
