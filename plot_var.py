# imports

'''

To be added later for more complicated plot

'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd, numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = 11.7,8.27
sns.set(style="white",font_scale=1.25)


# plot one variable: continuous variable: strip plot + boxplot; categorical variable: histogram

def one_variable_df(df, var, title='', x_label=''):

    fig, ax = plt.subplots()

    fig.set_size_inches(8, 6)

    if df[var].dtype not in ['int64', 'float64']:

        total = float(len(df))

        ax = sns.countplot(data=df, x=var, saturation=1, edgecolor=(0, 0, 0), linewidth=2, color='c')

        for p in ax.patches:
            height = p.get_height()

            ax.text(p.get_x() + p.get_width() / 2., height + 3, '{:1.2f}%'.format(100 * (height / total)), ha="center")
    else:

        ax = sns.boxplot(x=var, data=df, whis=np.inf, color='c')

        ax = sns.stripplot(x=var, data=df, jitter=True, color=".3")

    # set xlabel, and ylabel

    if x_label == '':

        ax.set(xlabel=var)

    else:

        ax.set(xlabel=x_label)

    if title != '':
        plt.title("Boxplot with number of observation", loc="left")

    # fig.savefig('example.png')


# plot two variables: 1. two continuous variables and 2. one continuous variable and one categorical variable

def two_variable_df(df, x_var='', y_var='', title='', x_label='', y_label=''):

    v = df[x_var], df[y_var]

    # two continous variables

    if all(x.dtype in ['float64', 'int64'] for x in v):

        plt.figure(figsize=(12, 12))

        ax = sns.jointplot(x_var, y_var, data=df, kind="reg", marginal_kws=dict(bins=10, rug=True))

    # one categorical one continuous

    elif any(x.dtype in ['float64', 'int64'] for x in v):

        fig, ax = plt.subplots()

        fig.set_size_inches(12, 8)

        ax = sns.boxplot(x=x_var, y=y_var, data=df, whis=np.inf)

        ax = sns.stripplot(x=x_var, y=y_var, data=df, jitter=True, color='.3')

        # set xlable, and ylable

        if x_label == '':

            ax.set(xlabel=x_var, ylabel=y_var)

        else:

            ax.set(xlabel=x_label, ylabel=y_label)

