import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from IPython.display import display, HTML

pd.options.display.float_format = '{:,.2f}'.format


def feature_importance(df, train, target, dummies=[], fill_na=-999,
                       methods=['rlasso', 'RFE', 'LinReg', 'Ridge', 'Lasso', 'RF', 'GBM']):

    # in lower and upper form of the methods names
    methods_lower = [x.lower() for x in methods]
    methods_upper = [x.upper() for x in methods]

    # combine names
    methods = methods + methods_lower + methods_upper

    # target
    Y = df[target].values

    # deal the training data
    df = df[train].fillna(fill_na)

    # dummies
    if dummies != []:
        for x in dummies:
            dummie_x = pd.get_dummies(df.x, prefix=x + '_').iloc[:, 1:]
            df = pd.concat([df, dummie_x], axis=1)

    def get_cat_features(df):
        return list(df.select_dtypes(include=['object']).columns)

    # automatically detect categorical variables
    cat_attr = get_cat_features(df)

    for x in cat_attr:
        dummie_x = pd.get_dummies(df.x, prefix=x + '_').iloc[:, 1:]
        df = pd.concat([df, dummie_x], axis=1)

    # get all attributes names
    colnames = df.columns

    # attributes
    X = df.values

    # Define dictionary to store our rankings
    ranks = {}

    # Create our function which stores the feature rankings to the ranks dictionary
    def ranking(ranks, names, order=1):
        minmax = MinMaxScaler()
        ranks = minmax.fit_transform(order * np.array([ranks]).T).T[0]
        ranks = map(lambda x: round(x, 2), ranks)
        return dict(zip(names, ranks))

    '''
    Randomized Lasso
    '''
    if 'rlasso' in methods:
        # Selection Stability method with Randomized Lasso
        rlasso = RandomizedLasso(alpha=0.04)
        rlasso.fit(X, Y)
        ranks["rlasso/Stability"] = ranking(np.abs(rlasso.scores_), colnames)

    '''
    Recursive Feature Elimination ( RFE )
    '''
    if 'RFE' in methods:
        # Construct our Linear Regression model
        lr = LinearRegression(normalize=True)
        lr.fit(X, Y)
        # stop the search when only the last feature is left
        rfe = RFE(lr, n_features_to_select=1, verbose=0)
        rfe.fit(X, Y)
        ranks['RFE'] = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)

    '''
    Linear Model Feature Ranking
    '''
    if 'LinReg' in methods:
        # Using Linear Regression
        lr = LinearRegression(normalize=True)
        lr.fit(X, Y)
        ranks['LinReg'] = ranking(np.abs(lr.coef_), colnames)

    # Using Ridge

    if 'Ridge' in methods:
        ridge = Ridge(alpha=7)
        ridge.fit(X, Y)
        ranks['Ridge'] = ranking(np.abs(ridge.coef_), colnames)

    # Using Lasso

    if 'Lasso' in methods:
        lasso = Lasso(alpha=.05)
        lasso.fit(X, Y)
        ranks['Lasso'] = ranking(np.abs(lasso.coef_), colnames)

    '''
    random forest
    '''
    if 'RF' in methods:
        # parameters
        rf_params = {
            'n_jobs': -1,
            'n_estimators': 100,
            'warm_start': True,
            'max_features': 0.3,
            'max_depth': 3,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 100,
            'verbose': 0
        }
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X, Y)

        ranks['RF'] = ranking(rf.feature_importances_, colnames)

    '''
    Gradient Boosting Machine
    '''
    if 'GBM' in methods:
        # parameters
        gbm_params = {
            'nthread': -1,
            'colsample_bytree': 0.4,
            'gamma': 0,
            'reg_alpha': 0.75,
            'reg_lambda': 0.45,
            'subsample': 0.6,
            'learning_rate': 0.07,
            'max_depth': 3,
            'min_child_weight': 1.5,
            'n_estimators': 100,
            'seed': 100
        }

        gbm = xgb.XGBRegressor(**gbm_params)
        gbm.fit(X, Y)
        ranks['GBM'] = ranking(gbm.feature_importances_, colnames)

    # Create empty dictionary to store the mean value calculated from all the scores
    r = {}
    for name in colnames:
        r[name] = round(np.mean([ranks[method][name]
                                 for method in ranks.keys()]), 2)

    methods = sorted(ranks.keys())
    ranks["Mean"] = r
    methods.append("Mean")

    matrix_importance = pd.DataFrame(ranks)

    print(matrix_importance.columns)

    # change the display oder of cols
    ordered_cols = ['rlasso/Stability','LinReg', 'Lasso', 'Ridge', 'RFE', 'GBM', 'RF', 'Mean']
    
    matrix_importance = matrix_importance[ordered_cols]

    # display the summary table
    display(HTML(matrix_importance.to_html()))

    # Put the mean scores into a Pandas dataframe
    meanplot = pd.DataFrame(list(r.items()), columns=['Feature', 'Mean Ranking'])

    # Sort the dataframe
    meanplot = meanplot.sort_values('Mean Ranking', ascending=False)

    # Let's plot the ranking of the features
    sns.factorplot(x="Mean Ranking", y="Feature", data=meanplot, kind="bar",
                   size=14, aspect=1.9, palette='coolwarm')
