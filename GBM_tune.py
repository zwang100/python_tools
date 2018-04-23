from sklearn.grid_search import GridSearchCV  # Perforing grid search
import xgboost as xgb

def gbm_tune_parameters(train, target, seed = 100, to_tune_parameter = {}):
    '''
    :param train:
    :param target:
    :param seed:
    :param tune_parameter: in the form of dict
    :return:
    '''

    default_tune_parameters = {
        'colsample_bytree': [0.4, 0.6, 0.8],
        'gamma': [0, 0.03, 0.1, 0.3],
        'min_child_weight': [1.5, 6, 10],
        'learning_rate': [0.1, 0.07],
        'max_depth': [3, 5],
        'n_estimators': [1000],
        'reg_alpha': [1e-5, 1e-2, 0.75],
        'reg_lambda': [1e-5, 1e-2, 0.45],
        'subsample': [0.6, 0.95]
    }

    if to_tune_parameter != {}:
        # combine the two dicts
        tune_parameters = {**default_tune_parameters, **to_tune_parameter}

    else:
        tune_parameters = default_tune_parameters

    xgb_model = xgb.XGBRegressor(learning_rate=0.1,
                                 n_estimators=100,
                                 max_depth=5,
                                 min_child_weight=1,
                                 gamma=0,
                                 subsample=0.8,
                                 colsample_bytree=0.8,
                                 nthread=-1,
                                 scale_pos_weight=1,
                                 seed=seed)

    gsearch1 = GridSearchCV(estimator=xgb_model, param_grid=tune_parameters, n_jobs=-1, iid=False, verbose=10,
                            scoring='neg_mean_squared_error')

    gsearch1.fit(train, target)
    print(gsearch1.grid_scores_)
    print('best params')
    print(gsearch1.best_params_)
    print('best score')
    print(gsearch1.best_score_)
