seed = 29
IS_ALERT_PARAMS = {
    'learning_rate': 0.01,
    'lambda_l1': 0.0009782288206799821,
    'lambda_l2': 0.00021248013254914345,
    'min_sum_hessian_in_leaf': 16,
    'feature_fraction': 0.25633622082985474,
    'feature_fraction_bynode': 0.8,
    'bagging_fraction': 0.8504557457590274,
    'bagging_freq': 16,
    'min_child_samples': 22,
    'num_leaves': 500,
    'max_depth': 15,
    'seed': seed,
    'feature_fraction_seed': seed,
    'bagging_seed': seed,
    'drop_seed': seed,
    'data_random_seed': seed,
    'boosting': 'gbdt',
    'verbosity': -1,
    'n_jobs': -1,
    'objective': 'binary',
    'metric': 'binary',
}

LGB_BASIC_SETTING = {
    'learning_rate': 0.01,
    'lambda_l1': 0.0009782288206799821,
    'lambda_l2': 0.00021248013254914345,
    'min_sum_hessian_in_leaf': 16,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    # 'bagging_freq': 16,
    'min_child_samples': 400,
    'num_leaves': 16,
    'max_depth': 3,
    'seed': seed,
    'feature_fraction_seed': seed,
    'bagging_seed': seed,
    'drop_seed': seed,
    'data_random_seed': seed,
    'boosting': 'gbdt',
    'verbosity': -1,
    'n_jobs': -1,
    'objective': 'binary',
    'metric': 'binary',
    'first_metric_only': True,
    "scale_pos_weight": 2,
}