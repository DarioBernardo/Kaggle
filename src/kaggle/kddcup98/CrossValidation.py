from sklearn.ensemble import RandomForestClassifier

__author__ = 'dario'

import xgboost as xgb


def build_xgboost_classifier(X_test, X_train_full, y_train_full):

    print "Building xgboost classifier..."

    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.01
    params["min_child_weight"] = 5
    params["subsample"] = 0.8
    params["colsample_bytree"] = 0.8
    params["scale_pos_weight"] = 1.0
    params["silent"] = 1
    params["max_depth"] = 8
    plst = list(params.items())
    num_rounds = 10000
    # Using 4000 rows for early stopping.
    offset = 4000
    # Construct Training matrix and early stopping matrix
    xgtrain = xgb.DMatrix(X_train_full[offset:, :], label=y_train_full[offset:])
    xgval = xgb.DMatrix(X_train_full[:offset, :], label=y_train_full[:offset])
    # Construct matrix for test set
    xgtest = xgb.DMatrix(X_test)
    watchlist = [(xgtrain, 'train'), (xgval, 'val')]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=80)
    y_out = model.predict(xgtest)
    return y_out

def build_random_forest_regressor(X_test, X_train_full, y_train_full):

    print "Building random forest regressor..."

    rf = RandomForestClassifier(n_estimators=800)
    probas_rf = rf.fit(X_train_full, y_train_full).predict(X_test)
    return probas_rf

def build_logistic_regressor(X_test, X_train_full, y_train_full):

    print "Building logistic regressor..."
    logreg = linear_model.LogisticRegression()
    logreg.fit(X_train_full, y_train_full)
    logreg_predict = logreg.predict(X_test)

    return logreg_predict

def build_extra_tree_regressor(X_test, X_train_full, y_train_full):


    print "Building ExtraTrees regressor..."
    etr = ExtraTreesRegressor(n_estimators=500)
    etr.fit(X_train_full, y_train_full)
    etr_predict = etr.predict(X_test)

    return etr_predict

def build_estimators():
    ESTIMATORS = {
        "xgb": build_xgboost_classifier,
        "Extra trees": build_extra_tree_regressor,
        #"Logistic regression": build_logistic_regressor,
        "Random forest": build_random_forest_regressor,
        }

    return ESTIMATORS

def run_cross_validation(train,labels,test,estimators):

    cv = cross_validation.KFold(len(train), n_folds=5, shuffle=True)
    results = []
    index = 1
    for train_indices, test_indices in cv:
        X_train_full, X_test = train[train_indices], train[test_indices]
        y_train_full, y_test = labels[train_indices], labels[test_indices]

        y_out = []
        for name, estimator in estimators.items():
            #print "I am trying to build a {}".format(name)
            y_out_temp = estimator(X_test, X_train_full, y_train_full)
            gini_temp = normalized_gini(y_test, y_out_temp)
            print "{} score is: {}\n".format(name, gini_temp)
            #print "{} predictions are: {}\n".format(name, y_out_temp)
            if len(y_out) == 0:
                y_out = y_out_temp
            else:
                y_out += y_out_temp

        gini = normalized_gini(y_test, y_out)
        print "All AUC scores: {}".format(gini)

        print "AUC score for fold {}: {}".format(index, gini)
        index += 1
        results.append(gini)

    # now print out the mean of the cross-validated results
    print "All results: {}".format(results)
    print "Avg auc across all folds: " + str(np.array(results).mean())
