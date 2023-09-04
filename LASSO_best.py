import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

def best_lasso(X,y,selection='cyclic'):
    '''
    :param X: Features
    :param y: target
    :return: el valor de lambda que encuentra LASSO.MIN en una grilla de penalziaciones
    '''
    n = len(y)
    eps = 0.001
    cv = 5
    lambda_max = np.linalg.norm(np.matmul(np.transpose(X),y) , np.inf)/n
    lambda_min = eps*lambda_max
    start = np.log10(lambda_min)
    end = np.log10(lambda_max)
    K=100
    lambdas = np.logspace(start,end,K)
    lasso = Lasso(random_state=0, max_iter=1000000,fit_intercept=False,selection=selection)
    tuned_parameters = [{'alpha': lambdas}]
    clf = GridSearchCV(lasso, tuned_parameters, cv=cv, scoring='neg_mean_squared_error')
    clf.fit(X, y)
    best_lambda = clf.best_params_['alpha']
    best_score = clf.cv_results_['mean_test_score'][clf.best_index_]
    best_std = clf.cv_results_['std_test_score'][clf.best_index_]

    lasso_1se_index = np.argmax(np.where(clf.cv_results_['mean_test_score'] < best_score + best_std))
    lambda_1se = lambdas[lasso_1se_index]

    return best_lambda, best_score,lambda_1se