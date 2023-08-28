import numpy as np
from sklearn.linear_model import Lasso

def area_tray_coef_lasso(X,y,selection='cyclic'):
    '''
    Considera una grilla de valores de lambdas y aplica LASSO en todos los lambda de la grilla
    :param X: Features
    :param y: target
    :return: El area bajo la curva del coeficiente, normalizado (para que sume uno el vector de las areas)
    '''
    n,p = X.shape
    eps = 0.001
    lambda_max = np.linalg.norm(np.matmul(np.transpose(X),y) , np.inf)/n
    lambda_min = eps*lambda_max
    start = np.log10(lambda_min)
    end = np.log10(lambda_max)
    K=100
    lambdas = np.logspace(start,end,K) # esta es la grilla de valores
    areas = p * [0]
    for i in range(len(lambdas)-1):
        lambda_ = lambdas[i]
        lambda_next = lambdas[i+1]
        clf = Lasso(alpha=lambda_, fit_intercept=False, selection=selection)
        clf.fit(X, y)
        coeficientes = clf.coef_
        for i in range(p):
            areas[i] += abs(coeficientes[i])*(lambda_next-lambda_)
    norm = [area / sum(areas) for area in areas]
    return norm
