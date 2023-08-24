import numpy as np
from sklearn.linear_model import Lasso


def frecuencias_lasso(X,y,selection='cyclic'):
    '''
    Considera una grilla de valores de lambdas y aplica LASSO en todos los lambda de la grilla
    :param X: Features
    :param y: target
    :return: La frecuencia con la que aparece cada variable cuando aplicamos LASSO en una grilla
    '''
    n,p = X.shape
    eps = 0.001
    lambda_max = np.linalg.norm(np.matmul(np.transpose(X),y) , np.inf)/n
    lambda_min = eps*lambda_max
    start = np.log10(lambda_min)
    end = np.log10(lambda_max)
    K=100
    lambdas = np.logspace(start,end,K) # esta es la grilla de valores
    cant_veces_que_elijo_por_feature = np.zeros(p) #
    for lambda_ in lambdas:
        clf = Lasso(alpha=lambda_, fit_intercept=False,selection=selection)
        clf.fit(X, y)
        coeficientes = clf.coef_
        selected_coef = [1 if abs(coeficientes[i])>0.00001 else 0 for i in range(p)]

        cant_veces_que_elijo_por_feature += selected_coef

    frecuencias = [cant/len(lambdas) for cant in cant_veces_que_elijo_por_feature]


    return frecuencias


def frecuencias_lasso_weighted(X,y,selection='cyclic'):
    '''
    Considera una grilla de valores de lambdas y aplica LASSO en todos los lambda de la grilla
    :param X: Features
    :param y: target
    :return: La frecuencia con la que aparece cada variable cuando aplicamos LASSO en una grilla
    '''
    n,p = X.shape
    eps = 0.001
    lambda_max = np.linalg.norm(np.matmul(np.transpose(X),y) , np.inf)/n
    lambda_min = eps*lambda_max
    start = np.log10(lambda_min)
    end = np.log10(lambda_max)
    K=100
    lambdas = np.logspace(start,end,K) # esta es la grilla de valores
    frecuencias_w = p*[0]
    suma_coef_abs=0
    for lambda_ in lambdas:
        clf = Lasso(alpha=lambda_, fit_intercept=False,selection=selection)
        clf.fit(X, y)
        coeficientes = clf.coef_
        suma_coef_abs += sum(abs(coeficientes))
        for i in range(p):
            frecuencias_w[i] += abs(coeficientes[i])


    return frecuencias_w