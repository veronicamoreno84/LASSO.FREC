from LASSO_areas import area_tray_coef_lasso
from LASSO_best import best_lasso
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from scenarios import scenario_1,scenario_2,scenario_3


def seleccion_variables_bis(X,y,eps = 0.05):
    frecuencias = area_tray_coef_lasso(X,y)
    best_lambda, best_score, lambda_1se = best_lasso(X,y)
    indeces_ordenan_frecuencias = np.argsort(frecuencias).tolist()
    indeces_ordenan_frecuencias.reverse()
    clf = Lasso(alpha=best_lambda, fit_intercept=False)
    clf.fit(X, y)
    coeficientes = clf.coef_
    selected_features_LASSOMIN = [i for i in range(len(coeficientes)) if abs(coeficientes[i]) > 0.00001]
    a = 0
    b = len(selected_features_LASSOMIN)-1
    cant_varaibles = len(selected_features_LASSOMIN)
    mse = -best_score
    stop = False
    while not stop:
        cant_variables_new = int((a+b)/2)
        print(b)
        print(cant_variables_new)
        selected_var = indeces_ordenan_frecuencias[:cant_variables_new]
        print('selected var eleccion del threshold', selected_var)
        #aca voy a mirar el mse pero despues tendria que cross-validarlo
        X_sel = [[X[i][j] for j in selected_var] for i in range(len(y))]
        reg = LinearRegression(fit_intercept=False).fit(X_sel, y)
        X_sel = [[X[i][j] for j in selected_var] for i in range(len(y))]
        y_pred = reg.predict(X_sel)
        mse_new = mean_squared_error(y, y_pred)
        print('mse', mse, 'mse new', mse_new)
        if mse_new < mse*(1+eps):
            mse = mse_new
            if cant_variables_new != b:
                b = cant_variables_new
                cant_variables = cant_variables_new
            else:
                stop = True
        elif cant_variables_new != a:
            a = cant_variables_new
        else:
            stop = True
    return indeces_ordenan_frecuencias[:cant_variables]

n = 400
p = 50
s = 10
rho = 0.9
sigma2 = 0.9
cant_clusters = 10
X, y = scenario_2(n, p, s,rho, sigma2, cant_clusters)
rb = RobustScaler()
X = rb.fit_transform(X)
vars = seleccion_variables_bis(X,y)
print('resultado', vars)