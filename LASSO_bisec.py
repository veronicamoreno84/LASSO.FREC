from LASSO_areas import area_tray_coef_lasso
from LASSO_best import best_lasso
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from scenarios import scenario_1,scenario_2,scenario_3
from sklearn.model_selection import KFold


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
    kf = KFold(n_splits=5)
    splits = kf.split(X)


    while not stop:
        cant_variables_new = int((a+b)/2)
        print(b)
        print(cant_variables_new)
        selected_var = indeces_ordenan_frecuencias[:cant_variables_new]
        mse_new = 0
        for train, test in kf.split(X):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]


            print('selected var eleccion del threshold', selected_var)
            #aca voy a mirar el mse pero despues tendria que cross-validarlo
            X_sel_train = [[X_train[i][j] for j in selected_var] for i in range(len(y_train))]
            reg = LinearRegression(fit_intercept=False).fit(X_sel_train, y_train)
            X_sel_test = [[X_test[i][j] for j in selected_var] for i in range(len(y_test))]
            y_pred = reg.predict(X_sel_test)
            mse_new +=  mean_squared_error(y_test, y_pred)
        mse_new = mse_new/5
        print('mse', mse, 'mse new', mse_new)
        if mse_new < mse*(1+eps):
            #mse = mse_new
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

def monte_carlo_bisec(scenario,n,p,s,cant_sim = 1, eps = 0.05):

    for _ in range(len(cant_sim)):
        if scenario == '1' or scenario == '3':
            cant_false_fea_selec_LASSOAREAS = 0
            cant_true_fea_selec_LASSOAREAS = 0
        else:
            cant_clusters = 10
            cant_false_fea_selec_LASSOAREAS = cant_clusters * [0]
            cant_true_fea_selec_LASSOAREAS = cant_clusters * [0]

        for _ in range(cantidad_sim):
            if scenario == '1':
                X, y = scenario_1(n, p, s, sigma2=0.9)
            if scenario == '2':
                X, y = scenario_2(n, p, s, rho, sigma2=0.9, cant_clusters=10)
            if scenario == '3':
                X, y = scenario_3(n, p, s, rho, sigma2=0.9)

            rb = RobustScaler()
            X = rb.fit_transform(X)

            if scenario == '1' or scenario == '3':
                selected_features_LASSOAREAS =  seleccion_variables_bis(X,y,eps = eps)
                false_fea_selec_LASSOAREAS = [i for i in selected_features_LASSOAREAS if i not in range(s)]
                cant_false_fea_selec_LASSOFREC += len(false_fea_selec_LASSOFREC)
                true_fea_selec_LASSOAREAS = [i for i in range(s) if i in selected_features_LASSOAREAS]
                cant_true_fea_selec_LASSOAREAS += len(true_fea_selec_LASSOAREAS)
            else:
                selected_features_LASSOAREAS =  seleccion_variables_bis(X,y,eps = eps)
                for resto in range(cant_clusters):
                    false_fea_selec_LASSO_AREAS = [i for i in selected_features_LASSOAREAS if
                                                   (i % cant_clusters) == resto and i not in range(s)]
                    cant_false_fea_selec_LASSOAREAS[resto] += len(false_fea_selec_LASSO_AREAS)
                    true_fea_selec_LASSO_AREAS = [i for i in range(s) if
                                                (i in selected_features_LASSOAREAS and (i % cant_clusters) == resto)]
                    cant_true_fea_selec_LASSOAREAS[resto] += len(true_fea_selec_LASSO_AREAS)


        if scenario == '1' or scenario == '3':
            mean_cant_false_fea_selec_LASSOAREAS = cant_false_fea_selec_LASSOAREAS / cantidad_sim
            mean_cant_true_fea_selec_LASSOAREAS = cant_true_fea_selec_LASSOAREAS / cantidad_sim

            return mean_cant_false_fea_selec_LASSOAREAS, mean_cant_true_fea_selec_LASSOAREAS

        else:
            mean_cant_false_fea_selec_LASSOAREAS = [cant / cantidad_sim for cant in
                                                  cant_false_fea_selec_LASSOAREAS
            mean_cant_true_fea_selec_LASSOAREAS = [cant / cantidad_sim for cant in cant_true_fea_selec_LASSOAREAS]

            return mean_cant_false_fea_selec_LASSOAREAS, mean_cant_true_fea_selec_LASSOAREAS

def grafico_montecarlo_bisec_esc1y2(scenario, n_list, p, s, cant_sim = 1, eps = 0.05, show_fig = False, save_fig = False, save_in = None):

    mean_cant_true_fea_selec_LASSOAREAS = []
    mean_cant_false_fea_selec_LASSOAREAS = []
    for n in n_list:
        mean_false_LASSOAREAS, mean_true_LASSOAREAS = monte_carlo_bisec(scenario,n,p,s,cant_sim = cant_sim, eps = eps)
        mean_cant_true_fea_selec_LASSOAREAS.append(mean_true_LASSOAREAS)
        mean_cant_false_fea_selec_LASSOAREAS.append(mean_false_LASSOAREAS)


    n_list_string = []
    for n in n_list:
        text = 'n=%s' % (str(n))
        n_list_string.append(text)

    fig, ax = plt.subplots()
    ax.bar(n_list_string, mean_true_LASSOAREAS, width=1, edgecolor="white", linewidth=0.7)
    ax.bar(n_list_string, mean_false_LASSOAREAS, bottom=mean_true_LASSOAREAS,
           width=1, edgecolor="white", linewidth=0.7)
    x_label = 'sample size (n)'
    plt.xlabel(x_label)
    y_label = 'true and false selected variables'
    plt.ylabel(y_label)
    text = r'LASSO.AREAS'
    text_ = r'%s p=%s, s=%s, $\rho=$ %s' % (text, p, s, rho)
    plt.title(text_)

    if savefig:
        filename = save_in + '\SCE%s_LASSOAREAS_s%s_rho%s_cant_sim%s.png' % (scenario, s, rho, cantidad_sim)
        plt.savefig(fname=filename)
    if showfig:
        plt.show()



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