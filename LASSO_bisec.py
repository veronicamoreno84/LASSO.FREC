from LASSO_areas import area_tray_coef_lasso
from LASSO_best import best_lasso
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from scenarios import scenario_1,scenario_2,scenario_3
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def seleccion_variables_bis(X,y,eps = 0.05,mse='todas'):
    n = len(y)
    frecuencias = area_tray_coef_lasso(X,y)
    best_lambda, best_score, lambda_1se = best_lasso(X,y)
    indeces_ordenan_frecuencias = np.argsort(frecuencias).tolist()
    indeces_ordenan_frecuencias.reverse()
    if mse =='LASSO.MIN':
        clf = Lasso(alpha=best_lambda, fit_intercept=False)
        clf.fit(X, y)
        coeficientes = clf.coef_
        selected_features_LASSOMIN = [i for i in range(len(coeficientes)) if abs(coeficientes[i]) > 0.00001]
        a = 0
        if len(selected_features_LASSOMIN) < n:
            b = len(selected_features_LASSOMIN)-1
        else:
            b = p-1
        mse = 0
        kf = KFold(n_splits=5)
        for train, test in kf.split(X):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

            X_sel_train = [[X_train[i][j] for j in selected_features_LASSOMIN] for i in range(len(y_train))]
            reg = LinearRegression(fit_intercept=False).fit(X_sel_train, y_train)
            X_sel_test = [[X_test[i][j] for j in selected_features_LASSOMIN] for i in range(len(y_test))]
            y_pred = reg.predict(X_sel_test)
            mse += mean_squared_error(y_test, y_pred)
        mse = mse/ 5
        stop = False

        cant_variables = len(selected_features_LASSOMIN)
    if mse == 'todas':
        if p<n:
            b = p-1
        else:
            b = n -1

        a = 0
        mse = 0
        selected_vars = indeces_ordenan_frecuencias[:b]
        kf = KFold(n_splits=5)
        for train, test in kf.split(X):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

            X_sel_train = [[X_train[i][j] for j in selected_vars] for i in range(len(y_train))]
            reg = LinearRegression(fit_intercept=False).fit(X_sel_train, y_train)
            X_sel_test = [[X_test[i][j] for j in selected_vars] for i in range(len(y_test))]
            y_pred = reg.predict(X_sel_test)
            mse += mean_squared_error(y_test, y_pred)
        mse = mse / 5
        stop = False

        cant_variables = b

    while not stop:
        cant_variables_new = int((a+b)/2)
        if cant_variables_new != 0:
            selected_var = indeces_ordenan_frecuencias[:cant_variables_new]
            mse_new = 0
            for train, test in kf.split(X):
                X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
                X_sel_train = [[X_train[i][j] for j in selected_var] for i in range(len(y_train))]
                reg = LinearRegression(fit_intercept=False).fit(X_sel_train, y_train)
                X_sel_test = [[X_test[i][j] for j in selected_var] for i in range(len(y_test))]
                y_pred = reg.predict(X_sel_test)
                mse_new +=  mean_squared_error(y_test, y_pred)
            mse_new = mse_new/5
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
        else:
            stop = True
    return indeces_ordenan_frecuencias[:cant_variables]

def monte_carlo_bisec(scenario, n,p,s,rho=None, cant_clusters = None,cant_sim = 1, eps = 0.05,mse = 'todas'):


    if scenario == '1' or scenario == '3':
        cant_false_fea_selec_LASSOAREAS = 0
        cant_true_fea_selec_LASSOAREAS = 0
    else:
        cant_clusters = 10
        cant_false_fea_selec_LASSOAREAS = cant_clusters * [0]
        cant_true_fea_selec_LASSOAREAS = cant_clusters * [0]

    for _ in range(cant_sim):
        if scenario == '1':
            X, y = scenario_1(n, p, s, sigma2=0.9)
        if scenario == '2':
            X, y = scenario_2(n, p, s, rho, sigma2=0.9, cant_clusters=10)
        if scenario == '3':
            X, y = scenario_3(n, p, s, rho, sigma2=0.9)

        rb = RobustScaler()
        X = rb.fit_transform(X)

        if scenario == '1' or scenario == '3':
            selected_features_LASSOAREAS =  seleccion_variables_bis(X,y,eps = eps,mse=mse)
            false_fea_selec_LASSOAREAS = [i for i in selected_features_LASSOAREAS if i not in range(s)]
            cant_false_fea_selec_LASSOAREAS+= len(false_fea_selec_LASSOAREAS)
            true_fea_selec_LASSOAREAS = [i for i in range(s) if i in selected_features_LASSOAREAS]
            cant_true_fea_selec_LASSOAREAS += len(true_fea_selec_LASSOAREAS)
        else:
            selected_features_LASSOAREAS =  seleccion_variables_bis(X,y,eps = eps,mse=mse)
            for resto in range(cant_clusters):
                false_fea_selec_LASSO_AREAS = [i for i in selected_features_LASSOAREAS if
                                               (i % cant_clusters) == resto and i not in range(s)]
                cant_false_fea_selec_LASSOAREAS[resto] += len(false_fea_selec_LASSO_AREAS)
                true_fea_selec_LASSO_AREAS = [i for i in range(s) if
                                            (i in selected_features_LASSOAREAS and (i % cant_clusters) == resto)]
                cant_true_fea_selec_LASSOAREAS[resto] += len(true_fea_selec_LASSO_AREAS)


    if scenario == '1' or scenario == '3':
        mean_cant_false_fea_selec_LASSOAREAS = cant_false_fea_selec_LASSOAREAS / cant_sim
        mean_cant_true_fea_selec_LASSOAREAS = cant_true_fea_selec_LASSOAREAS / cant_sim

        return mean_cant_false_fea_selec_LASSOAREAS, mean_cant_true_fea_selec_LASSOAREAS

    else:
        mean_cant_false_fea_selec_LASSOAREAS = [cant / cant_sim for cant in
                                              cant_false_fea_selec_LASSOAREAS]
        mean_cant_true_fea_selec_LASSOAREAS = [cant / cant_sim for cant in cant_true_fea_selec_LASSOAREAS]

        return mean_cant_false_fea_selec_LASSOAREAS, mean_cant_true_fea_selec_LASSOAREAS

def grafico_montecarlo_bisec(scenario, n_list, p, s, mse='todas',rho=None,cant_clusters= None, cant_sim = 1, eps = 0.05, showfig = False, savefig = False, save_in = None):

    if scenario == '1' or scenario == '3':
        true_LASSOAREAS_nlist = []
        false_LASSOAREAS_nlist = []
        for n in n_list:
            mean_false_LASSOAREAS, mean_true_LASSOAREAS = monte_carlo_bisec(scenario,n,p,s,rho,cant_sim = cant_sim, eps = eps, mse = mse)
            true_LASSOAREAS_nlist.append(mean_true_LASSOAREAS)
            false_LASSOAREAS_nlist.append(mean_false_LASSOAREAS)

        n_list_string = []
        for n in n_list:
            text = 'n=%s' % (str(n))
            n_list_string.append(text)

        fig, ax = plt.subplots()
        ax.bar(n_list_string, true_LASSOAREAS_nlist, width=1, edgecolor="white", linewidth=0.7)
        ax.bar(n_list_string, false_LASSOAREAS_nlist, bottom=true_LASSOAREAS_nlist,
               width=1, edgecolor="white", linewidth=0.7)
        x_label = 'sample size (n)'
        plt.xlabel(x_label)
        y_label = 'true and false selected variables'
        plt.ylabel(y_label)
        text = r'LASSO.AR Bisec $\epsilon = $ %s' %(eps)
        text_ = r'%s p=%s, s=%s, $\rho=$ %s' % (text, p, s, rho)
        plt.title(text_)

        if savefig:
            filename = save_in + '\SCE%s_LASSO_ARBIS_esp=%s_s%s_rho%s_cant_sim%s.png' % (scenario,eps, s, rho, cant_sim)
            plt.savefig(fname=filename)
        if showfig:
            plt.show()
    else:
        for n in n_list:
            mean_false_LASSOAREAS, mean_true_LASSOAREAS = monte_carlo_bisec(scenario, n,p,s,rho=rho, cant_clusters =cant_clusters,cant_sim = cant_sim, eps = 0.05,mse=mse)
            x = np.arange(cant_clusters)
            fig, ax = plt.subplots()
            ax.bar(x, mean_true_LASSOAREAS, width=1, edgecolor="white", linewidth=0.7)
            ax.bar(x, mean_false_LASSOAREAS, bottom=mean_true_LASSOAREAS,
                   width=1, edgecolor="white", linewidth=0.7)
            x_label = 'mod %s' % (str(cant_clusters))
            plt.xlabel(x_label)
            y_label = 'true and false selected variables'
            plt.ylabel(y_label)
            text = r'LASSO.AR Bisec $\epsilon = $ %s' %(eps)
            text_ = '%s p=%s, n=%s, s=%s, rho=%s' % (text, p, n, s, rho)
            plt.title(text_)

            if savefig:
                filename = save_in + '\SC%s_LASSO_AR_BISeps=%s_n=%s_p=%s_s=%s_rho=%s_cantsim=%s.png' % (
                scenario,eps,n, p, s, rho, cant_sim)
                plt.savefig(fname=filename)
            if showfig:
                plt.show()




scenario = '2'
n_list = [400]
p = 50
s = 10
rho_list = [0.9]
sigma2 = 0.9
cant_clusters = 10
cant_sim = 1000
eps_list= [0.0025,0.005,0.01]
save_in = r'C:\Vero\ML\codigos_Python\Figuras_paper\LASSOARBISmseLASSOMIN\SCENARIO%s' %(scenario)
for eps in eps_list:
    for rho in rho_list:
        grafico_montecarlo_bisec(scenario, n_list, p, s,mse = 'LASSO.MIN', rho=rho,cant_clusters= cant_clusters, cant_sim = cant_sim, eps = eps, showfig = False, savefig = True, save_in = save_in)


# X, y = scenario_1(n, p, s, sigma2)
# rb = RobustScaler()
# X = rb.fit_transform(X)
# vars = seleccion_variables_bis(X,y)
# print('resultado', vars)