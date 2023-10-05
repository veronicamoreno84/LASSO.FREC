import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler
import numpy as np
from scenarios import scenario_1,scenario_2,scenario_3, scenario_1_random_coef, scenario_2_random_coef,scenario_3_coef_random
import matplotlib.pyplot as plt

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
    for lambda_ in lambdas:
        clf = Lasso(alpha=lambda_, fit_intercept=False, selection=selection)
        clf.fit(X, y)
        coeficientes = clf.coef_
        for i in range(p):
            areas[i] += abs(coeficientes[i])*lambda_
    norm = [area / sum(areas) for area in areas]
    return norm


def grafico_frecuencias_ordenadas(scenario,n,p,s,rho=None, showfig = False, savefig = True, save_in = None,
                                  cant_simu = 1,selection = 'cyclic'):
    '''

    :param scenario: El escenario a simular
    :param n: la cantidad de datos
    :param p: la cantidad de features o variables
    :param s: la cantidad de variables verdaderas que generan el targuet
    :param cantidad_sim: cantidad de simulaciones independientes que hago
    :param tau_list : los posibles thresholds, selecciono las variables cuya frecuencia esta por encima de este threshold
    :param rh0: el coeficiente de correlacion, que se usa en los escenarios 2 y 3
    :return: genera los graficos de las frecuencias
    '''

    for sim in range(cant_simu):
        if scenario =='1':
            X, y = scenario_1(n, p, s, sigma2=0.9)
        if scenario =='1random':
            coef = [np.random.uniform(low=-0.5, high=0.5) for _ in range(s)]
            X, y= scenario_1_random_coef(n, p, s, sigma2=0.9,coef=coef)
        if scenario == '2':
            X, y = scenario_2(n, p, s, rho, sigma2=0.9, cant_clusters=10)
        if scenario =='2random':
            coef = [np.random.uniform(low=-0.5, high=0.5) for _ in range(s)]
            X, y= scenario_2_random_coef(n, p, s, rho,sigma2=0.9,cant_clusters=10,coef=coef)
        if scenario == '3':
            X, y = scenario_3(n, p, s, rho, sigma2=0.9)
        if scenario == '3random':
            coef = [np.random.uniform(low=-0.5, high=0.5) for _ in range(s)]
            X, y = scenario_3_coef_random(n, p, s, rho, sigma2=0.9,coef=coef)


        rb = RobustScaler()
        X = rb.fit_transform(X)

        frecuencias_ = area_tray_coef_lasso(X, y, selection)


        indeces_ordenan_frecuencias = np.argsort(frecuencias_).tolist()
        indeces_ordenan_frecuencias.reverse()
        indeces_ordenan_frecuencias_true = [idx for idx, x in enumerate(indeces_ordenan_frecuencias) if
                                            x in range(s)]
        indeces_ordenan_frecuencias_false = [idx for idx, x in enumerate(indeces_ordenan_frecuencias) if
                                             x not in range(s)]
        frecuencias_ord_true_var = [frecuencias_[i] for i in indeces_ordenan_frecuencias if i in range(s)]
        frecuencias_ord_false_var = [frecuencias_[i] for i in indeces_ordenan_frecuencias if i not in range(s)]
        fig, ax = plt.subplots()

        plt.plot(indeces_ordenan_frecuencias_true, frecuencias_ord_true_var, 'co', color='blue', label='True variables')
        plt.plot(indeces_ordenan_frecuencias_false, frecuencias_ord_false_var, 'co', color='green',label='False variables' )

        if scenario == '1random' or scenario == '2random' or scenario == '3random':
            coef_ordered = [coef[x] for idx, x in enumerate(indeces_ordenan_frecuencias) if
                                                x in range(s)]
            plt.plot(indeces_ordenan_frecuencias_true, coef_ordered, 'co', color='red', label = 'Coeficients')

        title = r'Ordered areas scenario %s  n=%s, s=%s, $\rho$=%s' % (scenario, n, s, rho)
        plt.title(title)
        plt.legend()
        if savefig:
            filename = r'\SC%s_AR%s_n%s_p%s_s%s_rho%s_%s.png' % (scenario,selection,n, p, s, rho,sim)
            if save_in is not None:
                filename = save_in +filename
            plt.savefig(fname=filename)
        if showfig:
            plt.show()


# scenario = '3random'
# n_list = [400]
# p = 50
# s = 10
# rho_list = [0.2,0.5,0.9]
# cant_sim = 1
# selection='cyclic'
# save_in = r'C:\Vero\ML\codigos_Python\Figuras_paper\Areas\SCENARIO%s' %(scenario)
# for n in n_list:
#     for rho in rho_list:
#         grafico_frecuencias_ordenadas(scenario, n ,p, s, rho = rho, showfig=True, savefig=False,
#                                            save_in = save_in, cant_simu = cant_sim,selection=selection)
# #
# cant_clusters=10
# for rho in rho_list:
#     for tau in tau_list:
#         #grafico_monte_carlo_por_cluster(scenario, cant_clusters, n, p, s, rho=rho, cantidad_sim=cant_sim, tau=tau,
#                                       # showfig=True, savefig=False, save_in=save_in)
#         grafico_frecuencias_ordenadas(scenario, n ,p, s, tau_list=tau_list, rho = rho, showfig=True, savefig=False,
#                                            save_in = None, cant_simu = cant_sim,selection=selection, weight = False)
# #