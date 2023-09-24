from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler
import numpy as np
from scenarios import scenario_1,scenario_2,scenario_3
import matplotlib.pyplot as plt
from LASSO_frec import frecuencias_lasso, frecuencias_lasso_weighted
from LASSO_areas import area_tray_coef_lasso
from LASSO_best import best_lasso


def grafico_frecuencias_ordenadas(scenario,n,p,s,tau_list=[0.8],rho=None, showfig = False, savefig = True, save_in = None,
                                  cant_simu = 1,selection = 'cyclic', weight = False):
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
        if scenario == '2':
            X, y = scenario_2(n, p, s, rho, sigma2=0.9, cant_clusters=10)
        if scenario == '3':
            X, y = scenario_3(n, p, s, rho, sigma2=0.9)

        rb = RobustScaler()
        X = rb.fit_transform(X)

        if not weight:
            frecuencias_ = frecuencias_lasso(X,  y,selection)
        else :
            frecuencias_ = frecuencias_lasso_weighted(X, y, selection)



        indeces_ordenan_frecuencias = np.argsort(frecuencias_).tolist()
        indeces_ordenan_frecuencias.reverse()
        print('indices ordenan areas', indeces_ordenan_frecuencias)
        indeces_ordenan_frecuencias_true = [idx for idx, x in enumerate(indeces_ordenan_frecuencias) if
                                            x in range(s)]
        indeces_ordenan_frecuencias_false = [idx for idx, x in enumerate(indeces_ordenan_frecuencias) if
                                             x not in range(s)]
        frecuencias_ord_true_var = [frecuencias_[i] for i in indeces_ordenan_frecuencias if i in range(s)]
        frecuencias_ord_false_var = [frecuencias_[i] for i in indeces_ordenan_frecuencias if i not in range(s)]
        fig, ax = plt.subplots()
        # for tau in tau_list:
        #     plt.axhline(y=tau, color='black', linestyle='--',
        #                 linewidth=1, label='bla')
        #
        # plt.axvline(x=s, color='black', linestyle='--',
        #             linewidth=1, label='bla')
        plt.plot(indeces_ordenan_frecuencias_true, frecuencias_ord_true_var, 'co', color='blue', )
        plt.plot(indeces_ordenan_frecuencias_false, frecuencias_ord_false_var, 'co', color='green', )
        title = r'Ordered frequencies  n=%s, s=%s, $\rho$=%s' % (n, s, rho)
        plt.title(title)
        if savefig:
            filename = r'\SC%s_FREC%s_n%s_p%s_s%s_rho%s_%s.png' % (scenario,selection,n, p, s, rho,sim)
            if save_in is not None:
                filename = save_in +filename
            plt.savefig(fname=filename)
        if showfig:
            plt.show()

def monte_carlo_LASSOMIN_VS_LASSOFREC(scenario,n,p,s, cantidad_sim=1,rho=None,tau=0.8):
    '''

    :param scenario: El scenario que voy a simular
    :param n: tamaño de la muestra
    :param p: cantidad de features
    :param s: cantidad de features involucradas en el modelo
    :param cantidad_sim:  cantidad de simulaciones a promediar
    :param rho: correlacion que solo se usa en los escenarios 2 y 3
    :param tau: parametro de LASSO.FREC: selecciono las variables cuya frecuencia es mayor a tau.
    :return: el promedio de las variables verdaderas y falsas seleccionadas en cada uno de los casos
    LASSO.MIN y LASSO.FREC, si el escenario es el 2 me devuelve las verdaderas y falsas en cada cluster
    '''

    if scenario == '1' or scenario == '3':
        cant_false_fea_selec_LASSOMIN = 0
        cant_true_fea_selec_LASSOMIN = 0
        cant_false_fea_selec_LASSO1se = 0
        cant_true_fea_selec_LASSO1se = 0
        cant_false_fea_selec_LASSOFREC = 0
        cant_true_fea_selec_LASSOFREC = 0
    else:
        cant_clusters = 10
        cant_false_fea_selec_LASSOMIN = cant_clusters*[0]
        cant_true_fea_selec_LASSOMIN = cant_clusters*[0]
        cant_false_fea_selec_LASSOFREC = cant_clusters*[0]
        cant_true_fea_selec_LASSOFREC = cant_clusters*[0]

    for _ in range(cantidad_sim):
        if scenario =='1':
            X, y = scenario_1(n, p, s, sigma2=0.9)
        if scenario == '2':
            X, y = scenario_2(n, p, s, rho, sigma2=0.9, cant_clusters=10)
        if scenario == '3':
            X, y = scenario_3(n, p, s, rho, sigma2=0.9)

        rb = RobustScaler()
        X = rb.fit_transform(X)

        best_lambda, lambda_1se = best_lasso(X, y)
        clf = Lasso(alpha=best_lambda, fit_intercept=False)
        clf.fit(X, y)
        coeficientes = clf.coef_
        selected_features_LASSOMIN = [i for i in range(len(coeficientes)) if abs(coeficientes[i]) > 0.00001]
        clf = Lasso(alpha=lambda_1se, fit_intercept=False)
        clf.fit(X, y)
        coeficientes = clf.coef_
        selected_features_LASSO1se = [i for i in range(len(coeficientes)) if abs(coeficientes[i]) > 0.00001]

        if scenario == '1' or scenario == '3':
            false_fea_selec_LASSOMIN = [i for i in range(p) if (i in selected_features_LASSOMIN and i not in range(s))]
            cant_false_fea_selec_LASSOMIN += len(false_fea_selec_LASSOMIN)
            true_fea_selec_LASSOMIN = [i for i in range(s) if i in selected_features_LASSOMIN]
            cant_true_fea_selec_LASSOMIN += len(true_fea_selec_LASSOMIN)

            false_fea_selec_LASSO1se = [i for i in range(p) if (i in selected_features_LASSO1se and i not in range(s))]
            cant_false_fea_selec_LASSO1se += len(false_fea_selec_LASSO1se)
            true_fea_selec_LASSO1se = [i for i in range(s) if i in selected_features_LASSO1se]
            cant_true_fea_selec_LASSO1se += len(true_fea_selec_LASSO1se)

            frecuencias_ = frecuencias_lasso(X, y)
            selected_features_LASSOFREC = [i for i in range(p) if frecuencias_[i] > tau]

            false_fea_selec_LASSOFREC = [i for i in range(p) if
                                         (i in selected_features_LASSOFREC and i not in range(s))]
            cant_false_fea_selec_LASSOFREC += len(false_fea_selec_LASSOFREC)
            true_fea_selec_LASSOFREC = [i for i in range(s) if i in selected_features_LASSOFREC]
            cant_true_fea_selec_LASSOFREC += len(true_fea_selec_LASSOFREC)
        else:
            frecuencias_ = frecuencias_lasso(X, y)
            selected_features_LASSOFREC = [i for i in range(p) if frecuencias_[i] > tau]
            for resto in range(cant_clusters):
                false_fea_selec_LASSO_MIN = [i for i in range(p) if (
                      i in selected_features_LASSOMIN and (i % cant_clusters) == resto) and i not in range(s)]
                cant_false_fea_selec_LASSOMIN[resto] += len(false_fea_selec_LASSO_MIN)
                true_fea_selec_LASSO_MIN = [i for i in range(s) if
                                                     (i in selected_features_LASSOMIN and (i % cant_clusters) == resto)]
                cant_true_fea_selec_LASSOMIN[resto] += len(true_fea_selec_LASSO_MIN)

                false_fea_selec_LASSO_FREC = [i for i in range(p) if (
                        i in selected_features_LASSOFREC and (i % cant_clusters) == resto) and i not in range(s)]
                cant_false_fea_selec_LASSOFREC[resto] += len(false_fea_selec_LASSO_FREC)
                true_fea_selec_LASSO_FREC = [i for i in range(s) if
                                            (i in selected_features_LASSOFREC and (i % cant_clusters) == resto)]
                cant_true_fea_selec_LASSOFREC[resto] += len(true_fea_selec_LASSO_FREC)






    if scenario == '1'or scenario == '3':
        mean_cant_false_fea_selec_LASSOMIN = cant_false_fea_selec_LASSOMIN/cantidad_sim
        mean_cant_true_fea_selec_LASSOMIN = cant_true_fea_selec_LASSOMIN/cantidad_sim

        mean_cant_false_fea_selec_LASSO1se = cant_false_fea_selec_LASSO1se / cantidad_sim
        mean_cant_true_fea_selec_LASSO1se = cant_true_fea_selec_LASSO1se / cantidad_sim

        mean_cant_false_fea_selec_LASSOFREC = cant_false_fea_selec_LASSOFREC/cantidad_sim
        mean_cant_true_fea_selec_LASSOFREC = cant_true_fea_selec_LASSOFREC/cantidad_sim
        return mean_cant_true_fea_selec_LASSOMIN, mean_cant_false_fea_selec_LASSOMIN, mean_cant_true_fea_selec_LASSO1se, mean_cant_false_fea_selec_LASSO1se, mean_cant_true_fea_selec_LASSOFREC, mean_cant_false_fea_selec_LASSOFREC


    else:
        mean_cant_false_fea_selec_LASSOMIN = [cant / cantidad_sim for cant in
                                               cant_false_fea_selec_LASSOMIN]
        mean_cant_true_fea_selec_LASSOMIN = [cant / cantidad_sim for cant in cant_true_fea_selec_LASSOMIN]

        #mean_cant_false_fea_selec_LASSO1se = [cant / cantidad_sim for cant in
                                             # cant_false_fea_selec_LASSO1se]
        #mean_cant_true_fea_selec_LASSO1se = [cant / cantidad_sim for cant in cant_true_fea_selec_LASSO1se]

        mean_cant_false_fea_selec_LASSOFREC = [cant / cantidad_sim for cant in
                                                           cant_false_fea_selec_LASSOFREC]
        mean_cant_true_fea_selec_LASSOFREC = [cant / cantidad_sim for cant in cant_true_fea_selec_LASSOFREC]


        return mean_cant_true_fea_selec_LASSOMIN,mean_cant_false_fea_selec_LASSOMIN, mean_cant_true_fea_selec_LASSOFREC,mean_cant_false_fea_selec_LASSOFREC


def grafico_monte_carlo(scenario, n_list,p,s,rho=0.9,cantidad_sim = 1, tau = 0.8,showfig=True, savefig = False, save_in=None):
    '''

    :param scenario: el escenario a simular, solo admite el scenario 1 y 3.
    :param n_list: los tamaños de los datos
    :param p: la cantidad de variables o features
    :param s: son las variables verdaderas involucras, de 1 a s.
    :param tau: selecciono las variables con frecuencia mayor a tau
    :return: el grafico de barra para cada n de la lista, con las variables verdaderas y las falasas que selecciono
    devuelvo un plot para LASSO.MIN y otro para LASSO.FREC
    '''

    mean_cant_true_fea_selec_LASSOMIN = []
    mean_cant_false_fea_selec_LASSOMIN = []
    mean_cant_true_fea_selec_LASSO1se = []
    mean_cant_false_fea_selec_LASSO1se = []
    mean_cant_true_fea_selec_LASSOFREC = []
    mean_cant_false_fea_selec_LASSOFREC = []
    for n in n_list:
        mean_true_LASSOMIN, mean_false_LASSOMIN,mean_true_LASSO1se, mean_false_LASSO1se,mean_true_LASSOFREC,mean_false_LASSOFREC =  monte_carlo_LASSOMIN_VS_LASSOFREC(scenario,n,p,s, cantidad_sim= cantidad_sim,rho=rho,tau=tau)
        print(mean_true_LASSOMIN, mean_false_LASSOMIN,mean_true_LASSO1se, mean_false_LASSO1se,mean_true_LASSOFREC,mean_false_LASSOFREC)
        mean_cant_true_fea_selec_LASSOMIN.append(mean_true_LASSOMIN)
        mean_cant_false_fea_selec_LASSOMIN.append(mean_false_LASSOMIN)
        mean_cant_true_fea_selec_LASSO1se.append(mean_true_LASSO1se)
        mean_cant_false_fea_selec_LASSO1se.append(mean_false_LASSO1se)
        mean_cant_true_fea_selec_LASSOFREC.append(mean_true_LASSOFREC)
        mean_cant_false_fea_selec_LASSOFREC.append(mean_true_LASSOFREC)

    n_list_string = []
    for n in n_list:
        text = 'n=%s' % (str(n))
        n_list_string.append(text)

    fig, ax = plt.subplots()
    ax.bar(n_list_string, mean_true_LASSOMIN, width=1, edgecolor="white", linewidth=0.7)
    ax.bar(n_list_string, mean_false_LASSOMIN, bottom=mean_true_LASSOMIN,
           width=1, edgecolor="white", linewidth=0.7)
    x_label = 'sample size (n)'
    plt.xlabel(x_label)
    y_label = 'true and false selected variables'
    plt.ylabel(y_label)
    text = r'LASSO.MIN'
    text_ = r'%s p=%s, s=%s, $\rho=$ %s' % (text, p, s,rho)
    plt.title(text_)

    if savefig:
        filename = save_in+'\SCE%s_LASSOMIN_s%s_rho%s_cant_sim%s.png' % (scenario, s, rho, cantidad_sim)
        plt.savefig(fname=filename)
    if showfig:
        plt.show()

    fig, ax = plt.subplots()
    ax.bar(n_list_string, mean_true_LASSO1se, width=1, edgecolor="white", linewidth=0.7)
    ax.bar(n_list_string, mean_false_LASSO1se, bottom=mean_true_LASSO1se,
           width=1, edgecolor="white", linewidth=0.7)
    x_label = 'sample size (n)'
    plt.xlabel(x_label)
    y_label = 'true and false selected variables'
    plt.ylabel(y_label)
    text = r'LASSO.1se'
    text_ = r'%s p=%s, s=%s, $\rho=$ %s' % (text, p, s, rho)
    plt.title(text_)

    if savefig:
        filename = save_in + '\SCE%s_LASSO1se_s%s_rho%s_cant_sim%s.png' % (scenario, s, rho, cantidad_sim)
        plt.savefig(fname=filename)
    if showfig:
        plt.show()
    fig, ax = plt.subplots()
    ax.bar(n_list_string, mean_true_LASSOFREC, width=1, edgecolor="white", linewidth=0.7)
    ax.bar(n_list_string, mean_false_LASSOFREC, bottom=mean_true_LASSOFREC,
           width=1, edgecolor="white", linewidth=0.7)
    x_label = 'sample size (n)'
    plt.xlabel(x_label)
    y_label = 'true and false selected variables'
    plt.ylabel(y_label)
    text = r'LASSO.FREC $\tau = $ %s' %(tau)
    text_ = r'%s p=%s, s=%s , $\rho = $ %s' % (text, p, s, rho)
    plt.title(text_)
    if savefig:
        filename = save_in+'\SCE%s_LASSOFREC%s_s_s%s_rho%s_cant_sim%s.png' % (scenario,tau, s,rho, cantidad_sim)
        plt.savefig(fname=filename)
    if showfig:
        plt.show()


def grafico_monte_carlo_por_cluster(scenario,cant_clusters, n, p, s, rho=0.9, cantidad_sim=1, tau=0.8,
                                    showfig=True, savefig=False, save_in=None):

    mean_true_LASSOMIN, mean_false_LASSOMIN, mean_true_LASSOFREC, mean_false_LASSOFREC = monte_carlo_LASSOMIN_VS_LASSOFREC(
            scenario, n, p, s, cantidad_sim=cantidad_sim, rho=rho, tau=tau)

    x = np.arange(cant_clusters)
    fig, ax = plt.subplots()
    ax.bar(x, mean_true_LASSOFREC, width=1, edgecolor="white", linewidth=0.7)
    ax.bar(x, mean_false_LASSOFREC, bottom=mean_true_LASSOFREC,
           width=1, edgecolor="white", linewidth=0.7)
    x_label = 'mod %s' % (str(cant_clusters))
    plt.xlabel(x_label)
    y_label = 'true and false selected variables'
    plt.ylabel(y_label)
    text = r'LASSO.FREC $\tau=$ %s' % (tau)
    text_ = '%s p=%s, n=%s, s=%s, rho=%s' % (text, p, n, s, rho)
    plt.title(text_)

    if savefig:
        filename =save_in+ '\SC%s_LASSO_FREC_%sn=%s_p=%s_s=%s_rho=%s_cantsim=%s.png' % (scenario,tau, n, p, s, rho, cantidad_sim)
        plt.savefig(fname=filename)
    if showfig:
        plt.show()

    fig, ax = plt.subplots()
    ax.bar(x, mean_true_LASSOMIN, width=1, edgecolor="white", linewidth=0.7)
    ax.bar(x, mean_false_LASSOMIN, bottom=mean_true_LASSOMIN,
           width=1, edgecolor="white", linewidth=0.7)
    x_label = 'mod %s' % (str(cant_clusters))
    plt.xlabel(x_label)
    y_label = 'true and false selected variables'
    plt.ylabel(y_label)
    text = r'LASSO.MIN '
    text_ = '%s p=%s, n=%s, s=%s, rho=%s' % (text, p, n, s, rho)
    plt.title(text_)

    if savefig:
        filename = save_in + '\SC%s_LASSO_MIN_n=%s_p=%s_s=%s_rho=%s_cantsim=%s.png' % (
        scenario, n, p, s, rho, cantidad_sim)
        plt.savefig(fname=filename)
    if showfig:
        plt.show()



# scenario = '1'
# n_list = [100,200,400]
# p = 50
# rho_list = [None]
# #tau = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# tau_for_MC_sim = [0.8]
# cant_sim = 1
# #save_in = r'C:\Vero\ML\codigos_Python\Figuras_paper\Frecuencias_ordenadas\SCENARIO%s' %(scenario)
# save_in = r'C:\Vero\ML\codigos_Python\Figuras_paper\Tau_fijo\SCENARIO%s' %(scenario)
# for rho in rho_list:
#     for tau in tau_for_MC_sim:
#         grafico_monte_carlo(scenario, n_list,p,s= 10,rho=rho,cantidad_sim = cant_sim, tau = tau,showfig= True,
#                             savefig=False,save_in = save_in)


# scenario = '1'
# n= 400
# p = 50
# s = 10
# rho_list = [None]
# tau_list = [0.8]
# cant_sim = 10
# selection='cyclic'
# save_in = r'C:\Vero\ML\codigos_Python\Figuras_paper\Tau_fijo\SCENARIO%s' %(scenario)
# cant_clusters=10
# for rho in rho_list:
#     for tau in tau_list:
#         #grafico_monte_carlo_por_cluster(scenario, cant_clusters, n, p, s, rho=rho, cantidad_sim=cant_sim, tau=tau,
#                                       # showfig=True, savefig=False, save_in=save_in)
#         grafico_frecuencias_ordenadas(scenario, n ,p, s, tau_list=tau_list, rho = rho, showfig=True, savefig=False,
#                                            save_in = None, cant_simu = cant_sim,selection=selection, weight = False)
# #
