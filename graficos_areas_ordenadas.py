from LASSO_areas import grafico_areas_ordenadas


scenario = '2'
n_list = [100,200,400]
p = 50
s = 10
rho_list = [0.2,0.5,0.9]
cant_sim = 1
cant_clusters = 10
selection='cyclic'
save_in = r'C:\Vero\ML\codigos_Python\Figuras_paper\Areas\SCENARIO%s' %(scenario)
for n in n_list:
    for rho in rho_list:
        grafico_areas_ordenadas(scenario, n ,p, s, rho = rho, showfig=True, savefig=False,
                                           save_in = save_in, cant_simu = cant_sim,selection=selection)
