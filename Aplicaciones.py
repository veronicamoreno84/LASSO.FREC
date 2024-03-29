import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold




def best_lasso(X,y,fit_intercept=False):
    n = len(y)
    eps = 0.001
    lambda_max = np.linalg.norm(np.matmul(np.transpose(X),y) , np.inf)/n
    lambda_min = eps*lambda_max
    start = np.log10(lambda_min)
    end = np.log10(lambda_max)
    K=100
    lambdas = np.logspace(start,end,K)
    lasso = Lasso(random_state=0, max_iter=100000,fit_intercept=fit_intercept)
    tuned_parameters = [{'alpha': lambdas}]
    n_folds = 5
    clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, scoring='neg_mean_squared_error')
    clf.fit(X,y)
    best_lambda = clf.best_params_['alpha']
    best_score  = clf.cv_results_['mean_test_score'][clf.best_index_]
    best_std = clf.cv_results_['std_test_score'][clf.best_index_]
    return best_lambda, best_score, best_std


def frecuencias_lasso(X,y,fit_intercept=False):
    n,p = X.shape
    eps = 0.001
    lambda_max = np.linalg.norm(np.matmul(np.transpose(X),y) , np.inf)/n
    lambda_min = eps*lambda_max
    start = np.log10(lambda_min)
    end = np.log10(lambda_max)
    K=100
    lambdas = np.logspace(start,end,K)
    cant_veces_que_elijo_por_feature = p* [0] #
    for lambda_ in lambdas:
       clf = Lasso(alpha=lambda_, fit_intercept=fit_intercept)
       clf.fit(X, y)
       coeficientes = clf.coef_
       selected_coef=[1 if abs(coeficientes[i])>0.00001 else 0 for i in range(p)]
       for i in range(p):
          cant_veces_que_elijo_por_feature[i]=cant_veces_que_elijo_por_feature[i]+selected_coef[i]

    frecuencias = [cant/len(lambdas) for cant in cant_veces_que_elijo_por_feature]


    return frecuencias

def grafico_frecuencias_ordenadas(X,y,name=False,savefig=False,showfig=True,save_in = None):
    rb = RobustScaler()
    X = rb.fit_transform(X)
    yc = y - y.mean()

    n, p = X.shape

    frecuencias_ = frecuencias_lasso(X, yc, fit_intercept=False)
    indices_ordenan_frecuencias = np.argsort(frecuencias_).tolist()
    indices_ordenan_frecuencias.reverse()
    frecuencias_reversed= [frecuencias_[i] for i in indices_ordenan_frecuencias]
    fig, ax = plt.subplots()
    tau_list=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for tau in tau_list:
        plt.axhline(y=tau, color='black', linestyle='--',
                    linewidth=1, label='bla')
    plt.plot([i for i in range(p)], frecuencias_reversed, 'co', color='blue', )
    title = r'Ordered frequencies '
    if name:
        title+=name

    plt.title(title)
    if savefig:
        filename = '\ %s.png' % (name)
        if save_in is not None:
            filename = save_in + filename
        plt.savefig(fname=filename)
    if showfig:
        plt.show()

    return indices_ordenan_frecuencias

def results(X,y,name,showfig,save_in = None,thresholds=[], cv = None):
    rb = RobustScaler()
    X = rb.fit_transform(X)
    yc = y - y.mean()
    n, p = X.shape



    if cv is None:

        metodo = []
        cant_selec = []
        mse_test = []

        X_train, X_test, y_train, y_test = train_test_split(X, yc, test_size=0.2)

        ''' LASSO.MIN'''

        best_lambda, best_score, std_best_lmabda = best_lasso(X_train,y_train,fit_intercept=False)
        clf = Lasso(alpha=best_lambda, fit_intercept=False)
        clf.fit(X_train, y_train)
        coeficientes = clf.coef_
        selected_features_LASSO_MIN = [i for i in range(p) if abs(coeficientes[i]) > 0.00001]
        X_sel = [[X_train[i][j] for j in selected_features_LASSO_MIN] for i in range(len(y_train))]
        reg = LinearRegression(fit_intercept=False).fit(X_sel, y_train)
        X_test_sel = [[X_test[i][j] for j in selected_features_LASSO_MIN] for i in range(len(y_test))]
        y_pred = reg.predict(X_test_sel)
        mse = mean_squared_error(y_test, y_pred)
        cant_fea_selected = len(selected_features_LASSO_MIN)

        metodo.append('LASSO.MIN')
        cant_selec.append(cant_fea_selected)
        mse_test.append(mse)


        '''LASSO.FREC'''

        frecuencias_ = frecuencias_lasso(X_train, y_train, fit_intercept=False)
        for thr in thresholds:
            selected_features_LASSO_FREC = [i for i in range(p) if frecuencias_[i] > thr]
            X_sel = [[X_train[i][j] for j in selected_features_LASSO_FREC] for i in range(len(y_train))]
            reg = LinearRegression(fit_intercept=False).fit(X_sel, y_train)
            X_test_sel = [[X_test[i][j] for j in selected_features_LASSO_FREC] for i in range(len(y_test))]
            y_pred = reg.predict(X_test_sel)
            mse = mean_squared_error(y_test, y_pred)
            cant_fea_sel = len(selected_features_LASSO_FREC)
            print(mse)

            met = 'LASSO.FREC tau= %s' % (str(thr))
            metodo.append(met)
            cant_selec.append(cant_fea_sel)
            mse_test.append(mse)

        df = pd.DataFrame(list(zip(metodo, cant_selec, mse_test)), columns=['Method', 'Number of selected variables', 'MSE test'])
        print(df)

        fig, ax = plt.subplots()

        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')


        ax.table(cellText=df.values, colLabels=df.columns, loc='center')

        fig.tight_layout()

    else:

        metodo = []
        cant_selec_mean = []
        cant_selec_std = []
        mse_test_mean = []
        mse_test_std = []

        result_cv = {'LASSO.MIN': [[],[]] , 'LASSO.1std': [[],[]]  } # en el primer lugar guardo las variables seleccionadas y en el segundo el mse_test

        for thr in thresholds:
            key = 'LASSO.FREC tau= %s' % (str(thr))
            result_cv[key]=[[],[]]

        kf = KFold(n_splits=cv)

        for train, test in kf.split(X):
            X_train, X_test, y_train, y_test = X[train], X[test], yc[train], yc[test]

            ''' LASSO.MIN'''

            best_lambda, best_score, best_std = best_lasso(X_train, y_train, fit_intercept=False)
            clf = Lasso(alpha=best_lambda, fit_intercept=False)
            clf.fit(X_train, y_train)
            coeficientes = clf.coef_
            selected_features_LASSO_MIN = [i for i in range(p) if abs(coeficientes[i]) > 0.00001]
            X_sel = [[X_train[i][j] for j in selected_features_LASSO_MIN] for i in range(len(y_train))]
            reg = LinearRegression(fit_intercept=False).fit(X_sel, y_train)
            X_test_sel = [[X_test[i][j] for j in selected_features_LASSO_MIN] for i in range(len(y_test))]
            y_pred = reg.predict(X_test_sel)

            metodo.append('LASSO.MIN')
            result_cv['LASSO.MIN'][1].append(mean_squared_error(y_test, y_pred))
            result_cv['LASSO.MIN'][0].append(len(selected_features_LASSO_MIN))

            lambda_1std = best_lambda + np.log(best_std)

            clf = Lasso(alpha=lambda_1std, fit_intercept=False)
            clf.fit(X_train, y_train)
            coeficientes = clf.coef_
            selected_features_LASSO_1std = [i for i in range(p) if abs(coeficientes[i]) > 0.00001]
            print('selected LASSO.1std', len(selected_features_LASSO_1std))
            X_sel = [[X_train[i][j] for j in selected_features_LASSO_1std] for i in range(len(y_train))]

            reg = LinearRegression(fit_intercept=False).fit(X_sel, y_train)
            X_test_sel = [[X_test[i][j] for j in selected_features_LASSO_1std] for i in range(len(y_test))]
            y_pred = reg.predict(X_test_sel)

            metodo.append('LASSO.1std')
            result_cv['LASSO.1std'][1].append(mean_squared_error(y_test, y_pred))
            result_cv['LASSO.1std'][0].append(len(selected_features_LASSO_1std))

            '''LASSO.FREC'''

            frecuencias_ = frecuencias_lasso(X_train, y_train, fit_intercept=False)
            for thr in thresholds:
                selected_features_LASSO_FREC = [i for i in range(p) if frecuencias_[i] > thr]
                X_sel = [[X_train[i][j] for j in selected_features_LASSO_FREC] for i in range(len(y_train))]
                reg = LinearRegression(fit_intercept=False).fit(X_sel, y_train)
                X_test_sel = [[X_test[i][j] for j in selected_features_LASSO_FREC] for i in range(len(y_test))]
                y_pred = reg.predict(X_test_sel)
                mse = mean_squared_error(y_test, y_pred)
                cant_fea_sel = len(selected_features_LASSO_FREC)


                met = 'LASSO.FREC tau= %s' % (str(thr))
                metodo.append(met)
                result_cv[met][1].append(mse)
                result_cv[met][0].append(cant_fea_sel)


        cant_selec_mean.append(round(np.mean(result_cv['LASSO.MIN'][0]),4))
        cant_selec_std.append(round(np.std(result_cv['LASSO.MIN'][0]),4))
        mse_test_mean.append(round(np.mean(result_cv['LASSO.MIN'][1]),4))
        mse_test_std.append(round(np.std(result_cv['LASSO.MIN'][1]),4))

        for thr in thresholds:
            met = 'LASSO.FREC tau= %s' % (str(thr))
            cant_selec_mean.append(round(np.mean(result_cv[met][0]),4))
            cant_selec_std.append(round(np.std(result_cv[met][0]),4))
            mse_test_mean.append(round(np.mean(result_cv[met][1]),4))
            mse_test_std.append(round(np.std(result_cv[met][1]),4))

        df = pd.DataFrame(list(zip(metodo, mse_test_mean,mse_test_std)),
                          columns=['Method', 'Mean MSE test', 'Std MSE test'])
        print(df)

    if save_in is not None:
        name_save = r'\%s.png' %name
        filename = save_in + name_save
        plt.savefig(fname=filename)
        name_csv = r'\%sResults.csv' %name
        filename = save_in + name_csv
        df.to_csv(filename,sep=';')


    if showfig:
        plt.show()

    return df





''''SEGURIDAD VIAL'''

# seg_vial = pd.read_csv('SegVial.csv',delimiter=',')
# print(seg_vial)
# features_names = ['DenPob ', 'ArCiclista', 'ArBajaVel', 'PMPeatones', 'PMCiclistas', 'PMTPublico', 'PMVMotor', 'Temp', 'Precipitacion', 'PBI']
# print(len(features_names))
# X_df = seg_vial[features_names]
# print(X_df)
# # corr_matrix = X_df.corr()
# # print(corr_matrix)
# # sn.clustermap(corr_matrix, annot=True)
# # plt.show()
# # Z = cluster.hierarchy.linkage(corr_matrix, method='single', metric='euclidean', optimal_ordering=False)
# # print(Z)
# X = X_df.values.tolist()
# print(X)
#
# seg_vial['AutoAuto_estan']= (10**6)*seg_vial['AutoAuto']/seg_vial['Poblacion']
# y_AutoAuto = np.array(seg_vial['AutoAuto_estan'].tolist())
# #print(y_AutoAuto)
#
# save_in = 'C:\Vero\ML\codigos_Python\Results_Aplicaciones'
# grafico_frecuencias_ordenadas(X,y_AutoAuto,name='SegVial',savefig=True,save_in = save_in)
#
# save_in = 'C:\Vero\ML\codigos_Python\Results_Aplicaciones'
#
# results = results(X,y_AutoAuto,'SegVial',thresholds=[0.4,0.5,0.6,0.7,0.8],save_in=save_in, cv=5)
# print(results)


'''DIABETES DATA SET'''

# diabetes = datasets.load_diabetes()
# X = diabetes.data[:]
# y=[ x for x in diabetes.target[:]]


'''STUDENTs MAT'''

datos_estudiantes_mat = pd.read_csv('student-mat.csv',delimiter = ';')
feature_names = datos_estudiantes_mat.columns[:30]
target_1_name = datos_estudiantes_mat.columns[30]
target_2_name = datos_estudiantes_mat.columns[31]
target_3_name = datos_estudiantes_mat.columns[32]

X_Df = datos_estudiantes_mat[feature_names]
X_Df_encod = pd.get_dummies(X_Df)
features = X_Df_encod.columns
X=X_Df_encod.values.tolist()


#y_1 = np.array(datos_estudiantes_mat[target_1_name].tolist())
#y_2 = datos_estudiantes_mat[target_2_name].tolist()
y_3 = np.array(datos_estudiantes_mat[target_3_name].tolist())
y=y_3

# rb = RobustScaler()
# X = rb.fit_transform(X)
# yc = y - y.mean()
# n = len(y)
# eps = 0.001
# lambda_max = np.linalg.norm(np.matmul(np.transpose(X),y) , np.inf)/n
# lambda_min = eps*lambda_max
# start = np.log10(lambda_min)
# end = np.log10(lambda_max)
# K=100
# lambdas = np.logspace(start,end,K)
#
# lasso = Lasso(random_state=0, max_iter=100000,fit_intercept=False)
# tuned_parameters = [{'alpha': lambdas}]
# n_folds = 5
# clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, scoring='neg_mean_squared_error')
# clf.fit(X,y)

name= 'StudentsMat3'

# #save_in = r'C:\Users\vmoreno\LASSO.FREC'
# save_in = 'C:\Vero\ML\codigos_Python\Results_Aplicaciones'
# indices_ordenan_frecuencias=grafico_frecuencias_ordenadas(X,y_3,name=name,savefig=True,save_in = save_in)
# print('tamanio muestra encondeada:' , len(indices_ordenan_frecuencias))
# for i in indices_ordenan_frecuencias:
#     print(features[i])


#save_in = 'C:\Vero\ML\codigos_Python\Results_Aplicaciones'
save_in = r'C:\Users\vmoreno\LASSO.FREC'
n = len(y)

results = results(X,y_3,name=name,thresholds=[0.4,0.5,0.6,0.7,0.8], showfig=False,save_in=save_in,cv= 5)




