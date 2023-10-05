import numpy as np

import numpy as np


def scenario_1_random_coef(n, p, s, sigma2,coef):
 mean = p * [0]
 cov = np.eye(p)
 X = np.random.multivariate_normal(mean, cov, n)

 error = np.random.normal(0, sigma2, n)
 y = [sum([coef[j] * X[i][j] for j in range(s)]) for i in range(n)] + error

 return X, y

def scenario_1(n, p, s, sigma2):
 mean = p * [0]
 cov = np.eye(p)
 X = np.random.multivariate_normal(mean, cov, n)

 error = np.random.normal(0, sigma2, n)
 y = [sum([(1.25) * X[i][j] for j in range(s)]) for i in range(n)] + error

 return X, y


def scenario_2(n, p, s, rho, sigma2, cant_clusters):
 '''
 :param rho: sigma_{jk}=rho , si mod_M(j)=mod_M(k) donde M es la cantidad de clusters
 :param n: data size
 :param p: features
 :param s: first s not null components in de regression (and equal 1)
 :return: X=N_n(0,Sigma), y=sum_{i=1}^s 1*X(i)+E, donde E=N_n(0,sigma^2I_n)
 '''
 mean = np.zeros(p)
 M = cant_clusters
 cov = np.eye(p)
 for i in range(p):
  for j in range(p):
   if i != j and i % M == j % M:
    cov[i, j] = rho
 X = np.random.multivariate_normal(mean, cov, n)

 error = np.random.normal(0, sigma2, n)
 y = [sum([X[i][j] for j in range(s)]) for i in range(n)] + error

 return X, y

def scenario_2_random_coef(n, p, s, rho, sigma2, cant_clusters,coef):
 '''
 :param rho: sigma_{jk}=rho , si mod_M(j)=mod_M(k) donde M es la cantidad de clusters
 :param n: data size
 :param p: features
 :param s: first s not null components in de regression (and equal 1)
 :return: X=N_n(0,Sigma), y=sum_{i=1}^s 1*X(i)+E, donde E=N_n(0,sigma^2I_n)
 '''
 mean = np.zeros(p)
 M = cant_clusters
 cov = np.eye(p)
 for i in range(p):
  for j in range(p):
   if i != j and i % M == j % M:
    cov[i, j] = rho
 X = np.random.multivariate_normal(mean, cov, n)

 error = np.random.normal(0, sigma2, n)
 y = [sum([coef[j]*X[i][j] for j in range(s)]) for i in range(n)] + error

 return X, y



def scenario_3(n,p,s,rho,sigma2):
 '''
 :param rho: sigma_{jk}=rho , si mod_M(j)=mod_M(k) donde M es la cantidad de clusters
 :param n: data size
 :param p: features
 :param s: first s not null components in de regression (and equal 1)
 :return: X=N_n(0,Sigma), y=sum_{i=1}^s 1*X(i)+E, donde E=N_n(0,sigma^2I_n)
 '''
 mean = p*[0]
 cov= [[rho**(abs(j-k)) for k in range(p)] for j in range(p)]
 X = np.random.multivariate_normal(mean, cov, n)

 error=np.random.normal(0,sigma2, n)
 y = [sum([(0.5)*X[i][j] for j in range(s)]) for i in range(n)]+error

 return X,y

def scenario_3_coef_random(n,p,s,rho,sigma2,coef):
 '''
 :param rho: sigma_{jk}=rho , si mod_M(j)=mod_M(k) donde M es la cantidad de clusters
 :param n: data size
 :param p: features
 :param s: first s not null components in de regression (and equal 1)
 :return: X=N_n(0,Sigma), y=sum_{i=1}^s 1*X(i)+E, donde E=N_n(0,sigma^2I_n)
 '''
 mean = p*[0]
 cov= [[rho**(abs(j-k)) for k in range(p)] for j in range(p)]
 X = np.random.multivariate_normal(mean, cov, n)

 error=np.random.normal(0,sigma2, n)
 y = [sum([coef[j]*X[i][j] for j in range(s)]) for i in range(n)]+error

 return X,y