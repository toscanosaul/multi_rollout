import sys
sys.path.append("..")
import numpy as np
from math import *
from matplotlib import pyplot as plt
import scipy.stats as stats
from scipy.stats import norm,poisson
import statsmodels.api as sm
import multiprocessing as mp
import os
from scipy.stats import poisson
import json
#from BGO.Source import *
import time
import random
from sklearn import datasets, linear_model
import itertools

randomSeed=int(sys.argv[1])
np.random.seed(randomSeed)

theta=np.loadtxt("coefficients.txt")
covariates = np.loadtxt("mymatrix.txt")

number_routes = 10
K = 5
sigma = 53.7
training = 5

cov_ind = np.array([17,31,38,5,6])

cov = covariates[np.random.choice(100, size=number_routes + training, replace=False),:]

#cov_ind = np.random.choice(len(theta), size=K, replace=False)
cov = cov[:, cov_ind]
theta = theta[cov_ind]


var_demand = sigma**2
prior_routes = np.random.choice(number_routes + training , size=training, replace=False)

prior_data = np.zeros(training)
index=0
for i in prior_routes:
    prior_data[index] = np.random.normal(np.dot(cov[i,:], theta),np.sqrt(var_demand))
    index += 1

covariates = cov

Z = np.transpose(cov[prior_routes,:])
sigma_inv_prior = np.dot(Z, np.transpose(Z)) / (var_demand)

sigma_prior = np.linalg.inv(sigma_inv_prior)
mu_prior = np.dot(np.dot(np.linalg.inv(np.dot(Z, np.transpose(Z))), Z), prior_data)

routes_candidates = np.delete(np.arange(number_routes + training),prior_routes)
cov_candidates = cov[routes_candidates, :]

number_new_routes = 5

def f_i(x,w):
    number_chosen = np.sum(w)
    number_previous_chosen = np.sum(x)
    matrix_mu_1 = np.zeros((number_chosen, number_previous_chosen))
    
    Z_tmp = np.zeros((number_previous_chosen, K))
    ind = 0
    ind_row=-1
    ind_col=-1
    for j in range(number_routes):
        ind_col=-1
        if w[j] == 1:
            ind_row +=1          
        if x[j] == 1:
            Z_tmp[ind, :] = cov_candidates[j, :]
            ind+=1
        for i in range(number_routes):
            if w[j] == 1 and x[i] == 1:
                ind_col += 1
                tmp = cov_candidates[j,:]
                tmp_2 = cov_candidates[i,:]
                matrix_mu_1[ind_row, ind_col] = np.dot( np.dot(tmp, sigma_prior), tmp_2)
    
    gamma_0 = np.dot(np.dot (Z_tmp, sigma_prior), np.transpose(Z_tmp))
    gamma_0 += np.diag(var_demand*np.ones(gamma_0.shape[0]))
    gamma_0_inverse = np.linalg.inv(gamma_0)
    first_product = np.dot(matrix_mu_1, gamma_0_inverse)
    
    sum_elements_matrix = first_product.sum(axis=0)
    
    
    
 
    matrix_variance = np.dot(np.dot(sum_elements_matrix,gamma_0),sum_elements_matrix)
    return np.sqrt(matrix_variance)

def e_i(x):
    result = 0
    for i in range(number_routes):
        if x[i] == 1:
            result += np.dot(cov_candidates[i,:],mu_prior)
    return result

def hvoi (b,c,keep):
    M=len(keep)
    if M>1:
        c=c[keep+1]
        c2=-np.abs(c[0:M-1])
        tmp=norm.pdf(c2)+c2*norm.cdf(c2) 
        return np.sum(np.diff(b[keep])*tmp)
    else:
        return 0
    
from AffineBreakPoints import *

possiblePoints2 = list(itertools.product([0, 1], repeat=number_routes))
indexesPoints=range(len(possiblePoints2))

possiblePoints3=[(i,possiblePoints2[i]) for i in indexesPoints if np.sum(possiblePoints2[i])<=number_new_routes]
possiblePoints=[(i,j) for (i,j) in possiblePoints3 if 0 < np.sum(j) < number_new_routes]
indexesPoints=range(len(possiblePoints))
possiblePoints=[(i,possiblePoints[i][1]) for i in indexesPoints  ]

aVec={}

i=0

for i,j in possiblePoints3:
    aVec[i] = e_i(j)
    
def VOI(x,a2=aVec,grad=False):
    lsum=np.sum(x)

    x_chosen = np.zeros(lsum)
    ind = 0
    for i in range(len(x)):
        if x[i] == 1:
            x_chosen[ind] = i
            ind += 1
    x_chosen=x_chosen.astype(np.int64)
    
    new_indexes=[(i,j) for (i,j) in possiblePoints3 if np.sum(np.array(j))==number_new_routes and all((np.array(j))[x_chosen]==np.ones(lsum))]
  
    f = np.zeros(len(new_indexes))
    e = np.zeros(len(new_indexes))
    
    ind=0
    for i,j in new_indexes:
        f[ind] = f_i(x,j)
        e[ind] = a2[i]
        ind += 1
    

    a,b,keep=AffineBreakPointsPrep(e,f)
    
    keep1,c=AffineBreakPoints(a,b)
    keep1=keep1.astype(np.int64)
    M=len(keep1)
    keep2=keep[keep1]
    
    return hvoi(b,c,keep1)

VOIval=np.zeros(len(possiblePoints))

Npoint=len(possiblePoints)
for i in range(Npoint):
    VOIval[i]=VOI(possiblePoints[i][1])
    
print "first"
#print possiblePoints[np.argmax(VOIval)
oldPoint=possiblePoints[np.argmax(VOIval)][1]

def f(x):
    n_routes = np.sum(x)
    observations = np.zeros(len(x))
    
    covariates_x = np.zeros((n_routes,K))
    ind = 0
    for i in range(len(x)):
        if x[i] == 1:
            covariates_x[ind,:] = cov_candidates[i, :]
            ind += 1
    mu_sample = np.dot(covariates_x, theta)
    
    samples = np.zeros(n_routes)
    
    for i in range(int(n_routes)):
        samples[i] = np.random.normal(mu_sample[i], np.sqrt(var_demand), 1)
    
    return samples

def newY(x, number_samples = 1):
    cont = 0
    for i in range(number_samples):
        cont += f(x)
    cont = cont / number_samples
    
    return cont,var_demand/number_samples

old_obs = newY(oldPoint)

def mu_1(w, x, old_obs):
    
    var_obs = old_obs[1]
    old_obs = old_obs[0]
    
    new_routes = np.sum(w)
    old_routes = np.sum(x)
    
    covariates_w = np.zeros((new_routes,K))
    covariates_x = np.zeros((old_routes,K))
    ind = 0
    ind_x = 0
    for i in range(len(x)):
        if w[i] == 1:
            covariates_w[ind,:] = cov_candidates[i, :]
            ind += 1
        if x[i] == 1:
            covariates_x[ind_x,:] = cov_candidates[i, :]
            ind_x += 1
    
    mu_0 = np.dot(covariates_w, mu_prior)
    
    mu_0_x = np.dot(covariates_x, mu_prior)
    gamma_0 = np.dot(np.dot(covariates_x, sigma_prior), np.transpose(covariates_x))
    gamma_0 = gamma_0 + np.diag(var_obs*np.ones(gamma_0.shape[0]))
    gamma_0_inverse = np.linalg.inv(gamma_0)
    
    combination_x_w = np.zeros((new_routes, old_routes))
    
    for i in range(new_routes):
        for j in range(old_routes):
            combination_x_w[i,j] = np.dot(np.dot(covariates_w[i,:], sigma_prior),covariates_x[j,:])
    
    
    observations = old_obs - mu_0_x
    observations = observations.reshape((len(old_obs),1))
    
    temp = np.dot(np.dot(combination_x_w, gamma_0_inverse), observations)
    
    mu_1 = mu_0 + temp.reshape(temp.shape[0])
    
    
    return np.sum(mu_1)
    
        
lsum=np.sum(oldPoint)

x_chosen = np.zeros(lsum)
ind = 0
for i in range(len(oldPoint)):
    if oldPoint[i] == 1:
        x_chosen[ind] = i
        ind += 1
x_chosen=x_chosen.astype(np.int64)

new_indexes=[(i,j) for (i,j) in possiblePoints3 if np.sum(np.array(j))==number_new_routes and all((np.array(j))[x_chosen]==np.ones(lsum))]

vector_mu_1 = np.zeros(len(new_indexes))

for i in range(len(vector_mu_1)):
    vector_mu_1[i] = mu_1(new_indexes[i][1] ,oldPoint ,old_obs)
    
new_sol = new_indexes[np.argmax(vector_mu_1)]


#####all stations in one step

def mu_0(x):    
    old_routes = np.sum(x)
    
    covariates_x = np.zeros((old_routes,K))
    ind_x = 0
    
    for i in range(len(x)):
        if x[i] == 1:
            covariates_x[ind_x,:] = cov_candidates[i, :]
            ind_x += 1
    mean_matrix = np.dot(covariates_x, mu_prior)
    
    return np.sum(mean_matrix)
    
possiblePointsAn2=[(i,j) for (i,j) in possiblePoints3 if np.sum(j)==number_new_routes]

mu_vec = np.zeros(len(possiblePointsAn2))

for i in range(len(possiblePointsAn2)):
    mu_vec[i] = mu_0(possiblePointsAn2[i][1])
    
point_chosen = possiblePointsAn2[np.argmax(mu_vec)][1]

def actual_demand(x):
    n_routes = np.sum(x)
    observations = np.zeros(len(x))
    
    covariates_x = np.zeros((n_routes,K))
    ind = 0
    for i in range(len(x)):
        if x[i] == 1:
            covariates_x[ind,:] = cov_candidates[i, :]
            ind += 1

    mu_sample = np.dot(covariates_x, theta)
    
    return np.sum(mu_sample)

one_stage = actual_demand(point_chosen)

two_stages = actual_demand(new_indexes[np.argmax(vector_mu_1)][1])

path = "ResultsOpening10_informs"

if not os.path.exists(path):
    os.makedirs(path)
    
f=open(os.path.join(path, '%d'%randomSeed+"results.txt"),'w')
f.close()

with open(os.path.join(path,'%d'%randomSeed+"results.txt"),"a") as f:
    np.savetxt(f, np.array(point_chosen))
    np.savetxt(f, np.array(one_stage).reshape(1))
    
    np.savetxt(f,np.array(new_sol))
    np.savetxt(f,np.array(oldPoint))
    np.savetxt(f, np.array(two_stages).reshape(1))