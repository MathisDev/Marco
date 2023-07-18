#!/usr/bin/env python
# coding: utf-8
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd

# ### Param's :

np.random.seed(42)
## Number of assets
n_assets = 5
## Number of observations
n_obs = 1000

return_vec = np.random.randn(n_assets, n_obs) #Randomly generates returns on each asset, for each observations (like a timeseries)

# ### Functions : 

#Produces n random weights that sum to 1
def rand_weights(n):
    
    k = np.random.rand(n)
    return k / sum(k)


#Returns a random portfolio defined by the mean and standard deviation of returns  
def random_portfolio(returns):    

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))
    
    mu = w * p.T #mean returns of portfolio
    sigma = np.sqrt(w * C * w.T) #std returns of portfolio
    
    return mu, sigma

#Generate random portfolios, defined by their mean and std of returns, using the previous function
def generate_portfolios(n_portfolios=500):
    
    means, stds = np.column_stack([random_portfolio(return_vec) for _ in range(n_portfolios)])
    
    return means, stds

#Plot the randomly generated portfolios
def plot_portfolio(n_portfolios=500):
    means, stds = generate_portfolios(n_portfolios)
    fig = plt.figure()
    plt.plot(stds, means, 'o', markersize=5)
    plt.title("Représentation graphiques des portefeuilles du marché")
    plt.ylabel('Rendements')
    plt.xlabel('Risques')
    plt.savefig("Ptfs-Marchés.png",dpi=144)

#Compute the optimal portfolio, based on Markowitz theory - Source : https://plotly.com/python/v3/ipython-notebooks/markowitz-portfolio-optimization/
def optimal_portfolio(returns):
    
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)] #Used to avoid linear mean
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus]
    
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    
    
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks

#Main program :

plot_portfolio(1000)
weights, returns, risks = optimal_portfolio(return_vec)

fig = plt.figure()
plt.plot(stds, means, 'o')
plt.plot(risks, returns, 'y-o')

plt.title("Frontière de l'efficience")
plt.ylabel('Rendements')
plt.xlabel('Risques')
plt.savefig("Frotnière-efficience.png",dpi=144)

print (weights)
print(np.sum(weights)) #100% is invested in n_assets (5 assets in this example)
