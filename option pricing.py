# -*- coding: utf-8 -*-
"""
Created on Sun May 27 20:30:40 2018

@author: mai1346
"""

import numpy as np
from numba import jit
from math import log, sqrt, exp
from scipy import stats
#%%

def binomial(S0, Strike, T, sigma, rf, freq, option = 'call'):
    '''
    Valuation of options using binomial tree.
    Input:
        S0: initial underlying asset price.
        Strike: strike price of the option
        T: maturity date (in years)
        sigma: price volatility
        rf: risk-free rate
        freq: time steps before maturity, higher value would give more accurate
              answer
        option: type of the option. 'call' or 'put'
    Output:
        option value at time 0
    '''
    
    # Initialize parameters
    Delta_T = T / freq
    Growth_factor = np.exp(rf * Delta_T)
    Up = np.exp(sigma * np.sqrt(Delta_T))
    Down = 1 / Up
    p = (Growth_factor - Down)/(Up - Down)
    
    # Calculate the underlying price
    S = np.zeros((freq + 1,freq + 1))
    S[0, 0] = S0
    for i in range(1, freq + 1):
        for j in range(i + 1):
            S[i, j] = S[0, 0] * (Up ** j) * (Down ** (i - j))
    
    # Calculate the payoff of options
    try:      
        payoffs = np.zeros((freq + 1,freq + 1))
        if option == 'call':            
            payoffs[-1] = np.maximum(S[-1] - Strike, 0)
        elif option == 'put':             
            payoffs[-1] = np.maximum(Strike - S[-1], 0)
        
        # Discount to get the option value
        pv = np.zeros((freq + 1, freq + 1))
        pv[freq] = payoffs[freq]
        for i in range(freq - 1, -1, -1):
            for j in range(i+1):
                pv[i,j] = ((1-p) * pv[i+1,j] + p * pv[i+1,j+1]) / Growth_factor
    except:
        print ('The option type could only be "put" or "call", please recall the'
               'function with right arguments')    
    return pv[0,0]
#binomial(100,100,1,0.2,0.05,0,1000)
#%%
def mcs(paths, S0, Strike, T, sigma, rf, freq, option = 'call'):
    '''
    Valuation of options using Monte Carlo Simulation.
    Input:
        paths: paths generated simutaneously during the simulation
        S0: initial underlying asset price.
        Strike: strike price of the option
        T: maturity date (in years)
        sigma: price volatility
        rf: risk-free rate
        q: dividend yield
        freq: time steps before maturity, higher value would give more accurate
              answer
        option: type of the option. 'call' or 'put'
    Output:
        option value at time 0
    '''
    Delta_T = T / freq
    rand = np.random.standard_normal((freq + 1, paths))
    S = np.zeros((freq + 1, paths))
    S[0] = S0
    for t in range(1, freq + 1):
        S[t] = S[t - 1] * np.exp((rf - 0.5 * sigma ** 2) * Delta_T
         + sigma * np.sqrt(Delta_T) * rand[t])
    try:
        if option == 'call':            
            payoffs = np.maximum(S[-1] - Strike, 0)
        elif option == 'put':             
            payoffs = np.maximum(Strike - S[-1], 0)
        
        value = np.exp(-rf * T) * 1 / paths * np.sum(payoffs)
    except:
        print ('The option type could only be "put" or "call", please recall the'
               'function with right arguments')
    return value



def bsm(S0, Strike, T, sigma, rf, option = 'call'):
    '''
    Valuation of European call option in BSM model.
    Inputs:
        S0: initial underlying asset price.
        Strike: strike price of the option
        T: maturity date (in years)
        sigma: price volatility
        rf: risk-free rate
        q: dividend yield

        option: type of the option. 'call' or 'put'
    
    Output:
        option value at time 0
    '''

    S0 = float(S0)
    d1 = (log(S0 / Strike) + (rf + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S0 / Strike) + (rf - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    if option == 'call':       
        value = (S0 * stats.norm.cdf(d1, 0.0, 1.0)
                - Strike * exp(-rf * T) * stats.norm.cdf(d2, 0.0, 1.0))
    elif option == 'put':
        value = -(S0 * stats.norm.cdf(-d1, 0.0, 1.0)
                + Strike * exp(-rf * T) * stats.norm.cdf(-d2, 0.0, 1.0))
    else:
        print ('The option type could only be "put" or "call", please recall the'
               'function with right arguments')
    return value

#%%
if __name__ == '__main__':
    bi = binomial(100,100,1,0.2,0.05,1000)
    mcs = mcs(50000,100,100,1,0.2,0.05,1000)
    bsm = bsm(100,100,1,0.2,0.05)
    print ('Option value by'
           '\nBinomial Tree: %4.4f'
           '\nMonte Carlo Simulation: %4.4f'
           '\nBSM Model: %4.4f' % (bi, mcs, bsm))