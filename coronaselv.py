# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:28:57 2020

@author: jakob
"""
#You have been Schauser-corrupted!!
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
import pandas as pd
from scipy import stats

area_cph = ["Copenhagen", "Frederiksberg", "Gentofte", "Dragør", "Vallensbæk", "Ballerup", "Gladsaxe", "Herlev", "Hvidovre", "Lyngby-Taarbæk", "Rødovre", "Brøndby", "Glostrup", "Høje-Taastrup", "Tårnby", "Ishøj", "Albertslund"]
data_posi = pd.read_csv("C:/Users/jakob/Documents/Blok1Troels/Municipality_cases_time_series.csv", sep=';', thousands='.', index_col=0)
data_test = pd.read_csv("C:/Users/jakob/Documents/Blok1Troels/Municipality_tested_persons_time_series.csv", sep=';', thousands='.', index_col=0)

Nposi_cph = data_posi.loc["2020-03-01":"2020-09-22"][area_cph].values.sum(axis=1)
Ntest_cph = data_test.loc["2020-03-01":"2020-09-22"][area_cph].values.sum(axis=1)[:-1]
eNtest_cph = np.sqrt(Ntest_cph)

Nposi_all = data_posi.loc["2020-03-01":"2020-09-22"].sum(axis=1)
Ntest_all = data_test.loc["2020-03-01":"2020-09-22"].sum(axis=1)
eNtest_all = np.sqrt(Ntest_all)


Nposi_cph*=1000
StartDay,EndDay = 125,len(Nposi_cph)
# Nposi_cph = (Nposi_cph/Ntest_cph)*100
plt.plot(Nposi_cph[StartDay:EndDay],'.')
plt.show()

day  = np.arange(0,EndDay)

# =============================================================================
# From Troels
# =============================================================================
Fit_StartDay = StartDay

def func_SIRmodel(x, dayLockdown, nI0, beta0, beta1, gamma) :

    # These numbers are the current status for each type of case, NOT cumulative.
    NdaysModelExtendsBeyondData = 5
    NdaysModel = len(Nposi_cph) + NdaysModelExtendsBeyondData
    mS   = np.zeros(NdaysModel)   # Susceptible
    mI   = np.zeros(NdaysModel)   # Infected
    mR   = np.zeros(NdaysModel)   # Recovered
    mTot = np.zeros(NdaysModel)   # Overall number (for check)

    # Initial numbers:
    dayN = Fit_StartDay
    mS[dayN]  = 5800000-nI0
    mI[dayN]  = nI0
    mR[dayN]  = 0
    mTot[dayN] = mS[dayN]+mI[dayN]     # There are no Recovered, yet!

    
    # Model loop:
    # -----------------
    while (dayN < len(Nposi_cph)-1 and mI[dayN] > 0) :
        dayN += 1
        if (dayN < dayLockdown) :     # Could potentially be a fitting parameter!
            beta = beta0
#        elif (dayN < 25) :
#            beta = beta1
        else :
            beta = beta1
            # beta = max(beta0 - dbeta*(dayN-dayLockdown), beta_inf)

        dI = beta*mI[dayN-1] * (mS[dayN-1] / mTot[dayN-1])
        dR = gamma*mI[dayN-1]
        mS[dayN]   = mS[dayN-1] - dI
        mI[dayN]   = mI[dayN-1] + dI - dR
        mR[dayN]   = mR[dayN-1] + dR
        mTot[dayN] = mS[dayN] + mI[dayN] + mR[dayN]       # Should remain constant!

    return mI[x]

func_SIRmodel_vec = np.vectorize(func_SIRmodel)

def func_SEIRmodel(x, dayLockdown, nI0, beta0, beta1, lambdaE, lambdaI) :

    # Initial numbers:
    N_tot = 5800000
    S  = N_tot - nI0

    # The initial number of exposed and infected are scaled to match beta0. Factors in front are ad hoc! 
    Norm = np.exp(0.8*lambdaE * beta0) + np.exp(0.7*lambdaE * beta0) + np.exp(0.6*lambdaE * beta0) + np.exp(0.5*lambdaE * beta0) +\
           np.exp(0.4*lambdaE * beta0) + np.exp(0.3*lambdaE * beta0) + np.exp(0.2*lambdaE * beta0) + np.exp(0.1*lambdaE * beta0)

    E1 = nI0 * np.exp(0.8*lambdaE * beta0) / Norm
    E2 = nI0 * np.exp(0.7*lambdaE * beta0) / Norm
    E3 = nI0 * np.exp(0.6*lambdaE * beta0) / Norm
    E4 = nI0 * np.exp(0.5*lambdaE * beta0) / Norm
    I1 = nI0 * np.exp(0.4*lambdaE * beta0) / Norm
    I2 = nI0 * np.exp(0.3*lambdaE * beta0) / Norm
    I3 = nI0 * np.exp(0.2*lambdaE * beta0) / Norm
    I4 = nI0 * np.exp(0.1*lambdaE * beta0) / Norm
    """
    E1 = nI0/8
    E2 = nI0/8
    E3 = nI0/8
    E4 = nI0/8
    I1 = nI0/8
    I2 = nI0/8
    I3 = nI0/8
    I4 = nI0/8
    """
    R  = 0
    Tot = S + E1+E2+E3+E4 + I1+I2+I3+I4 + R

    # We define the first day given in the array of days (x) as day 0:
    dayN = Fit_StartDay
    NdaysModelExtendsBeyondData = 5
    NdaysModel = len(nPos_aarhus) + NdaysModelExtendsBeyondData
    # x_trans = x - x[0]       # Translate x by the number of days to the start of the fit/plot
    
    # We store the results (time, S, E, I, and R) for each day here:
    SEIR_result = np.zeros((NdaysModel, 5))
    SEIR_result[dayN] = [dayN, S, E1+E2+E3+E4, I1+I2+I3+I4, R]

    # Numerical settings:
    nStepsPerDay = 24           # We simulated in time steps of 1 hour
    dt = 1.0 / nStepsPerDay     # Time step length in days

    
    # Model loop:
    # -----------------
    while (dayN < NdaysModel-1 and I1 >= 0) :

        dayN += 1
        if (dayN < dayLockdown) :
            beta = beta0
        else :
            beta = beta1
            # beta = max(beta0 - dbeta*(dayN-dayLockdown), beta_inf)   # Minimum value of beta (at time infinity)

        # Now divide the daily procedure into time steps:
        for i in range(nStepsPerDay) :
            dS = -beta*(I1+I2+I3+I4) * (S / N_tot)
            dE1 = -dS          - lambdaE * E1
            dE2 = lambdaE * E1 - lambdaE * E2
            dE3 = lambdaE * E2 - lambdaE * E3
            dE4 = lambdaE * E3 - lambdaE * E4
            dI1 = lambdaE * E4 - lambdaI * I1
            dI2 = lambdaI * I1 - lambdaI * I2
            dI3 = lambdaI * I2 - lambdaI * I3
            dI4 = lambdaI * I3 - lambdaI * I4
            dR  = lambdaI * I4
            
            S  += dt * dS
            E1 += dt * dE1
            E2 += dt * dE2
            E3 += dt * dE3
            E4 += dt * dE4
            I1 += dt * dI1
            I2 += dt * dI2
            I3 += dt * dI3
            I4 += dt * dI4
            R  += dt * dR
            Tot = S + E1+E2+E3+E4 + I1+I2+I3+I4 + R

        # Record status of model every day:
        SEIR_result[dayN] = [dayN, S, E1+E2+E3+E4, I1+I2+I3+I4, R]

        # print(dayN, E1+E2+E3+E4, I1+I2+I3+I4, R)

    # Return only the number of infected and only for the relevant days:
    return SEIR_result[x,3]

func_SEIRmodel_vec = np.vectorize(func_SEIRmodel)
dayLockdown = 190



Fit_EndDay,Fit_StartDay = EndDay,StartDay
fig, ax = plt.subplots(figsize=(12, 7))
ax.set(xlabel="Day (1st of March is day 0)", ylabel="Newly infected / day", title="")
ax.errorbar(day[StartDay:EndDay], Nposi_cph[StartDay:EndDay], yerr=np.sqrt(Nposi_cph[StartDay:EndDay]), fmt='.', linewidth=2, label='Data (scaled)', color='red')


nI0_plot = 2.0
beta0_plot = 0.34
beta1_plot = 0.06
gamma_plot = 1.0/7.0

# Calculate Chi2, Ndof, and Chi2-Probability for this model:
nPos_est = func_SIRmodel(day[Fit_StartDay:Fit_EndDay], dayLockdown, nI0_plot, beta0_plot, beta1_plot, gamma_plot)
chi2 = np.sum(((Nposi_cph[Fit_StartDay:Fit_EndDay] - nPos_est) / Nposi_cph[Fit_StartDay:Fit_EndDay])**2)
Ndof = len(day[Fit_StartDay:Fit_EndDay]) - 5
Prob = stats.chi2.sf(chi2, Ndof)

def ChiSquareCalcSIR(dayL, nI0, beta0, beta1, gamma) :
        nPos_est = func_SIRmodel(day[Fit_StartDay:Fit_EndDay], dayL, nI0, beta0, beta1, gamma)
        chi2 = np.sum(((Nposi_cph[Fit_StartDay:Fit_EndDay] - nPos_est) / np.sqrt(Nposi_cph[Fit_StartDay:Fit_EndDay]))**2)
        return chi2

minuit_SIR = Minuit(ChiSquareCalcSIR, pedantic=False, print_level=0, dayL=dayLockdown, nI0=2.0, beta0=0.45, beta1=0.04, gamma=0.14, fix_gamma=True)
minuit_SIR.migrad();
if (not minuit_SIR.fmin.is_valid) :
    print("  WARNING: The ChiSquare fit DID NOT converge!!! ")

dayL_fit, nI0_fit, beta0_fit, beta1_fit, gamma_fit = minuit_SIR.values.values()       # Same as minuit_SIR.args
edayL_fit, enI0_fit, ebeta0_fit, ebeta1_fit, egamma_fit = minuit_SIR.errors.values()



ax.plot(day[Fit_StartDay:Fit_EndDay], func_SIRmodel_vec(day[Fit_StartDay:Fit_EndDay], dayLockdown, nI0_plot, beta0_plot, beta1_plot, gamma_plot), 'blue', linewidth=1.0, label='SIR Model')

print(f"  SIR Model (fixed for plot):  Prob(Chi2={chi2:6.1f}, Ndof={Ndof:3d}) = {Prob:7.5f}")
   

