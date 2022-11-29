import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

df = pd.read_csv('Admission.csv')
admis = np.array(df['Admission'])
gre = np.array(df['GRE'])
gpa = np.array(df['GPA'])
y_0 = pd.Categorical(df['Admission']).codes
x_n = ['GRE', 'GPA']
x_1 = df[x_n].values


def Ex1():
    with pm.Model() as model_1:
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta = pm.Normal('beta', mu=0, sd=2, shape=len(x_n))
        miu = alpha + pm.math.dot(x_1, beta)
        teta = pm.Deterministic('teta', 1 / (1 + pm.math.exp(-miu)))
        bd = pm.Deterministic('bd', -alpha / beta[1] - beta[0] / beta[1] * x_1[:, 0])
        yl = pm.Bernoulli('yl', p=teta, observed=admis)
        idata_1 = pm.sample(2000, target_accept=0.9, return_inferencedata=True)

    az.plot_trace(idata_1, var_names=["alpha", "beta", "teta"])
    plt.show()
    return idata_1, model_1

def Ex2():
    idata_1, model = Ex1()
    idx = np.argsort(x_1[:, 0])
    bd = idata_1.posterior['bd'].mean(("GRE", "GPA"))[idx]
    plt.scatter(x_1[:, 0], x_1[:, 1], c=[f'C{x}' for x in y_0])
    plt.plot(x_1[:, 0][idx], bd, color='k');
    az.plot_hdi(x_1[:, 0], idata_1.posterior['bd'], color='k')
    plt.xlabel(x_n[0])
    plt.ylabel(x_n[1])
    plt.show()

Ex1()
