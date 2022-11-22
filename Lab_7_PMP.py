import math

import pymc3 as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az


df = pd.read_csv('Prices.csv')
price = np.array(df["Price"])
speed = np.array(df["Speed"])
hard_drive = np.log(np.array(df["HardDrive"]))
ram = np.array(df["Ram"])
premium = np.array([1 if x == True else 0 for x in df["Premium"]])


def Ex_1():
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=np.mean(price), sd=10)
        beta1 = pm.Normal("beta_1", mu=-5, sd=10)
        beta2 = pm.Normal("beta_2", mu=8, sd=12)
        sigma = pm.HalfCauchy("sigma", np.std(price))
        miu = alpha + beta1 * speed + beta2 * hard_drive

        y_pred = pm.Normal("y_pred", mu=miu, sd=sigma, observed=price)
        idata_g = pm.sample(2000, tune=2000, return_inferencedata=True)

    az.plot_trace(idata_g, var_names=["alpha", "beta_1", "beta_2", "sigma"])
    plt.show()

    return idata_g, model


def Ex_3():
    result = Ex_1()
    idata_g = result[0]
    model = result[1]

    ppc = pm.sample_posterior_predictive(idata_g, samples=100, model=model)
    correlation = az.r2_score(price, ppc["y_pred"])
    print(correlation)


def Ex_4_and_5():
    x1 = [33 for i in price]
    x2 = [np.log(540) for i in price]

    with pm.Model as model:
        alpha = pm.Normal("alpha", mu=np.mean(price), sd=2)
        beta1 = pm.Normal("beta_1", mu=-5, sd=10)
        beta2 = pm.Normal("beta_2", mu=8, sd=12)
        sigma = pm.HalfCauchy("sigma", np.std(price))
        miu = pm.Deterministic("miu", alpha + beta1 * x1 + beta2 * x2)

        y_pred = pm.Normal("y_pred", mu=miu, sd=sigma, observed=price)
        idata_g = pm.sample(5000, tune=5000, return_inferencedata=True)

    az.plot_trace(idata_g, var_names=["alpha", "beta_1", "beta_2", "sigma"])
    plt.show()




Ex_1()
# ex3()
# ex_4_and_5()

