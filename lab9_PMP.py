import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import random


def model(file, order):
    az.style.use('arviz-darkgrid')
    dummy_data = np.loadtxt(file)
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    x_1p = np.vstack([x_1**i for i in range(1, order+1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')

    with pm.Model() as model_l:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10)
        ε = pm.HalfNormal('ε', 5)
        μ = α + β * x_1s[0]
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_l = pm.sample(2000, return_inferencedata=True)

    with pm.Model() as model_p:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10, shape=order)
        # β = pm.Normal('β', mu=0, sd=100, shape=order) # distribuitie avand Beta cu sd = 100
        # β = pm.Normal('β', mu=0, sd=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)

    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)

    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.show()
    return idata_l, idata_p


def Ex_1():
    file = './date.csv'
    order = 5
    model(file, order)


def Ex_2():
    file = "./date2.csv"
    f = open(file, "a")
    for i in range(0, 500):
        rand1 = round(random.uniform(-1.99, 9.99), 3)
        rand2 = round(random.uniform(-1.99, 9.99), 3)
        f.write(str(rand1) + ' ')
        f.write(str(rand2) + '\n')
    f.close()

    order = 5
    model(file, order)


def Ex_3():
    file = './date.csv'
    order = 2
    idata_l, idata_p = model(file, order)

    waic_l = az.waic(idata_l, scale="deviance")
    loo_l = az.loo(idata_l, scale="deviance")

    cmp_df = az.compare({'model_l': idata_l, 'model_p': idata_p},
                        method='BB-pseudo-BMA', ic="waic", scale="deviance")

    print(cmp_df)


if __name__ == "__main__":
    Ex_1()