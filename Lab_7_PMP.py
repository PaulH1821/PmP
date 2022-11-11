import arviz as az
import matplotlib.pyplot as plt

import numpy as np
import pymc3 as pm
import pandas as  pd


f=pd.read_csv("prices.csv")

if __name__ == "__main__":

    f=pd.read_csv("prices.csv")


    price = f['Price'].values
    speed = f['Speed'].values
    hardDrive = f['HardDrive'].values
    ram = f['Ram'].values
    premium = f['Premium'].values



    fig, axes = plt.subplots(2, 2, sharex=False, figsize=(10, 8))
    axes[0,0].scatter(speed, price, alpha=0.6)
    axes[0,1].scatter(hardDrive, price, alpha=0.6)
    axes[1,0].scatter(ram, price, alpha=0.6)
    axes[1,1].scatter(premium, price, alpha=0.6)
    axes[0,0].set_ylabel("Price")
    axes[0,0].set_xlabel("Speed")
    axes[0,1].set_xlabel("HardDrive")
    axes[1,0].set_xlabel("Ram")
    axes[1,1].set_xlabel("Premium")
    plt.savefig('price_correlations.png')

    model = pm.Model()

    with model:
    
        a = pm.Normal('a',mu=0,sd=10 )

        bspeed=pm.Normal('bspeed', mu=0, sd=10)
        bhard=pm.Normal('bhard', mu=0, sd=10)

        sigma = pm.HalfNormal('sigma', sd=1)

        mu = pm.Deterministic('mu', a + bspeed * speed + bhard * np.log(hardDrive))

        price_like = pm.Normal('price_like', mu=mu, sd=sigma, observed = price)

        trace = pm.sample(200, tune=200, cores=4)

        a_mean = trace['a'].mean().item()
        bspeed_mean= trace['bspeed'].mean().item()
        bhard_mean= trace['bhard'].mean().item()

        

