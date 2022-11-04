from multiprocessing.dummy import freeze_support
import pandas as  pd
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm


f=pd.read_csv("data.csv")

if __name__== '__main__ ':

    t=f["ppvt"] 
    age=f["momage"]

    basic_model=pm.Model()



    with pm.Model() as model_g:
        alpha=pm.Normal('a',mu=0, sd = 10)
        beta=pm.Normal('b', mu=0 , sd = 1)
        e=pm.HalfCauchy('e',5)
        u=pm.Deterministic('u',alpha + beta * age)
        y_pred=pm.Normal('y_pred', mu=u, sd=e, observed = t)

        idata_g=pm.sample(200, tune=200, return_inferencedata=True )


        plt.plot_trace( idata_g )
        plt.show()
