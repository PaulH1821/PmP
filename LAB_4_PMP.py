from typing import NoReturn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as av

model = pm.Model()

if __name__== '__main__ ':
    
    with model:
        nr_clienti = pm.Poisson('nr',l=20)
        comanda = pm.Normal('c',m=1,s=0.5)
        statie = pm.Exponential('s',x=1/10)
        trace = pm.sample(2000)

