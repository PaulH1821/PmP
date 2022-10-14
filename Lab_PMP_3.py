import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

model = pm.Model()


if __name__== '__main__ ':

    with model:

        cutremur = pm.Bernoulli('C', 0.0005)
        incendiu_p = pm.Deterministic(cutremur,0.03,0.01)
        incendiu = pm.Bernoulli('I',p=incendiu_p)
        alarma_accidental = pm.Deterministic('A_c',pm.math.switch(incendiu,pm.math.swich(cutremur,0.98,0.95),pm.math.switch(cutremur,0.02,0.0001)))
        alarma = pm.Bernoulli('A',p=alarma_accidental)
        trace = pm.sample(20000)


dictionary =  {
                'cutermur' : trace['C'].tolist(),
                'incendiu' :trace['I'].tolist(),
                'alarma' :trace['A'].tolist()
              }
df=pd.Dataframe(dictionary)

p_2 = df[((df['alarma'] == 1) & (df['cutremur'] == 1))].shape[0] / df[df['alarma'] == 1].shape[0]
print(p_2)